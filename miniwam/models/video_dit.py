"""
video_dit.py

Direct native-feature DiT expert for MiniWAM ablations.

Structure adapted from FastWAM's video expert, built from the MiniWAM DiT/MoT
components. Inputs and outputs are channel-first feature grids:
    [B, C, T, H, W]
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .dit_block import (
    DiTBlock,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)


class VideoHead(nn.Module):
    """Final AdaLN projection from video tokens back to native feature patches."""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        patch_size: Tuple[int, int, int],
        eps: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.patch_size = tuple(int(x) for x in patch_size)
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_channels * math.prod(self.patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 3:
            mod = self.modulation.unsqueeze(0).to(dtype=t.dtype, device=t.device)
            shift, scale = (mod + t.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
            return self.proj(self.norm(tokens) * (1 + scale) + shift)

        shift, scale = (
            self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)
        ).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(tokens) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class VideoDiT(nn.Module):
    """
    Wan-style DiT expert for denoising native feature videos.

    The first latent frame is a clean conditioning frame. `pre_dit` assigns it a
    zero diffusion timestep and assigns all future latent frames the sampled
    video timestep.

    By default the current frame shares the video patch embedding with the
    future frames. With `frame0_patch_size=(1, 1)` it is instead projected one
    token per feature patch, as the PRISM world expert projects DINOv3 current
    features; the future frames keep `patch_size`.
    """

    def __init__(
        self,
        hidden_dim: int = 1536,
        in_channels: int = 16,
        out_channels: int = 16,
        ffn_dim: int = 6144,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_heads: int = 12,
        attn_head_dim: int = 128,
        num_layers: int = 12,
        video_attention_mask_mode: str = "first_frame_causal",
        use_gradient_checkpointing: bool = False,
        rope_table_size: int = 1024,
        frame0_patch_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.patch_size = tuple(int(x) for x in patch_size)
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.video_attention_mask_mode = str(video_attention_mask_mode)
        self.use_gradient_checkpointing = use_gradient_checkpointing

        if len(self.patch_size) != 3:
            raise ValueError(f"`patch_size` must have 3 values, got {self.patch_size}")
        if self.patch_size[0] != 1:
            raise ValueError(
                "Direct-feature ablation requires temporal patch size 1 so the clean "
                f"current feature frame remains separable, got {self.patch_size}"
            )
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.patch_embedding = nn.Conv3d(
            in_channels,
            hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                attn_head_dim=attn_head_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                eps=eps,
            )
            for _ in range(num_layers)
        ])
        self.head = VideoHead(hidden_dim, out_channels, self.patch_size, eps)
        self.freqs = precompute_freqs_cis_3d(attn_head_dim, end=rope_table_size)

        self.frame0_patch_size = (
            None if frame0_patch_size is None else tuple(int(x) for x in frame0_patch_size)
        )
        self.frame0_projection = None
        if self.frame0_patch_size is not None:
            if self.frame0_patch_size != (1, 1):
                raise ValueError(
                    "`frame0_patch_size` must be [1, 1] (one token per current feature "
                    f"patch) or unset, got {list(self.frame0_patch_size)}."
                )
            if self.video_attention_mask_mode == "per_frame_causal":
                raise ValueError(
                    "Per-patch current-frame tokens support first_frame_causal or "
                    "bidirectional video attention, not per_frame_causal."
                )
            # Created under a forked RNG so every other parameter keeps the
            # initialization it has without this projection.
            with torch.random.fork_rng(devices=[]):
                self.frame0_projection = nn.Linear(in_channels, hidden_dim)

    def _validate_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"`x` must be [B, C, T, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"`x` channel dim must be {self.in_channels}, got {x.shape[1]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be [B, L, D], got {tuple(context.shape)}")

        batch_size = x.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between video latents and context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        pt, ph, pw = self.patch_size
        if x.shape[2] % pt != 0 or x.shape[3] % ph != 0 or x.shape[4] % pw != 0:
            raise ValueError(
                "Video latent shape must be divisible by patch size, got "
                f"T/H/W={tuple(x.shape[2:])}, patch={self.patch_size}"
            )

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        elif context_mask.ndim != 2:
            raise ValueError(f"`context_mask` must be [B, L], got {tuple(context_mask.shape)}")

        return x, timestep, context_mask.to(device=context.device, dtype=torch.bool)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)

    def unpatchify(self, tokens: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        return rearrange(
            tokens,
            "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
            t=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            pt=pt,
            ph=ph,
            pw=pw,
            c=self.out_channels,
        )

    def _build_3d_rope_freqs(
        self,
        grid_size: Tuple[int, int, int],
        device: torch.device,
        w_offset: int = 0,
        t_offset: int = 0,
    ) -> torch.Tensor:
        t, h, w = grid_size
        f_freqs, h_freqs, w_freqs = self.freqs
        if w_offset < 0:
            raise ValueError(f"RoPE width offset must be non-negative, got {w_offset}.")
        if w_offset + w > w_freqs.shape[0]:
            raise ValueError(
                f"Tiled RoPE width overflow: w_offset={w_offset} + w={w} = {w_offset + w} "
                f"exceeds precomputed table size {w_freqs.shape[0]}. "
                "Increase `rope_table_size` or reduce the camera slot count/stride."
            )
        freqs = torch.cat([
            f_freqs[t_offset:t_offset + t].view(t, 1, 1, -1).expand(t, h, w, -1),
            h_freqs[:h].view(1, h, 1, -1).expand(t, h, w, -1),
            w_freqs[w_offset:w_offset + w].view(1, 1, w, -1).expand(t, h, w, -1),
        ], dim=-1).reshape(t * h * w, 1, -1).to(device)
        return freqs

    def rope_freqs(
        self, meta: Dict[str, Any], device: torch.device, w_offset: int = 0,
    ) -> torch.Tensor:
        """RoPE frequencies for one camera's video tokens, in `pre_dit` token order."""
        if self.frame0_projection is None:
            return self._build_3d_rope_freqs(meta["grid_size"], device, w_offset=w_offset)
        # Current tokens sit at t=0 on their own patch grid, like PRISM's z0 tokens;
        # future tokens keep their grid at t=1, 2, ...
        current = self._build_3d_rope_freqs(meta["current_grid_size"], device, w_offset=w_offset)
        future = self._build_3d_rope_freqs(meta["grid_size"], device, w_offset=w_offset, t_offset=1)
        return torch.cat([current, future], dim=0)

    def build_video_to_video_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        mode: Optional[str] = None,
        first_frame_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        if video_seq_len <= 0:
            raise ValueError(f"`video_seq_len` must be positive, got {video_seq_len}")
        if video_tokens_per_frame <= 0:
            raise ValueError(
                f"`video_tokens_per_frame` must be positive, got {video_tokens_per_frame}"
            )

        mode = self.video_attention_mask_mode if mode is None else str(mode)
        if mode == "bidirectional":
            return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)

        if video_seq_len % video_tokens_per_frame != 0:
            raise ValueError(
                "`video_seq_len` must be divisible by `video_tokens_per_frame`, got "
                f"{video_seq_len} and {video_tokens_per_frame}"
            )
        num_frames = video_seq_len // video_tokens_per_frame

        if mode == "per_frame_causal":
            frame_mask = torch.tril(
                torch.ones((num_frames, num_frames), dtype=torch.bool, device=device)
            )
            return frame_mask.repeat_interleave(video_tokens_per_frame, dim=0).repeat_interleave(
                video_tokens_per_frame, dim=1
            )

        if mode == "first_frame_causal":
            mask = torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)
            if first_frame_tokens is None:
                first_frame_tokens = video_tokens_per_frame
            first_frame_tokens = min(first_frame_tokens, video_seq_len)
            mask[:first_frame_tokens, first_frame_tokens:] = False
            return mask

        raise ValueError(f"Unsupported video attention mask mode: {mode}")

    def pre_dit(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        x, timestep, context_mask = self._validate_inputs(x, timestep, context, context_mask)
        if self.frame0_projection is not None:
            return self._pre_dit_per_patch_current(x, timestep, context, context_mask)
        batch_size = x.shape[0]

        patched = self.patchify(x)  # [B, D, T, h, w]
        t_grid, h_grid, w_grid = patched.shape[2:]
        tokens_per_frame = h_grid * w_grid
        seq_len = t_grid * tokens_per_frame

        token_timesteps = torch.empty(
            (batch_size, t_grid, tokens_per_frame),
            dtype=timestep.dtype,
            device=timestep.device,
        )
        token_timesteps[:] = timestep.view(batch_size, 1, 1)
        token_timesteps[:, 0, :] = 0
        token_timesteps = token_timesteps.reshape(batch_size, seq_len)

        t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(t_emb).reshape(batch_size, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        tokens = rearrange(patched, "b d t h w -> b (t h w) d").contiguous()
        freqs = self._build_3d_rope_freqs((t_grid, h_grid, w_grid), tokens.device)
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "grid_size": (t_grid, h_grid, w_grid),
                "tokens_per_frame": tokens_per_frame,
                "num_current_tokens": tokens_per_frame,
                "seq_len": seq_len,
            },
        }

    def _pre_dit_per_patch_current(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """`pre_dit` with one token per current feature patch and patched future frames."""
        batch_size, _, _, height, width = x.shape
        current_tokens = self.frame0_projection(
            rearrange(x[:, :, :1], "b c t h w -> b (t h w) c")
        )
        patched = self.patchify(x[:, :, 1:])  # [B, D, T-1, h, w]
        t_grid, h_grid, w_grid = patched.shape[2:]
        tokens_per_frame = h_grid * w_grid
        num_current = current_tokens.shape[1]
        seq_len = num_current + t_grid * tokens_per_frame

        token_timesteps = timestep.view(batch_size, 1).repeat(1, seq_len)
        token_timesteps[:, :num_current] = 0

        t_emb = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
        t = self.time_embedding(t_emb).reshape(batch_size, seq_len, self.hidden_dim)
        t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))

        tokens = torch.cat(
            [current_tokens, rearrange(patched, "b d t h w -> b (t h w) d")], dim=1,
        ).contiguous()
        meta = {
            "batch_size": batch_size,
            "grid_size": (t_grid, h_grid, w_grid),
            "current_grid_size": (1, height, width),
            "tokens_per_frame": tokens_per_frame,
            "num_current_tokens": num_current,
            "seq_len": seq_len,
        }
        return {
            "tokens": tokens,
            "freqs": self.rope_freqs(meta, tokens.device),
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_mask.unsqueeze(1).expand(-1, seq_len, -1),
            "meta": meta,
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        if self.frame0_projection is None:
            patch_tokens = self.head(tokens, pre_state["t"])
            return self.unpatchify(patch_tokens, pre_state["meta"]["grid_size"])
        # Only future tokens are decoded; the clean current frame comes back as zeros,
        # which training and inference discard as they do the default prediction.
        meta = pre_state["meta"]
        num_current = int(meta["num_current_tokens"])
        future = self.unpatchify(
            self.head(tokens[:, num_current:], pre_state["t"][:, num_current:]),
            meta["grid_size"],
        )
        _, height, width = meta["current_grid_size"]
        current = future.new_zeros((future.shape[0], future.shape[1], 1, height, width))
        return torch.cat([current, future], dim=2)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pre_state = self.pre_dit(x, timestep, context, context_mask)
        tokens = pre_state["tokens"]
        video_mask = self.build_video_to_video_mask(
            video_seq_len=tokens.shape[1],
            video_tokens_per_frame=int(pre_state["meta"]["tokens_per_frame"]),
            device=tokens.device,
            first_frame_tokens=int(pre_state["meta"]["num_current_tokens"]),
        )

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                from .dit_block import gradient_checkpoint_forward

                tokens = gradient_checkpoint_forward(
                    block,
                    self.use_gradient_checkpointing,
                    tokens,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    context_mask=pre_state["context_mask"],
                    self_attn_mask=video_mask,
                )
            else:
                tokens = block(
                    tokens,
                    pre_state["context"],
                    pre_state["t_mod"],
                    pre_state["freqs"],
                    context_mask=pre_state["context_mask"],
                    self_attn_mask=video_mask,
                )

        return self.post_dit(tokens, pre_state)
