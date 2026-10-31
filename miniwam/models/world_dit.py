"""
world_dit.py

World DiT expert for MiniWAM.
Denoises compact future world tokens using 3D RoPE (since tokens come from a 3D grid).
Structure adapted from FastWAM's ActionDiT, with 3D positional encoding.

Supports prepending frame-0 visual tokens (z0_tokens) with a separate timestep
(z0 → timestep=0, L → sampled timestep).
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .dit_block import (
    DiTBlock,
    precompute_freqs_cis,
    precompute_freqs_cis_3d,
    sinusoidal_embedding_1d,
)


class WorldHead(nn.Module):
    """Final projection head with AdaLN modulation for latent prediction."""

    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D]
            t_mod: [B, D] (global) or [B, S, D] (per-token)
        """
        if t_mod.ndim == 2:
            # Global t_mod: [B, D] → broadcast
            shift, scale = (
                self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)
            ).chunk(2, dim=1)
            shift = shift.squeeze(1)
            scale = scale.squeeze(1)
            return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
        else:
            # Per-token t_mod: [B, S, D]
            mod = self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device)
            # mod is [1, 1, 2, D], t_mod is [B, S, D] → expand
            shift, scale = (mod + t_mod.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)  # [B, S, D]
            scale = scale.squeeze(2)  # [B, S, D]
            return self.proj(self.norm(x) * (1 + scale) + shift)


class WorldDiT(nn.Module):
    """
    Wan-style DiT expert for denoising compact world tokens.
    Uses 3D RoPE since tokens represent a spatiotemporal grid from E_φ.

    Supports optional z0_tokens (patchified frame-0 VAE features) prepended
    to the world-token sequence with separated timestep.

    Input:  noisy world tokens [B, num_tokens, latent_dim]
    Output: predicted velocity   [B, num_tokens, latent_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 1536,
        latent_dim: int = 64,
        ffn_dim: int = 6144,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 12,
        attn_head_dim: int = 128,
        num_layers: int = 12,
        use_gradient_checkpointing: bool = False,
        use_3d_rope: bool = True,
        # Frame-0 patchification params
        vae_channels: int = 16,
        frame0_patch_size: Tuple[int, int] = (2, 2),
        visual_encoder_type: str = "wan_vae",
        current_feature_dim: int = 768,
        current_patch_grid: Tuple[int, int] = (14, 14),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.use_3d_rope = use_3d_rope
        self.frame0_patch_size = frame0_patch_size
        self.visual_encoder_type = str(visual_encoder_type).lower()
        if self.visual_encoder_type not in ("wan_vae", "dinov3"):
            raise ValueError(
                f"visual_encoder_type must be 'wan_vae' or 'dinov3', got {visual_encoder_type!r}."
            )
        self.current_feature_dim = int(current_feature_dim)
        self.current_patch_grid = tuple(int(x) for x in current_patch_grid)

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        # Input projection (L tokens: latent_dim → hidden_dim)
        self.latent_encoder = nn.Linear(latent_dim, hidden_dim)

        # Frame-0 patchification (z0: VAE latents → hidden_dim tokens)
        # Conv3d with temporal kernel=1 (single frame)
        ph, pw = frame0_patch_size
        if self.visual_encoder_type == "wan_vae":
            self.frame0_patcher = nn.Conv3d(
                vae_channels, hidden_dim,
                kernel_size=(1, ph, pw), stride=(1, ph, pw),
            )
            self.frame0_projection = None
        else:
            self.frame0_patcher = None
            self.frame0_projection = nn.Linear(self.current_feature_dim, hidden_dim)

        # Conditioning embeddings
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
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

        # Transformer blocks
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

        # Output head
        self.head = WorldHead(hidden_dim, latent_dim, eps)

        # RoPE frequencies
        if use_3d_rope:
            self.freqs = precompute_freqs_cis_3d(attn_head_dim, end=128)
        else:
            self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    def patchify_frame0(self, z_t: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        Patchify frame-0 VAE latent into tokens.

        Args:
            z_t: [B, C, 1, H, W] or [B, C, H, W] current frame VAE latent

        Returns:
            z0_tokens: [B, h*w, hidden_dim] where h = H/ph, w = W/pw
            z0_grid_shape: (h, w) spatial grid dimensions for 3D RoPE
        """
        if self.visual_encoder_type == "dinov3":
            if z_t.ndim == 4 and z_t.shape[1] == 1:
                z_t = z_t[:, 0]
            if z_t.ndim != 3 or z_t.shape[-1] != self.current_feature_dim:
                raise ValueError(
                    f"DINO current features must be [B,N,{self.current_feature_dim}], "
                    f"got {tuple(z_t.shape)}."
                )
            h, w = self.current_patch_grid
            if z_t.shape[1] != h * w:
                raise ValueError(
                    f"DINO current patch count must be {h*w} for grid {(h,w)}, got {z_t.shape[1]}."
                )
            return self.frame0_projection(z_t), (h, w)
        if z_t.ndim == 4:
            z_t = z_t.unsqueeze(2)  # [B, C, H, W] → [B, C, 1, H, W]
        if z_t.ndim != 5 or z_t.shape[2] != 1:
            raise ValueError(
                f"`z_t` must be [B, C, 1, H, W] or [B, C, H, W], got {tuple(z_t.shape)}"
            )
        x = self.frame0_patcher(z_t)  # [B, hidden_dim, 1, h, w]
        h, w = x.shape[3], x.shape[4]
        x = rearrange(x, "b d 1 h w -> b (h w) d")
        return x, (h, w)

    def _build_3d_rope_freqs(
        self, grid_shape: tuple, device: torch.device, w_offset: int = 0,
    ) -> torch.Tensor:
        """Build 3D RoPE frequencies from (t, h, w) grid shape.

        Args:
            grid_shape: (t, h, w) spatiotemporal grid dimensions.
            device: target device.
            w_offset: horizontal offset for tiled camera layouts. Width positions
                      are shifted by w_offset so each camera slot occupies a
                      distinct band in the RoPE frequency table.
        """
        t, h, w = grid_shape
        f_freqs, h_freqs, w_freqs = self.freqs
        if w_offset < 0:
            raise ValueError(f"RoPE width offset must be non-negative, got {w_offset}.")
        if w_offset + w > w_freqs.shape[0]:
            raise ValueError(
                f"Tiled RoPE width overflow: w_offset={w_offset} + w={w} = {w_offset + w} "
                f"exceeds precomputed table size {w_freqs.shape[0]}. "
                "Increase the `end` parameter in precompute_freqs_cis_3d or reduce slot count."
            )
        freqs = torch.cat([
            f_freqs[:t].view(t, 1, 1, -1).expand(t, h, w, -1),
            h_freqs[:h].view(1, h, 1, -1).expand(t, h, w, -1),
            w_freqs[w_offset:w_offset + w].view(1, 1, w, -1).expand(t, h, w, -1),
        ], dim=-1).reshape(t * h * w, 1, -1).to(device)
        return freqs

    def _build_z0_rope_freqs(
        self, num_z0_tokens: int, device: torch.device,
        z0_grid_shape: tuple[int, int] | None = None,
        w_offset: int = 0,
    ) -> torch.Tensor:
        """Build RoPE frequencies for z0 tokens (spatial-only, t=0).

        z0 tokens represent a single frame's spatial patches at temporal
        position 0. We assign them 3D RoPE with t=0, varying h/w.

        Args:
            num_z0_tokens: total number of z0 tokens (h * w)
            device: target device
            z0_grid_shape: optional (h, w) shape of the z0 spatial grid.
                           If None, attempts square factorization; falls back to 1D RoPE.
            w_offset: horizontal offset for tiled camera layouts.
        """
        if self.use_3d_rope:
            import math
            if z0_grid_shape is not None:
                h, w = z0_grid_shape
            else:
                # Try square factorization
                side = int(math.isqrt(num_z0_tokens))
                if side * side == num_z0_tokens:
                    h, w = side, side
                else:
                    if w_offset:
                        raise ValueError(
                            "Tiled z0 RoPE requires z0_grid_shape for non-square token grids."
                        )
                    # Non-square grids without a tiled offset use 1-D RoPE.
                    freqs_1d = precompute_freqs_cis(self.attn_head_dim, end=max(num_z0_tokens + 1, 128))
                    return freqs_1d[:num_z0_tokens].view(num_z0_tokens, 1, -1).to(device)
            f_freqs, h_freqs, w_freqs = self.freqs
            if w_offset < 0:
                raise ValueError(f"RoPE width offset must be non-negative, got {w_offset}.")
            if w_offset + w > w_freqs.shape[0]:
                raise ValueError(
                    f"Tiled RoPE width overflow: w_offset={w_offset} + w={w} = {w_offset + w} "
                    f"exceeds precomputed table size {w_freqs.shape[0]}. "
                    "Increase the `end` parameter in precompute_freqs_cis_3d or reduce slot count."
                )
            # t=0 for all z0 tokens
            freqs = torch.cat([
                f_freqs[0:1].expand(1, h, w, -1),
                h_freqs[:h].view(1, h, 1, -1).expand(1, h, w, -1),
                w_freqs[w_offset:w_offset + w].view(1, 1, w, -1).expand(1, h, w, -1),
            ], dim=-1).reshape(h * w, 1, -1).to(device)
            return freqs
        else:
            if w_offset:
                raise ValueError("Tiled camera RoPE requires use_3d_rope=true.")
            freqs_1d = precompute_freqs_cis(self.attn_head_dim, end=max(num_z0_tokens + 1, 128))
            return freqs_1d[:num_z0_tokens].view(num_z0_tokens, 1, -1).to(device)


    def pre_dit(
        self,
        world_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        grid_shape: Optional[tuple] = None,
        z0_tokens: Optional[torch.Tensor] = None,
        z0_grid_shape: Optional[Tuple[int, int]] = None,
        z0_freqs: Optional[torch.Tensor] = None,
        latent_freqs: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process inputs before DiT blocks (for MoT integration).

        Args:
            world_tokens: [B, num_tokens, latent_dim] noisy world tokens
            timestep:      [B] diffusion timestep for L tokens
            context:       [B, L, hidden_dim] pre-projected context (text only)
            context_mask:  [B, L] boolean mask
            grid_shape:    (t, h, w) original grid shape for 3D RoPE
            z0_tokens:     [B, S_z0, hidden_dim] optional patchified frame-0 tokens
                          (already in hidden_dim space from frame0_patcher)
            z0_grid_shape: optional (h, w) spatial shape of z0 tokens for 3D RoPE
        """
        if world_tokens.ndim != 3:
            raise ValueError(
                f"`world_tokens` must be 3D [B, T, D], got shape {tuple(world_tokens.shape)}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")

        batch_size = world_tokens.shape[0]
        seq_len_L = world_tokens.shape[1]

        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )

        # Project L tokens: latent_dim -> hidden_dim
        L_projected = self.latent_encoder(world_tokens)  # [B, S_L, hidden]

        # Build per-token timestep and modulation
        if z0_tokens is not None:
            # Separated timestep: z0 → timestep=0, L → sampled timestep
            S_z0 = z0_tokens.shape[1]
            total_seq = S_z0 + seq_len_L

            # Per-token timestep: [B, S_z0 + S_L]
            token_timesteps = torch.zeros(
                (batch_size, total_seq),
                dtype=timestep.dtype, device=timestep.device,
            )
            token_timesteps[:, S_z0:] = timestep.unsqueeze(1).expand(-1, seq_len_L)

            # Per-token time embedding: [B * total_seq] → [B, total_seq, hidden]
            t_emb_flat = sinusoidal_embedding_1d(self.freq_dim, token_timesteps.reshape(-1))
            t = self.time_embedding(t_emb_flat).reshape(batch_size, total_seq, self.hidden_dim)
            t_mod = self.time_projection(t).unflatten(2, (6, self.hidden_dim))
            # t_mod: [B, total_seq, 6, hidden_dim] — per-token modulation

            # Concatenate tokens: [z0_tokens, L_projected]
            tokens = torch.cat([z0_tokens, L_projected], dim=1)

            # Build RoPE: [z0_freqs, L_freqs]
            if z0_freqs is None:
                z0_freqs = self._build_z0_rope_freqs(S_z0, tokens.device, z0_grid_shape=z0_grid_shape)
            else:
                z0_freqs = z0_freqs.to(tokens.device)
            if seq_len_L == 0:
                L_freqs = z0_freqs.new_empty((0, 1, z0_freqs.shape[-1]))
            elif latent_freqs is not None:
                L_freqs = latent_freqs.to(tokens.device)
            elif self.use_3d_rope and grid_shape is not None:
                L_freqs = self._build_3d_rope_freqs(grid_shape, tokens.device)
            else:
                L_freqs_1d = precompute_freqs_cis(self.attn_head_dim, end=max(seq_len_L + 1, 128))
                L_freqs = L_freqs_1d[:seq_len_L].view(seq_len_L, 1, -1).to(tokens.device)
            freqs = torch.cat([z0_freqs, L_freqs], dim=0)

        else:
            # No z0 — standard single-timestep mode
            S_z0 = 0
            total_seq = seq_len_L
            tokens = L_projected

            t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
            t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

            if latent_freqs is not None:
                freqs = latent_freqs.to(tokens.device)
            elif self.use_3d_rope and grid_shape is not None:
                freqs = self._build_3d_rope_freqs(grid_shape, tokens.device)
            else:
                freqs_1d = precompute_freqs_cis(self.attn_head_dim, end=max(seq_len_L + 1, 128))
                freqs = freqs_1d[:seq_len_L].view(seq_len_L, 1, -1).to(tokens.device)

        # Context for cross-attention
        context_emb = context
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, total_seq, -1)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": total_seq,
                "seq_len_L": seq_len_L,
                "S_z0": S_z0,
                "grid_shape": grid_shape,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """
        Project transformer output back to latent space.
        If z0 was prepended, strip those tokens before projection.
        """
        S_z0 = pre_state["meta"]["S_z0"]
        L_tokens = tokens[:, S_z0:, :]  # Strip z0 tokens
        return self.head(L_tokens, pre_state["t"] if pre_state["t"].ndim == 2 else pre_state["t"][:, S_z0:, :])

    def forward(
        self,
        world_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        grid_shape: Optional[tuple] = None,
        z0_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standalone forward (without MoT). For testing / single-expert use."""
        pre_state = self.pre_dit(world_tokens, timestep, context, context_mask, grid_shape, z0_tokens)
        x = pre_state["tokens"]
        ctx = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        ctx_mask = pre_state["context_mask"]

        for block in self.blocks:
            if self.use_gradient_checkpointing:
                from .dit_block import gradient_checkpoint_forward
                x = gradient_checkpoint_forward(
                    block, self.use_gradient_checkpointing,
                    x, ctx, t_mod, freqs, context_mask=ctx_mask,
                )
            else:
                x = block(x, ctx, t_mod, freqs, context_mask=ctx_mask)

        return self.post_dit(x, pre_state)
