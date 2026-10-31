"""
action_dit.py

Action DiT expert for MiniWAM.
Denoises action sequences using 1D RoPE (temporal sequence).
Adapted from FastWAM's ActionDiT with configurable dimensions.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .dit_block import (
    DiTBlock,
    precompute_freqs_cis,
    sinusoidal_embedding_1d,
)


class ActionHead(nn.Module):
    """Final projection head with AdaLN modulation for action prediction."""

    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (
            self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)
        ).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    """
    Wan-style DiT expert for denoising action sequences.
    Uses 1D RoPE since action tokens form a temporal sequence.

    Input:  noisy action tokens [B, Ta, action_dim]
    Output: predicted velocity   [B, Ta, action_dim]
    """

    def __init__(
        self,
        hidden_dim: int = 1536,
        action_dim: int = 7,
        ffn_dim: int = 6144,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 12,
        attn_head_dim: int = 128,
        num_layers: int = 12,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim

        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        # Input projection
        self.action_encoder = nn.Linear(action_dim, hidden_dim)

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
        self.head = ActionHead(hidden_dim, action_dim, eps)

        # 1D RoPE frequencies
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Pre-process inputs before DiT blocks (for MoT integration).

        Args:
            action_tokens: [B, Ta, action_dim] noisy action tokens
            timestep:      [B] diffusion timestep
            context:       [B, L, text_dim] text embeddings
            context_mask:  [B, L] boolean mask
        """
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, D], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got shape {tuple(timestep.shape)}")
        if context.ndim != 3:
            raise ValueError(f"`context` must be 3D [B, L, D], got shape {tuple(context.shape)}")

        batch_size = action_tokens.shape[0]
        seq_len = action_tokens.shape[1]

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

        # Time embedding
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        # Token embedding
        tokens = self.action_encoder(action_tokens)

        # Context — already projected by ContextEncoder to hidden_dim
        # (text_embedding is only used in standalone forward, not via MoT)
        context_emb = context
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)

        # 1D RoPE
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context_emb,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        """Project transformer output back to action space."""
        return self.head(tokens, pre_state["t"])

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standalone forward (without MoT). For testing / single-expert use."""
        pre_state = self.pre_dit(action_tokens, timestep, context, context_mask)
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
