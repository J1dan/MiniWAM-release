"""
context_encoder.py

Encodes always-clean context signals (text embeddings + current observation VAE latent)
into context tokens for MoT cross-attention.
"""

from typing import Optional

import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    Produces context tokens from pre-computed text embeddings.
    These tokens are NEVER noised and NEVER predicted — they only condition the targets
    via cross-attention in the MoT.

    Text tokens:  [B, L_text, text_dim]  → projected to [B, L_text, hidden_dim]

    Note: Frame-0 observation is handled separately by WorldDiT.patchify_frame0()
    and placed in self-attention, not cross-attention.
    """

    def __init__(
        self,
        text_dim: int = 4096,
        hidden_dim: int = 1536,
        proprio_dim: Optional[int] = None,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.proprio_dim = None if proprio_dim in (None, 0) else int(proprio_dim)

        # Text projection (shared with expert text_embeddings for consistency)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proprio_proj = (
            nn.Linear(self.proprio_dim, hidden_dim) if self.proprio_dim is not None else None
        )

    def encode_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeds: [B, L, text_dim] pre-computed text embeddings
        Returns:
            [B, L, hidden_dim]
        """
        return self.text_proj(text_embeds)

    def encode_context(
        self,
        text_embeds: torch.Tensor,
        text_mask: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project text and append one current-proprio token when configured."""
        context = self.encode_text(text_embeds)
        mask = text_mask.to(device=context.device, dtype=torch.bool)
        if self.proprio_proj is None:
            if proprio is not None:
                raise ValueError("`proprio` was provided but proprio_dim is disabled.")
            return context, mask
        if proprio is None:
            raise ValueError("`proprio` is required when proprio_dim is enabled.")
        if proprio.ndim != 2 or proprio.shape != (context.shape[0], self.proprio_dim):
            raise ValueError(
                f"`proprio` must be [B,{self.proprio_dim}], got {tuple(proprio.shape)}."
            )
        token = self.proprio_proj(
            proprio.to(device=context.device, dtype=context.dtype)
        ).unsqueeze(1)
        token_mask = torch.ones((context.shape[0], 1), dtype=torch.bool, device=context.device)
        return torch.cat([context, token], dim=1), torch.cat([mask, token_mask], dim=1)
