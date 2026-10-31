"""
latent_encoder.py

3D Patch-Merging Autoencoder (E_φ).
Compresses concatenated VAE latents (z_t, z_fwd) into a compact 3D latent grid.
Supervised by IDM loss through MoT attention (Stage 1), then frozen (Stage 2).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange


@dataclass(frozen=True)
class PosteriorParameters:
    """Deterministic parameters of the transition bottleneck posterior.

    Tensor grids use the model's native ``[B,D,T,H,W]`` layout. Optional
    intermediates are returned only when explicitly requested so ordinary
    training does not retain additional activations.
    """

    mu_grid: torch.Tensor
    logvar_grid: Optional[torch.Tensor]
    kl_elementwise: Optional[torch.Tensor]
    grid_shape: Tuple[int, int, int]
    mu_tokens: torch.Tensor
    prepared_input: Optional[torch.Tensor] = None
    pre_mu_head_grid: Optional[torch.Tensor] = None


class LatentEncoder3D(nn.Module):
    """
    3D Patch-Merging encoder with optional KL regularization.

    Input:  [B, C_vae, T_lat, H_lat, W_lat]  — concatenated current + future VAE latents
    Output: L_clean [B, num_tokens, latent_dim]
            L_grid  [B, latent_dim, t, h, w]   — 3D grid (for 3D RoPE)
            grid_shape (t, h, w)                — shape metadata
            kl_loss scalar                      — KL divergence (0 if use_kl=False)
    """

    def __init__(
        self,
        in_channels: int = 16,
        latent_dim: int = 64,
        compression_stages: Optional[List] = None,
        use_kl: bool = True,
        bottleneck_type: str = "none",
        target_grid: Optional[Sequence[Optional[int]]] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_kl = use_kl
        self.bottleneck_type = str(bottleneck_type).lower()
        if self.bottleneck_type not in ("none", "adaptive_avg_pool3d"):
            raise ValueError(
                "bottleneck_type must be 'none' or 'adaptive_avg_pool3d', "
                f"got {bottleneck_type!r}."
            )
        if target_grid is not None:
            if len(target_grid) != 3:
                raise ValueError(f"target_grid must be [T,H,W], got {target_grid!r}.")
            self.target_grid = tuple(None if x is None else int(x) for x in target_grid)
            if any(x is not None and x <= 0 for x in self.target_grid):
                raise ValueError(f"target_grid entries must be positive or null, got {target_grid!r}.")
        else:
            self.target_grid = None
        if self.bottleneck_type == "adaptive_avg_pool3d" and self.target_grid is None:
            raise ValueError("adaptive_avg_pool3d requires an explicit target_grid.")

        if compression_stages is None:
            # Default: spatial 8× (32→4), temporal 2× (2→1)
            compression_stages = [
                # (out_ch, kernel, stride, padding)
                (32,  (1, 3, 3), (1, 2, 2), (0, 1, 1)),   # spatial 2×
                (64,  (1, 3, 3), (1, 2, 2), (0, 1, 1)),   # spatial 2×
                (128, (3, 3, 3), (2, 2, 2), (1, 1, 1)),   # temporal + spatial 2×
            ]

        layers = []
        ch = in_channels
        for out_ch, kernel, stride, padding in compression_stages:
            layers.extend([
                nn.Conv3d(ch, out_ch, kernel_size=kernel, stride=stride, padding=padding),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ])
            ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.compression_stages = tuple(compression_stages)
        self.bottleneck = (
            nn.AdaptiveAvgPool3d(self.target_grid)
            if self.bottleneck_type == "adaptive_avg_pool3d"
            else nn.Identity()
        )

        # KL heads (output mu and logvar for reparameterization)
        self.mu_head = nn.Conv3d(ch, latent_dim, kernel_size=1)
        if use_kl:
            self.logvar_head = nn.Conv3d(ch, latent_dim, kernel_size=1)
        else:
            self.logvar_head = None

    def _prepare_input(self, z_in: torch.Tensor) -> torch.Tensor:
        """Convert native cached features to the common 3D encoder input."""
        return z_in

    def posterior_parameters(
        self,
        z_in: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> PosteriorParameters:
        """Return deterministic posterior parameters without sampling.

        This method is independent of ``self.training`` and does not consume
        RNG state. It is the canonical analysis entry point for both Wan-VAE
        and DINOv3 encoders.
        """
        prepared = self._prepare_input(z_in)
        x = self.bottleneck(self.backbone(prepared))
        mu = self.mu_head(x)
        if self.use_kl and self.logvar_head is not None:
            logvar = self.logvar_head(x)
            # Elementwise KL divergence from the standard normal prior.
            kl_elementwise = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        else:
            logvar = None
            kl_elementwise = None
        grid_shape = tuple(mu.shape[2:])
        return PosteriorParameters(
            mu_grid=mu,
            logvar_grid=logvar,
            kl_elementwise=kl_elementwise,
            grid_shape=grid_shape,
            mu_tokens=rearrange(mu, "b d t h w -> b (t h w) d"),
            prepared_input=prepared if return_intermediates else None,
            pre_mu_head_grid=x if return_intermediates else None,
        )

    def forward(
        self,
        z_in: torch.Tensor,
        sample: Optional[bool] = None,
        kl_reduction: str = "mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], torch.Tensor]:
        """
        Args:
            z_in: VAE latents [B, C, T, H, W]
            sample: whether to sample from q(z|x). If None, follows self.training.
            kl_reduction: "mean" returns a scalar KL; "none" returns [B].

        Returns:
            L_clean:    [B, num_tokens, latent_dim]  — flattened world tokens
            L_grid:     [B, latent_dim, t, h, w]     — 3D grid form
            grid_shape: (t, h, w)                    — for 3D RoPE construction
            kl_loss:    scalar or [B]                — KL divergence term
        """
        if kl_reduction not in ("mean", "none"):
            raise ValueError(f"`kl_reduction` must be 'mean' or 'none', got {kl_reduction!r}")

        posterior = self.posterior_parameters(z_in)
        mu = posterior.mu_grid

        if sample is None:
            sample = self.training

        if posterior.logvar_grid is not None and posterior.kl_elementwise is not None:
            logvar = posterior.logvar_grid
            # KL divergence: KL(q(z|x) || N(0,I))
            kl_per_sample = posterior.kl_elementwise.sum(dim=(1, 2, 3, 4))
            kl_loss = kl_per_sample if kl_reduction == "none" else kl_per_sample.mean()

            if sample:
                # Reparameterization trick
                std = (0.5 * logvar).exp()
                L_grid = mu + std * torch.randn_like(std)
            else:
                L_grid = mu
        else:
            if kl_reduction == "none":
                kl_loss = torch.zeros((z_in.shape[0],), device=z_in.device, dtype=z_in.dtype)
            else:
                kl_loss = torch.tensor(0.0, device=z_in.device, dtype=z_in.dtype)
            L_grid = mu

        grid_shape = tuple(L_grid.shape[2:])  # (t, h, w)

        # Tokenize: flatten spatiotemporal dims into a sequence
        L_clean = rearrange(L_grid, "b d t h w -> b (t h w) d")

        return L_clean, L_grid, grid_shape, kl_loss

    def output_grid_shape(self, input_grid: Sequence[int]) -> Tuple[int, int, int]:
        """Return the exact output grid for an input ``(T,H,W)`` without a forward."""
        shape = [int(x) for x in input_grid]
        for _, kernel, stride, padding in self.compression_stages:
            for axis in range(3):
                shape[axis] = (
                    shape[axis] + 2 * int(padding[axis]) - int(kernel[axis])
                ) // int(stride[axis]) + 1
        if self.bottleneck_type == "adaptive_avg_pool3d":
            shape = [src if dst is None else dst for src, dst in zip(shape, self.target_grid)]
        return tuple(shape)


class DINOv3LatentEncoder3D(LatentEncoder3D):
    """Task encoder for framewise DINO patch maps.

    The current map is adapted once. Every future map is adapted and then
    compressed temporally by a learned, factor-based Conv3D stack before the
    ordinary spatial/temporal E_phi backbone. Thus no future RGB frame is
    discarded by uniform subsampling.
    """

    def __init__(
        self,
        input_dim: int = 768,
        adapter_dim: int = 64,
        temporal_downsample_factor: int = 4,
        patch_grid: Sequence[int] = (14, 14),
        **kwargs,
    ):
        factor = int(temporal_downsample_factor)
        if factor < 1 or factor & (factor - 1):
            raise ValueError("temporal_downsample_factor must be a positive power of two.")
        self.input_dim = int(input_dim)
        self.adapter_dim = int(adapter_dim)
        self.temporal_downsample_factor = factor
        self.patch_grid = tuple(int(x) for x in patch_grid)
        if len(self.patch_grid) != 2 or self.patch_grid[0] * self.patch_grid[1] <= 0:
            raise ValueError(f"patch_grid must be [H,W], got {patch_grid!r}.")
        super().__init__(in_channels=self.adapter_dim, **kwargs)
        self.channel_adapter = nn.Sequential(
            nn.Conv3d(self.input_dim, self.adapter_dim, kernel_size=1),
            nn.GroupNorm(min(32, self.adapter_dim), self.adapter_dim),
            nn.SiLU(),
        )
        temporal_layers = []
        remaining = factor
        while remaining > 1:
            temporal_layers.extend([
                nn.Conv3d(
                    self.adapter_dim, self.adapter_dim,
                    kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0),
                ),
                nn.GroupNorm(min(32, self.adapter_dim), self.adapter_dim),
                nn.SiLU(),
            ])
            remaining //= 2
        self.temporal_compressor = nn.Sequential(*temporal_layers)

    def _as_grid(self, features: torch.Tensor) -> torch.Tensor:
        # [B,T,N,D] or [B,D,T,H,W] -> [B,D,T,H,W]
        if features.ndim == 5:
            if features.shape[1] != self.input_dim:
                raise ValueError(
                    f"DINO grid channels must be {self.input_dim}, got {tuple(features.shape)}."
                )
            return features
        if features.ndim != 4 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"DINO features must be [B,T,N,{self.input_dim}] or [B,{self.input_dim},T,H,W], "
                f"got {tuple(features.shape)}."
            )
        h, w = self.patch_grid
        if features.shape[2] != h * w:
            raise ValueError(
                f"DINO patch count must equal patch_grid product {h*w}, got {features.shape[2]}."
            )
        return rearrange(features, "b t (h w) d -> b d t h w", h=h, w=w)

    def _prepare_input(self, z_in: torch.Tensor) -> torch.Tensor:
        grid = self._as_grid(z_in)
        if grid.shape[2] < 2:
            raise ValueError("DINO E_phi input must contain one current and at least one future frame.")
        current = self.channel_adapter(grid[:, :, :1])
        future = self.temporal_compressor(self.channel_adapter(grid[:, :, 1:]))
        return torch.cat([current, future], dim=2)

    def output_grid_shape(self, input_grid: Sequence[int]) -> Tuple[int, int, int]:
        raw_t, h, w = (int(x) for x in input_grid)
        if raw_t < 2:
            raise ValueError("DINO input grid includes current + future and must have T >= 2.")
        future_t = raw_t - 1
        for _ in range(self.temporal_downsample_factor.bit_length() - 1):
            future_t = (future_t + 1) // 2
        return super().output_grid_shape((1 + future_t, h, w))


class FutureFeatureReconstructionDecoder(nn.Module):
    """Lightweight decoder from the compact grid to native future features."""

    def __init__(self, latent_dim: int, hidden_dim: int = 64, out_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, latent_grid: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        target_shape = tuple(int(x) for x in target_shape)
        x = torch.nn.functional.interpolate(
            latent_grid, size=target_shape, mode="trilinear", align_corners=False,
        )
        return self.net(x)
