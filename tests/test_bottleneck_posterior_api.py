import torch
from einops import rearrange

from miniwam.models.latent_encoder import (
    DINOv3LatentEncoder3D,
    LatentEncoder3D,
)

torch.set_num_threads(1)


def _reference_forward(encoder, z_in, *, sample, kl_reduction):
    prepared = encoder._prepare_input(z_in)
    x = encoder.bottleneck(encoder.backbone(prepared))
    mu = encoder.mu_head(x)
    if encoder.use_kl and encoder.logvar_head is not None:
        logvar = encoder.logvar_head(x)
        kl_per_sample = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=(1, 2, 3, 4))
        kl = kl_per_sample if kl_reduction == "none" else kl_per_sample.mean()
        if sample:
            std = (0.5 * logvar).exp()
            grid = mu + std * torch.randn_like(std)
        else:
            grid = mu
    else:
        kl = (
            torch.zeros((z_in.shape[0],), device=z_in.device, dtype=z_in.dtype)
            if kl_reduction == "none"
            else torch.tensor(0.0, device=z_in.device, dtype=z_in.dtype)
        )
        grid = mu
    return rearrange(grid, "b d t h w -> b (t h w) d"), grid, tuple(grid.shape[2:]), kl


def _assert_exact_tuple(actual, expected):
    assert actual[2] == expected[2]
    for actual_tensor, expected_tensor in zip(
        (actual[0], actual[1], actual[3]),
        (expected[0], expected[1], expected[3]),
        strict=True,
    ):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_posterior_api_matches_reference_forward_and_rng():
    torch.manual_seed(3)
    encoder = LatentEncoder3D(
        in_channels=4,
        latent_dim=6,
        compression_stages=[(8, (3, 3, 3), (1, 2, 2), (1, 1, 1))],
        use_kl=True,
    ).train()
    z_in = torch.randn(2, 4, 3, 6, 6)

    state_before = torch.random.get_rng_state()
    posterior = encoder.posterior_parameters(z_in, return_intermediates=True)
    assert torch.equal(torch.random.get_rng_state(), state_before)
    assert posterior.mu_grid.shape == (2, 6, 3, 3, 3)
    assert posterior.mu_tokens.shape == (2, 27, 6)
    assert posterior.logvar_grid is not None
    assert posterior.kl_elementwise is not None
    assert posterior.prepared_input is z_in
    assert posterior.pre_mu_head_grid is not None

    torch.manual_seed(19)
    actual = encoder(z_in, sample=True, kl_reduction="none")
    actual_rng = torch.random.get_rng_state()
    torch.manual_seed(19)
    expected = _reference_forward(encoder, z_in, sample=True, kl_reduction="none")
    expected_rng = torch.random.get_rng_state()
    _assert_exact_tuple(actual, expected)
    assert torch.equal(actual_rng, expected_rng)

    encoder.eval()
    eval_params = encoder.posterior_parameters(z_in)
    encoder.train()
    train_params = encoder.posterior_parameters(z_in)
    torch.testing.assert_close(eval_params.mu_grid, train_params.mu_grid, rtol=0, atol=0)
    torch.testing.assert_close(eval_params.logvar_grid, train_params.logvar_grid, rtol=0, atol=0)


def test_no_kl_posterior_is_explicitly_unavailable():
    encoder = LatentEncoder3D(
        in_channels=4,
        latent_dim=5,
        compression_stages=[(8, (1, 3, 3), (1, 1, 1), (0, 1, 1))],
        use_kl=False,
    )
    z_in = torch.randn(2, 4, 2, 4, 4)
    posterior = encoder.posterior_parameters(z_in)
    assert posterior.logvar_grid is None
    assert posterior.kl_elementwise is None
    actual = encoder(z_in, sample=False, kl_reduction="none")
    expected = _reference_forward(encoder, z_in, sample=False, kl_reduction="none")
    _assert_exact_tuple(actual, expected)


def test_dino_preparation_is_shared_by_posterior_and_forward():
    torch.manual_seed(5)
    encoder = DINOv3LatentEncoder3D(
        input_dim=8,
        adapter_dim=4,
        temporal_downsample_factor=2,
        patch_grid=(2, 2),
        latent_dim=5,
        compression_stages=[(8, (3, 3, 3), (1, 1, 1), (1, 1, 1))],
        use_kl=True,
    )
    features = torch.randn(2, 5, 4, 8)
    posterior = encoder.posterior_parameters(features, return_intermediates=True)

    grid = encoder._as_grid(features)
    current = encoder.channel_adapter(grid[:, :, :1])
    future = encoder.temporal_compressor(encoder.channel_adapter(grid[:, :, 1:]))
    prepared = torch.cat([current, future], dim=2)
    hidden = encoder.bottleneck(encoder.backbone(prepared))
    expected_mu = encoder.mu_head(hidden)
    torch.testing.assert_close(posterior.prepared_input, prepared, rtol=0, atol=0)
    torch.testing.assert_close(posterior.mu_grid, expected_mu, rtol=0, atol=0)

    tokens, grid_out, shape, _ = encoder(features, sample=False)
    torch.testing.assert_close(grid_out, posterior.mu_grid, rtol=0, atol=0)
    torch.testing.assert_close(tokens, posterior.mu_tokens, rtol=0, atol=0)
    assert shape == posterior.grid_shape


