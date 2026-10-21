from pathlib import Path

import torch

from ddpm.image.data import transform
from ddpm.image.models.dit import (
    PatchEmbed,
    get_2d_sincos_pos_embed,
    unpatchify,
)

CIFAR10_SAMPLE = Path(__file__).parents[1] / "assets" / "cifar10_sample.pt"


def identity_patch_embed(img_size: int, patch_size: int, channels: int) -> PatchEmbed:
    """A patchify conv whose output channel idx copies the idx-th (row, col, channel) pixel of
    its patch, so a token is its patch's pixels."""
    hidden_dim = patch_size * patch_size * channels
    patch_embed = PatchEmbed(img_size, patch_size, channels, hidden_dim)
    identity = torch.eye(hidden_dim).reshape(hidden_dim, patch_size, patch_size, channels)
    patch_embed.proj.weight.data = identity.permute(0, 3, 1, 2).contiguous()
    assert patch_embed.proj.bias is not None
    patch_embed.proj.bias.data.zero_()
    return patch_embed


def test_patchify_unpatchify_round_trip_with_identity_weights(viz: Path | None) -> None:
    """With `identity_patch_embed`, unpatchify must give back the input exactly: the conv's
    and unpatchify's (row, col, channel) layouts agree."""
    img_size, patch_size, channels = 32, 2, 3  # DiT on CIFAR-10
    patch_embed = identity_patch_embed(img_size, patch_size, channels)
    x = torch.randn(2, channels, img_size, img_size)

    with torch.no_grad():
        tokens = patch_embed(x)
        reconstructed = unpatchify(tokens, patch_size, channels, patch_embed.grid_size)

    assert torch.allclose(reconstructed, x, atol=1e-5)

    if viz:
        from visualisation.image.dit import plot_patch_tokens

        image = torch.load(CIFAR10_SAMPLE)[0]  # (3, 32, 32) uint8
        with torch.no_grad():
            image_tokens = patch_embed(transform(image[None], "cpu"))[0]
        plot_patch_tokens(viz, image, image_tokens, patch_size)


def test_2d_sincos_pos_embed_similarity_depends_only_on_the_offset(viz: Path | None) -> None:
    """Each [sin | cos] half gives <e(p), e(q)> = sum_k cos(w_k (p - q)), so two patches'
    similarity is a function of their (row, col) offset alone, and peaks at D / 2 (one
    sin^2 + cos^2 = 1 per frequency) only for the patch itself."""
    D, G = 64, 16  # DiT's embd_dim and its patch grid on CIFAR-10
    pos_embed = get_2d_sincos_pos_embed(D, G)  # (G * G, D), row-major
    gram = pos_embed @ pos_embed.T

    torch.testing.assert_close(gram.diagonal(), torch.full((G * G,), D / 2))
    assert torch.all(gram[~torch.eye(G * G, dtype=torch.bool)] < D / 2 - 0.1)

    grid = pos_embed.reshape(G, G, D)
    for dy, dx in [(0, 1), (1, 0), (2, 3)]:
        sim = (grid[: G - dy, : G - dx] * grid[dy:, dx:]).sum(-1)  # every pair at this offset
        torch.testing.assert_close(sim, sim[0, 0].expand_as(sim))

    if viz:
        from visualisation.image.dit import plot_pos_embed_similarity

        plot_pos_embed_similarity(viz, pos_embed, [(0, 0), (8, 8), (4, 13)])
