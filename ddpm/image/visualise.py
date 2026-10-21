import torch
import torchvision
from PIL import Image


def plot_tensor_grid(
    samples: torch.Tensor | list[torch.Tensor], out_path: str, nrow: int = 8
) -> None:
    """uint8 samples, (B, C, H, W) or a list of (C, H, W) -> a PNG grid, ``nrow`` per row."""
    # PIL saves at native resolution; a matplotlib figure would resample the 32x32 tiles.
    grid = torchvision.utils.make_grid(samples, nrow=nrow).permute((1, 2, 0))
    Image.fromarray(grid.cpu().detach().numpy()).save(out_path)
