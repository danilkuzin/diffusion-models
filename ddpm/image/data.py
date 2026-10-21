import datasets
import torch

Batch = dict[str, torch.Tensor]


def load_data(cache_dir: str) -> datasets.DatasetDict:
    """CIFAR-10: 'train' (50,000) and 'test' (10,000); ``img`` (3, 32, 32) uint8, ``label`` long."""
    ds = datasets.load_dataset("uoft-cs/cifar10", cache_dir=cache_dir)
    ds.set_format(type="torch")

    return ds


def get_dataloaders(
    ds: datasets.DatasetDict, batch_size: int
) -> tuple[torch.utils.data.DataLoader[Batch], torch.utils.data.DataLoader[Batch]]:
    """(train, test) loaders: train shuffled, test in order (``validation_loss`` needs that)."""
    return torch.utils.data.DataLoader(
        ds["train"], shuffle=True, batch_size=batch_size
    ), torch.utils.data.DataLoader(ds["test"], shuffle=False, batch_size=batch_size)


def transform(img_batch: torch.Tensor, device: str, random_flip: bool = False) -> torch.Tensor:
    """uint8 (B, C, H, W) in [0, 255] -> float32 in [-1, 1] (DDPM §3.3) on ``device``, optionally
    flipped (DDPM App. B: random horizontal flips for CIFAR-10)."""
    x = img_batch.to(torch.float).to(device) / 127.5 - 1
    if random_flip:
        # One draw per image: torchvision's RandomHorizontalFlip on a batch flips all or none.
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    return x


def reverse_transform(img_batch: torch.Tensor) -> torch.Tensor:
    """float in [-1, 1] (clamped first) -> uint8 in [0, 255] on the CPU."""
    return ((img_batch.clamp(-1, 1) + 1) * 127.5).to("cpu").to(torch.uint8)
