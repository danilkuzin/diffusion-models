"""The first ``--num-images`` CIFAR-10 test images, (N, 3, 32, 32) uint8, for the tests' figures."""

import argparse
from pathlib import Path

import torch

from ddpm.image.data import load_data
from ddpm.paths import Paths, add_root_argument

OUT_PATH = Path(__file__).resolve().parents[1] / "tests" / "assets" / "cifar10_sample.pt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_root_argument(parser)
    parser.add_argument("-n", "--num-images", type=int, default=4)
    args = parser.parse_args()

    ds = load_data(Paths(root=args.root, run_name="").data_cache_dir)  # no run: only the cache
    images = torch.stack([ds["test"][i]["img"] for i in range(args.num_images)])

    OUT_PATH.parent.mkdir(exist_ok=True)
    torch.save(images, OUT_PATH)
    print(f"Saved {tuple(images.shape)} {images.dtype} to {OUT_PATH}")
