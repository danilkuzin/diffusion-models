"""FID's floor and ceiling on this pipeline: train vs train, train vs test, train vs noise."""

import argparse
import json
import logging
import os
from collections.abc import Iterable, Iterator

import datasets
import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from ddpm.image.config import Config, load_config
from ddpm.image.data import load_data, reverse_transform
from ddpm.paths import Paths, add_root_argument

# compute_fid_kid's sample batch size: the protocol's fake side is --num-fid-batches of these.
SAMPLE_BATCH_SIZE = 128


def dataset_images(split: datasets.Dataset) -> Iterator[torch.Tensor]:
    """uint8 (128, C, H, W) batches of ``split``, in order."""
    for batch in torch.utils.data.DataLoader(split, shuffle=False, batch_size=SAMPLE_BATCH_SIZE):
        yield batch["img"]


def noise_images(config: Config, num_batches: int) -> Iterator[torch.Tensor]:
    """``num_batches`` batches of N(0, I) noise, mapped to uint8 the way samples are."""
    for _ in range(num_batches):
        yield reverse_transform(torch.randn((SAMPLE_BATCH_SIZE, *config.data.shape)))


def fid_between(
    real: Iterable[torch.Tensor], fake: Iterable[torch.Tensor], device: str
) -> dict[str, float]:
    """FID between two streams of uint8 (B, C, H, W) batches -> {"fid", "num_real", "num_fake"}."""
    fid = FrechetInceptionDistance().to(device)
    num_real = num_fake = 0
    for batch in real:
        fid.update(batch.to(device), real=True)
        num_real += batch.shape[0]
    for batch in fake:
        fid.update(batch.to(device), real=False)
        num_fake += batch.shape[0]

    return {"fid": float(fid.compute()), "num_real": num_real, "num_fake": num_fake}


def calibrate(
    ds: datasets.DatasetDict, config: Config, num_fid_batches: int
) -> dict[str, dict[str, float]]:
    """-> {comparison name: ``fid_between`` result} for the three comparisons."""
    train = ds["train"]
    num_fake = num_fid_batches * SAMPLE_BATCH_SIZE
    if num_fake >= len(train):
        raise ValueError(f"{num_fake} fake images leave no disjoint train images to compare to")

    return {
        "train_vs_train": fid_between(
            dataset_images(train.select(range(num_fake, len(train)))),
            dataset_images(train.select(range(num_fake))),
            config.device,
        ),
        "train_vs_test": fid_between(
            dataset_images(train), dataset_images(ds["test"]), config.device
        ),
        "train_vs_noise": fid_between(
            dataset_images(train), noise_images(config, num_fid_batches), config.device
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    add_root_argument(parser)
    parser.add_argument("--num-fid-batches", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = Paths.from_config(args.root, args.config, config.run_name)

    torch.manual_seed(0)
    ds = load_data(paths.data_cache_dir)
    comparisons = calibrate(ds, config, args.num_fid_batches)

    print(f"{'comparison':16} {'FID':>8} {'real':>7} {'fake':>7}")
    for name, r in comparisons.items():
        print(f"{name:16} {r['fid']:>8.2f} {r['num_real']:>7.0f} {r['num_fake']:>7.0f}")

    out_path = paths.under_root(f"fid_calibration_{args.num_fid_batches}batches.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "num_fid_batches": args.num_fid_batches,
                "sample_batch_size": SAMPLE_BATCH_SIZE,
                "seed": 0,
                "comparisons": comparisons,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")
