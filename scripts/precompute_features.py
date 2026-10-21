"""The ResNet18 features of the CIFAR-10 train set that ``find_nearest_image`` compares against."""

import argparse
import logging
import os

import datasets
import torch

from ddpm.image.config import Config, load_config
from ddpm.image.data import load_data
from ddpm.image.eval import build_resnet_feature_extractor, extract_features
from ddpm.paths import Paths, add_root_argument


def extract_resnet_features(ds: datasets.DatasetDict, config: Config, paths: Paths) -> None:
    model = build_resnet_feature_extractor().to(config.device)
    model.eval()
    # Unshuffled: row i of the saved (50000, 512) tensor is train image i.
    dataloader_unshuffled = torch.utils.data.DataLoader(
        ds["train"], shuffle=False, batch_size=config.data.batch_size
    )

    preds = []
    with torch.no_grad():
        for batch in dataloader_unshuffled:
            preds.append(extract_features(model, batch["img"], config))
    preds_tensor = torch.cat(preds, dim=0)
    os.makedirs(os.path.dirname(paths.resnet_features_path), exist_ok=True)
    torch.save(preds_tensor, paths.resnet_features_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    add_root_argument(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = Paths.from_config(args.root, args.config, config.run_name)

    torch.manual_seed(0)
    logging.info(f"features with config {args.config}")
    logging.info(config)

    ds = load_data(paths.data_cache_dir)

    extract_resnet_features(ds, config, paths)
