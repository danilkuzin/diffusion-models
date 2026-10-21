"""Where a run's artifacts live, under the ``--root`` every entry point takes."""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = "/data"


def find_latest_checkpoint(weights_dir: str) -> tuple[str, int] | None:
    """(path, epoch) of the highest-epoch ``weights_<epoch>.pt`` in ``weights_dir``, or None."""
    if not os.path.isdir(weights_dir):
        return None
    epochs = [
        int(m.group(1))
        for name in os.listdir(weights_dir)
        if (m := re.fullmatch(r"weights_(\d+)\.pt", name)) is not None
    ]
    if not epochs:
        return None
    latest = max(epochs)
    return os.path.join(weights_dir, f"weights_{latest}.pt"), latest


def resolve_epoch(weights_dir: str, epoch: int | None) -> int:
    """``epoch`` if given (not checked: ``torch.load`` raises if it's missing), else the latest."""
    if epoch is not None:
        return epoch
    latest = find_latest_checkpoint(weights_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint found in {weights_dir}")
    return latest[1]


@dataclass(frozen=True)
class Paths:
    root: str
    run_name: str
    # The config's filename stem; only `eval_results_path` uses it.
    config_name: str | None = None
    # Keeps an eval-only override's results file apart from the base config's.
    tag: str | None = None

    @classmethod
    def from_config(
        cls, root: str, config_path: str, run_name: str, tag: str | None = None
    ) -> "Paths":
        return cls(root=root, run_name=run_name, config_name=Path(config_path).stem, tag=tag)

    @property
    def weights_dir(self) -> str:
        return f"{self.root}/weights/{self.run_name}"

    @property
    def log_dir(self) -> str:
        return f"{self.root}/logs/{self.run_name}"

    @property
    def visualise_dir(self) -> str:
        return f"{self.root}/plots/{self.run_name}"

    @property
    def data_cache_dir(self) -> str:
        return f"{self.root}/huggingface/datasets"

    @property
    def text_data_dir(self) -> str:
        return f"{self.root}/text_data"

    @property
    def resnet_features_path(self) -> str:
        return f"{self.root}/weights/cifar10_resnet_features.pt"

    @property
    def eval_results_path(self) -> str:
        name = self.config_name or self.run_name
        if self.tag:
            name = f"{name}_{self.tag}"
        return f"{self.root}/eval_results_{name}.json"

    def under_root(self, path: str) -> str:
        """An ``--out`` value: absolute paths as given, bare filenames under the root."""
        return path if Path(path).is_absolute() else f"{self.root}/{path}"


def add_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-r", "--root", default=DEFAULT_ROOT)
