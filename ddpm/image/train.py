import argparse
import logging
import os
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from ddpm.image.config import Config, UNetModelConfig, load_config
from ddpm.image.data import Batch, get_dataloaders, load_data, reverse_transform, transform
from ddpm.image.diffusion import create_sampler, create_schedule
from ddpm.image.diffusion.base import BaseSchedule
from ddpm.image.ema import EMA
from ddpm.image.models import create_model
from ddpm.image.models.base import BaseModel
from ddpm.image.visualise import plot_tensor_grid
from ddpm.paths import Paths, add_root_argument, find_latest_checkpoint

NUM_LOSS_BUCKETS = 10
VAL_SEED = 0


def apply_train_overrides(
    config: Config,
    *,
    lr: float | None = None,
    num_epochs: int | None = None,
    num_ch: int | None = None,
    omega: float | None = None,
    run_name: str | None = None,
) -> Config:
    """One-off training variants without a config file. They change what gets trained, so the
    caller must also pass a new ``run_name``: resume is silent, and would continue the base run.
    ``omega`` turns classifier-free guidance on; it is only the preview grids' weight."""
    if lr is not None:
        config.train.lr = lr
    if num_epochs is not None:
        config.train.num_epochs = num_epochs
    if num_ch is not None:
        if not isinstance(config.model, UNetModelConfig):
            raise ValueError("--num-ch only applies to a UNet model")
        config.model.num_ch = num_ch
    if omega is not None:
        config.guidance = "classifier-free"
        config.omega_classifier_free = omega
    if run_name is not None:
        config.run_name = run_name
    return config


def loss_bucket(ts: torch.Tensor, config: Config) -> torch.Tensor:
    """Timesteps (B, 1) -> the tenth of the time range each falls in: (B,) long in [0, 10)."""
    return ts.squeeze(1) * NUM_LOSS_BUCKETS // config.noise_schedule.T


def validation_loss(
    model: BaseModel,
    noise_schedule: BaseSchedule,
    dataloader: DataLoader[Batch],
    config: Config,
) -> tuple[float, list[float]]:
    """Held-out L_simple (DDPM Eq. 14) with the same (t, eps) per image on every call, given an
    unshuffled loader -> (mean over images, mean per tenth of the time range). True labels when
    guided, no flips. Leaves the global RNG as it was."""
    model.eval()
    loss_sum = torch.zeros(NUM_LOSS_BUCKETS, device=config.device)
    count = torch.zeros(NUM_LOSS_BUCKETS, device=config.device)
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(VAL_SEED)
        for batch in dataloader:
            x_0 = transform(batch["img"], config.device)
            lbl = batch["label"].to(config.device) if config.guidance is not None else None
            eps = torch.randn_like(x_0)
            ts = noise_schedule.sample_random_timestamps(x_0.shape[0], config.device)
            x_t = noise_schedule.forward(x_0, ts, eps)
            target = noise_schedule.training_target(x_0, ts, eps)
            per_sample = ((target - model(x_t, ts, lbl)) ** 2).mean(dim=(1, 2, 3))  # (B,)
            bucket = loss_bucket(ts, config)
            loss_sum.index_add_(0, bucket, per_sample)
            count.index_add_(0, bucket, torch.ones_like(per_sample))

    # Sums over images, so the short last batch counts once per image, not once per batch.
    by_bucket = loss_sum / count.clamp(min=1)
    return float(loss_sum.sum() / count.sum()), by_bucket.tolist()


class DiffusionTrainer:
    """DDPM training (Ho et al. 2020, Alg. 1), with CFG label dropout and EMA. Every
    ``eval_every_n_epochs``: a checkpoint, the validation loss (raw and EMA weights) and a sample
    grid. Resumes from the run's latest checkpoint."""

    def __init__(
        self,
        model: BaseModel,
        dataloader_train: DataLoader[Batch],
        dataloader_test: DataLoader[Batch],
        writer: SummaryWriter,
        config: Config,
        paths: Paths,
        resume: bool = True,
    ):
        self.model = model
        self.writer = writer
        self.paths = paths
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.loss_fn = torch.nn.MSELoss()

        self.noise_schedule = create_schedule(config.noise_schedule)
        self.sampler = create_sampler(config.sampler, self.noise_schedule)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr)
        self.config = config

        self.start_epoch = 0
        checkpoint: dict[str, Any] | None = None
        if resume:
            latest = find_latest_checkpoint(paths.weights_dir)
            if latest is not None:
                path, epoch = latest
                checkpoint = torch.load(path, weights_only=True, map_location=config.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.start_epoch = epoch + 1
                logging.info(f"resuming from {path}, starting at epoch {self.start_epoch}")

        self.ema = None
        if config.ema is not None:
            self.ema = EMA(self.model, config.ema.decay)
            if checkpoint is not None:
                # KeyError on purpose: EMA switched on mid-run is a different run
                self.ema.load_state_dict(checkpoint["ema_shadow_model"])

        # With guidance, preview with real labels so predict_noise's CFG blend runs.
        if config.guidance is not None:
            # One row per class: repeat_interleave groups by class, e.g. 128 // 10 -> 12 per row.
            per_class = max(1, config.data.batch_size // config.data.num_classes)
            self.sample_lbls: torch.Tensor | None = torch.arange(
                config.data.num_classes, device=config.device
            ).repeat_interleave(per_class)
            num_preview_samples = config.data.num_classes * per_class
            self.grid_nrow = per_class
        else:
            self.sample_lbls = None
            num_preview_samples = config.data.batch_size
            self.grid_nrow = 8
        self.sample_x0 = torch.randn(
            [
                num_preview_samples,
                *config.data.shape,
            ],
            dtype=torch.float,
            device=config.device,
        )
        self.global_step = self.start_epoch * len(dataloader_train)

        self.loss_sum_by_bucket = torch.zeros(NUM_LOSS_BUCKETS, device=config.device)
        self.loss_count_by_bucket = torch.zeros(NUM_LOSS_BUCKETS, device=config.device)

    @property
    def lr(self) -> float:
        """The optimizer's current learning rate, not ``config.train.lr``."""
        return float(self.optimizer.param_groups[0]["lr"])

    def train_batch(self, batch: dict[str, torch.Tensor], batch_idx: int) -> tuple[float, float]:
        """One step of Alg. 1 on a batch -> (loss, gradient norm before clipping)."""
        img_batch = transform(batch["img"], self.config.device, self.config.data.random_flip)
        lbl_batch = batch["label"].to(self.config.device)

        eps = torch.randn_like(img_batch).to(self.config.device)
        self.optimizer.zero_grad()
        ts = self.noise_schedule.sample_random_timestamps(img_batch.shape[0], self.config.device)
        x_t = self.noise_schedule.forward(img_batch, ts, eps)
        # eps for diffusion schedules
        target = self.noise_schedule.training_target(img_batch, ts, eps)

        if self.config.guidance is None:
            pred = self.model(x_t, ts, None)
        else:
            # CFG training (Ho & Salimans 2022, Alg. 1): p_uncond = 0.1, null class = num_classes
            drop_mask = torch.rand(lbl_batch.shape, device=self.config.device) < 0.1
            lbl_batch[drop_mask] = self.config.data.num_classes
            pred = self.model(x_t, ts, lbl_batch)
        loss: torch.Tensor = self.loss_fn(target, pred)  # L_simple, DDPM Eq. 14
        loss.backward()  # type: ignore[no-untyped-call]
        # Returns the norm before clipping; max_norm=inf measures it without clipping.
        max_norm = self.config.train.grad_clip
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm if max_norm is not None else torch.inf
        )
        self.optimizer.step()
        with torch.no_grad():
            # Loss per tenth of the time range, logged per epoch
            per_sample = ((target - pred) ** 2).mean(dim=(1, 2, 3))  # (B,)
            bucket = loss_bucket(ts, self.config)
            self.loss_sum_by_bucket.index_add_(0, bucket, per_sample)
            self.loss_count_by_bucket.index_add_(0, bucket, torch.ones_like(per_sample))

        if self.ema is not None and self.config.ema is not None:
            if batch_idx % self.config.ema.update_freq == 0:
                self.ema.update(self.model)

        return loss.item(), grad_norm.item()

    def train_epoch(self, epoch: int) -> tuple[list[float], list[float]]:
        losses = []
        grad_norms = []
        self.model.train()

        self.loss_sum_by_bucket = torch.zeros(NUM_LOSS_BUCKETS, device=self.config.device)
        self.loss_count_by_bucket = torch.zeros(NUM_LOSS_BUCKETS, device=self.config.device)

        for batch_idx, batch in enumerate(self.dataloader_train):
            loss, grad_norm = self.train_batch(batch, batch_idx)
            losses.append(loss)
            grad_norms.append(grad_norm)
            self.writer.add_scalar("train/mse_loss", loss, self.global_step)
            self.writer.add_scalar("train/grad_norm", grad_norm, self.global_step)
            self.writer.add_scalar("train/lr", self.lr, self.global_step)
            self.global_step += 1
        means = self.loss_sum_by_bucket / self.loss_count_by_bucket.clamp(min=1)
        for i, m in enumerate(means.tolist()):
            self.writer.add_scalar(f"train/mse_loss_bucket_{i}", m, epoch)
        return losses, grad_norms

    def log_validation_loss(self, tag: str, epoch: int) -> float:
        """Validation loss of the current weights, logged as ``tag`` and ``tag``_bucket_<i>."""
        loss, by_bucket = validation_loss(
            self.model, self.noise_schedule, self.dataloader_test, self.config
        )
        self.writer.add_scalar(tag, loss, epoch)
        for i, m in enumerate(by_bucket):
            self.writer.add_scalar(f"{tag}_bucket_{i}", m, epoch)
        return loss

    def save_model(self, epoch: int) -> None:
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.ema is not None:
            checkpoint["ema_shadow_model"] = self.ema.shadow_weights
        os.makedirs(self.paths.weights_dir, exist_ok=True)
        torch.save(checkpoint, f"{self.paths.weights_dir}/weights_{epoch}.pt")

    def train(self) -> None:
        os.makedirs(self.paths.weights_dir, exist_ok=True)
        os.makedirs(self.paths.visualise_dir, exist_ok=True)
        for epoch in range(self.start_epoch, self.config.train.num_epochs):
            t0 = time.time()
            losses, grad_norms = self.train_epoch(epoch)
            epoch_seconds = time.time() - t0

            avg_loss = float(np.array(losses).mean())
            avg_grad_norm = float(np.array(grad_norms).mean())
            lr = self.lr
            print(f"{epoch=} {avg_loss=:.4f} {avg_grad_norm=:.4f} {lr=:.2e} {epoch_seconds=:.1f}")
            self.writer.add_scalar("train/mse_loss_epoch", avg_loss, epoch)
            is_last_epoch = epoch == self.config.train.num_epochs - 1
            if epoch % self.config.train.eval_every_n_epochs == 0 or is_last_epoch:
                self.save_model(epoch)

                val_loss = self.log_validation_loss("val/mse_loss", epoch)
                print(f"{epoch=} {val_loss=:.4f}")
                if self.ema is not None:
                    self.ema.load_average_weights(self.model)
                    val_loss_ema = self.log_validation_loss("val/mse_loss_ema", epoch)
                    print(f"{epoch=} {val_loss_ema=:.4f}")

                self.model.eval()
                with torch.no_grad():
                    samples = self.sampler.sample(
                        self.model,
                        self.sample_x0,
                        self.sample_lbls,
                        self.config,
                        num_steps=1000,
                    )
                    samples = reverse_transform(samples)
                plot_tensor_grid(
                    samples,
                    f"{self.paths.visualise_dir}/samples_epoch_{epoch}.png",
                    nrow=self.grid_nrow,
                )

                self.model.train()
                if self.ema is not None:
                    self.ema.restore_current_weights(self.model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    add_root_argument(parser)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--num-ch", type=int, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    overridden = (
        args.lr is not None
        or args.num_epochs is not None
        or args.num_ch is not None
        or args.omega is not None
    )
    if overridden and args.run_name is None:
        parser.error(
            "--run-name is required alongside --lr/--num-epochs/--num-ch/--omega, so this run "
            "doesn't silently resume into (and contaminate) the base config's checkpoint "
            "directory"
        )
    config = apply_train_overrides(
        config,
        lr=args.lr,
        num_epochs=args.num_epochs,
        num_ch=args.num_ch,
        omega=args.omega,
        run_name=args.run_name,
    )
    paths = Paths.from_config(args.root, args.config, config.run_name)
    os.makedirs(paths.visualise_dir, exist_ok=True)
    torch.manual_seed(0)
    logging.info(f"train with config {args.config}")
    logging.info(config)

    ds = load_data(paths.data_cache_dir)

    dataloader_train, dataloader_test = get_dataloaders(ds, config.data.batch_size)
    model = create_model(config).to(config.device)

    writer = SummaryWriter(log_dir=paths.log_dir)

    trainer = DiffusionTrainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        writer=writer,
        config=config,
        paths=paths,
        resume=not args.fresh,
    )

    trainer.train()
