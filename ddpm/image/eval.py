import argparse
import json
import logging
import os
import time
from dataclasses import replace
from typing import get_args

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Normalize, Resize

from ddpm.image.config import (
    AnySamplerConfig,
    Config,
    DDIMSamplerConfig,
    DDPMSamplerConfig,
    UNetModelConfig,
    load_config,
)
from ddpm.image.data import Batch, get_dataloaders, load_data, reverse_transform
from ddpm.image.diffusion import create_sampler, create_schedule
from ddpm.image.models import create_model
from ddpm.image.models.base import BaseModel
from ddpm.image.train import validation_loss
from ddpm.paths import Paths, add_root_argument, resolve_epoch

logger = logging.getLogger(__name__)

# Sampler config class by its `type`, read off the union. All fields have defaults, so the name
# alone builds one.
SAMPLER_CONFIGS: dict[str, type[AnySamplerConfig]] = {
    cls.model_fields["type"].default: cls for cls in get_args(AnySamplerConfig)
}


def apply_eval_overrides(
    config: Config,
    *,
    sampler: str | None = None,
    eta: float | None = None,
    omega: float | None = None,
    run_name: str | None = None,
    num_ch: int | None = None,
) -> Config:
    """Eval-only overrides: they change how a checkpoint is sampled, never what was trained.
    ``omega`` turns guidance on, so a checkpoint trained without it fails to load."""
    if eta is not None and sampler != "ddim":
        raise ValueError("--eta only applies to --sampler ddim")
    if sampler is not None:
        if sampler not in SAMPLER_CONFIGS:
            raise ValueError(
                f"unknown sampler {sampler!r}, expected one of {list(SAMPLER_CONFIGS)}"
            )
        config.sampler = (
            DDIMSamplerConfig(eta=eta) if eta is not None else SAMPLER_CONFIGS[sampler]()
        )
    if omega is not None:
        config.guidance = "classifier-free"
        config.omega_classifier_free = omega
    if run_name is not None:
        config.run_name = run_name
    if num_ch is not None:
        if not isinstance(config.model, UNetModelConfig):
            raise ValueError("--num-ch only applies to a UNet model")
        config.model.num_ch = num_ch
    return config


def load_model(config: Config, paths: Paths, epoch: int, *, force_raw: bool = False) -> BaseModel:
    """``weights_<epoch>.pt`` -> model, with the EMA shadow over the raw weights when
    ``config.ema`` is set and not ``force_raw``."""
    checkpoint = torch.load(
        f"{paths.weights_dir}/weights_{epoch}.pt", weights_only=True, map_location=config.device
    )
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    if config.ema is not None and not force_raw:
        incompatible = model.load_state_dict(checkpoint["ema_shadow_model"], strict=False)
        # Only frozen params may be absent; a missing trainable one would mix raw and EMA weights.
        frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
        missing_trainable = sorted(set(incompatible.missing_keys) - frozen)
        if missing_trainable or incompatible.unexpected_keys:
            raise ValueError(
                f"EMA shadow doesn't match the model: missing trainable {missing_trainable}, "
                f"unexpected {incompatible.unexpected_keys}"
            )

    return model


def compute_fid_kid(
    config: Config,
    dataloader_train: torch.utils.data.DataLoader[Batch],
    model: BaseModel,
    sample_batch_size: int = 128,
    num_sample_batches: int = 100,
    num_sampler_steps: int = 1000,
    max_real_batches: int | None = None,
) -> dict[str, float]:
    """FID (Heusel et al. 2017) and KID (Binkowski et al. 2018) of num_sample_batches x
    sample_batch_size samples against max_real_batches (or all) training batches -> {"fid",
    "kid_mean", "kid_std", "generation_seconds"}. Relative at small budgets. With guidance, the
    samples are conditioned on random class labels."""
    noise_schedule = create_schedule(config.noise_schedule)
    sampler = create_sampler(config.sampler, noise_schedule)
    logger.info(
        "FID/KID: %d sample batches x %d imgs at %d sampler steps/batch (%d model forwards); "
        "real batches capped at %s; Inception on %s",
        num_sample_batches,
        sample_batch_size,
        num_sampler_steps,
        num_sample_batches * num_sampler_steps,
        max_real_batches if max_real_batches is not None else "all",
        config.device,
    )

    # Inception on the model's device
    fid = FrechetInceptionDistance().to(config.device)
    kid = KernelInceptionDistance().to(config.device)

    t_real = time.time()
    num_real = 0
    for real_idx, batch in enumerate(dataloader_train):
        if max_real_batches is not None and real_idx >= max_real_batches:
            break
        imgs = batch["img"].to(config.device)
        fid.update(imgs, real=True)
        kid.update(imgs, real=True)
        num_real += 1
    logger.info("FID/KID: ingested %d real batches in %.1fs", num_real, time.time() - t_real)

    model.eval()
    t_gen = time.time()
    with torch.no_grad():
        for batch_idx in range(num_sample_batches):
            # Fresh noise per batch: with a fixed one, DDIM at eta=0 repeats the same images.
            sample_x0 = torch.randn(
                [
                    sample_batch_size,
                    *config.data.shape,
                ],
                dtype=torch.float,
                device=config.device,
            )
            sample_lbls = (
                torch.randint(
                    0, config.data.num_classes, (sample_batch_size,), device=config.device
                )
                if config.guidance is not None
                else None
            )
            samples = sampler.sample(model, sample_x0, sample_lbls, config, num_sampler_steps)
            samples = reverse_transform(samples).to(config.device)
            fid.update(samples, real=False)
            kid.update(samples, real=False)

            done = batch_idx + 1
            elapsed = time.time() - t_gen
            eta = elapsed / done * (num_sample_batches - done)
            logger.info(
                "FID/KID: generated batch %d/%d (%.1fs/batch avg, elapsed %.0fs, eta %.0fs)",
                done,
                num_sample_batches,
                elapsed / done,
                elapsed,
                eta,
            )

    kid_mean, kid_std = kid.compute()
    # FID's eigvals on the CPU: cuSOLVER's geev fails on some Colab GPUs
    # (CUSOLVER_STATUS_INTERNAL_ERROR), after the whole sampling run.
    fid = fid.cpu()
    return {
        "fid": float(fid.compute()),
        "kid_mean": float(kid_mean),
        "kid_std": float(kid_std),
        "generation_seconds": round(time.time() - t_gen, 1),
    }


def preprocess_local(t: torch.Tensor, config: Config) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> ImageNet-normalised float (B, 3, 64, 64) for ResNet18."""
    t = t.to(torch.float).to(config.device) / 255.0
    t = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t)
    t = Resize(64)(t)

    return t


def build_resnet_feature_extractor() -> torch.nn.Module:
    """Pretrained ResNet18, truncated to its 512-d global-average-pooled embedding."""
    model: torch.nn.Module = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = create_feature_extractor(model, ["avgpool"])
    return model


def extract_features(
    feature_model: torch.nn.Module, images: torch.Tensor, config: Config
) -> torch.Tensor:
    """uint8 (B, 3, H, W) -> (B, 512) ResNet18 features."""
    out: dict[str, torch.Tensor] = feature_model(preprocess_local(images, config))
    return out["avgpool"].flatten(1)


def find_nearest_image(
    diffusion_model: BaseModel, config: Config, paths: Paths, num_sampler_steps: int
) -> dict[str, list[int] | list[float]]:
    """Memorisation check: for one batch of samples, the nearest training image by cosine
    similarity of ResNet18 features -> {"nearest_idx", "nearest_sim"}, one entry per sample."""
    train_features = torch.load(paths.resnet_features_path, weights_only=True)
    feature_model = build_resnet_feature_extractor().to(config.device)
    feature_model.eval()
    diffusion_model.eval()

    noise_schedule = create_schedule(config.noise_schedule)
    sampler = create_sampler(config.sampler, noise_schedule)
    x = torch.randn(
        [config.data.batch_size, *config.data.shape], dtype=torch.float, device=config.device
    )
    lbl_batch = (
        torch.randint(0, config.data.num_classes, (config.data.batch_size,), device=config.device)
        if config.guidance is not None
        else None
    )

    with torch.no_grad():
        samples = sampler.sample(diffusion_model, x, lbl_batch, config, num_sampler_steps)
        samples = reverse_transform(samples)
        sample_features = extract_features(feature_model, samples, config)

    sample_features = torch.nn.functional.normalize(sample_features, dim=1)
    train_features = torch.nn.functional.normalize(train_features, dim=1)

    sims = sample_features @ train_features.T
    nearest_idx = sims.argmax(dim=1)
    nearest_sim = sims.gather(1, nearest_idx.unsqueeze(1)).squeeze(1)

    return {"nearest_idx": nearest_idx.tolist(), "nearest_sim": nearest_sim.tolist()}


def parse_seeds(spec: str) -> list[int]:
    """``--seeds``: ``0-4`` -> [0, 1, 2, 3, 4], ``0,2,5`` -> [0, 2, 5]; the forms combine."""
    seeds: list[int] = []
    try:
        for part in spec.split(","):
            first, _, last = part.partition("-")
            seeds.extend(range(int(first), int(last or first) + 1))
    except ValueError:
        seeds = []
    if not seeds:
        raise argparse.ArgumentTypeError(f"expected seeds like 0-4 or 0,2,5, got {spec!r}")
    return seeds


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    add_root_argument(parser)
    parser.add_argument("--sampler", choices=list(SAMPLER_CONFIGS), default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--num-sampler-steps", type=int, default=1000)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-ch", type=int, default=None)
    parser.add_argument("--num-fid-batches", type=int, default=100)
    parser.add_argument("--max-real-batches", type=int, default=None)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=0)
    seed_group.add_argument("--seeds", type=parse_seeds, default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    overridden = (
        args.sampler is not None
        or args.omega is not None
        or args.no_ema
        or args.epoch is not None
        or args.run_name is not None
        or args.num_ch is not None
        or args.num_sampler_steps != 1000
        or args.seed != 0
        or args.num_fid_batches != 100
        or args.max_real_batches is not None
    )
    if overridden and args.tag is None:
        parser.error(
            "--tag is required alongside --sampler/--omega/--no-ema/--epoch/--run-name/"
            "--num-ch/--num-sampler-steps/a non-zero --seed/--num-fid-batches/--max-real-batches, "
            "so results don't overwrite the base config's eval_results file"
        )
    config = apply_eval_overrides(
        config,
        sampler=args.sampler,
        eta=args.eta,
        omega=args.omega,
        run_name=args.run_name,
        num_ch=args.num_ch,
    )
    # DDPM's sampler would raise on it too, but only after loading the model and data.
    if args.num_sampler_steps != 1000 and isinstance(config.sampler, DDPMSamplerConfig):
        parser.error("--num-sampler-steps doesn't apply to ddpm, which always runs all T steps")
    paths = Paths.from_config(args.root, args.config, config.run_name, tag=args.tag)

    # One results file per seed. A single --seed keeps the plain <config>[_<tag>] name and
    # overwrites it; --seeds suffixes each file with _seed<n> and skips seeds already scored.
    if args.seeds is None:
        runs = [(args.seed, paths)]
    else:
        runs = []
        for seed in args.seeds:
            seed_paths = replace(paths, tag=f"{args.tag}_seed{seed}" if args.tag else f"seed{seed}")
            if os.path.exists(seed_paths.eval_results_path):
                logging.info(f"seed {seed}: {seed_paths.eval_results_path} exists, skipping")
            else:
                runs.append((seed, seed_paths))
        if not runs:
            raise SystemExit(0)

    logging.info(f"eval with config {args.config}")
    logging.info(config)

    ds = load_data(paths.data_cache_dir)

    dataloader_train, dataloader_test = get_dataloaders(ds, config.data.batch_size)
    eval_epoch = resolve_epoch(paths.weights_dir, args.epoch)
    diffusion_model = load_model(config, paths, eval_epoch, force_raw=args.no_ema).to(config.device)
    # Fixed (t, eps) and its own RNG: the same for every seed, and the same draws as the trainer's
    # val/mse_loss at this epoch.
    val_loss, val_loss_by_bucket = validation_loss(
        diffusion_model, create_schedule(config.noise_schedule), dataloader_test, config
    )
    print(f"{val_loss=:.4f}")

    for seed, seed_paths in runs:
        # Seed after load_model's random init, so the sampling noise (and KID's subsets) depend
        # only on the seed.
        torch.manual_seed(seed)

        metrics = compute_fid_kid(
            config,
            dataloader_train,
            diffusion_model,
            num_sample_batches=args.num_fid_batches,
            num_sampler_steps=args.num_sampler_steps,
            max_real_batches=args.max_real_batches,
        )
        print(metrics)
        nearest_neighbor = find_nearest_image(
            diffusion_model, config, seed_paths, args.num_sampler_steps
        )
        print(nearest_neighbor)

        results = {
            "epoch": eval_epoch,
            "sampler": config.sampler.type,
            "num_sampler_steps": args.num_sampler_steps,
            "omega": config.omega_classifier_free,
            "num_fid_batches": args.num_fid_batches,
            "max_real_batches": args.max_real_batches,
            "seed": seed,
            "metrics": metrics,
            "val_loss": {"mean": val_loss, "by_bucket": val_loss_by_bucket},
            "nearest_neighbor": nearest_neighbor,
        }
        os.makedirs(os.path.dirname(seed_paths.eval_results_path), exist_ok=True)
        with open(seed_paths.eval_results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {seed_paths.eval_results_path}")
