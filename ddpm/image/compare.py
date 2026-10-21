"""Score trained checkpoints side by side: params, sampler, FID/KID, generation time."""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field

import torch

from ddpm.image.config import Config, UNetModelConfig, load_config
from ddpm.image.data import get_dataloaders, load_data
from ddpm.image.eval import apply_eval_overrides, compute_fid_kid, load_model
from ddpm.paths import Paths, add_root_argument, resolve_epoch


def parse_variant(spec: str) -> tuple[str, str, dict[str, str]]:
    """'label=config_path[,key=value,...]' -> (label, config_path, overrides).

    Keys: steps, sampler, eta, omega (``apply_eval_overrides``), ema=false (raw weights), epoch,
    run_name (another run's checkpoints), num_ch (the UNet width the checkpoint was trained at).
    """
    label, _, rest = spec.partition("=")
    config_path, _, override_str = rest.partition(",")
    if not label or not config_path:
        raise argparse.ArgumentTypeError(
            f"expected LABEL=CONFIG_PATH[,key=value,...], got {spec!r}"
        )
    overrides = {}
    for pair in filter(None, override_str.split(",")):
        key, _, value = pair.partition("=")
        if not key or not value:
            raise argparse.ArgumentTypeError(f"expected key=value in {spec!r}, got {pair!r}")
        overrides[key] = value
    return label, config_path, overrides


@dataclass
class Variant:
    label: str
    config: Config
    paths: Paths
    steps: int | None
    force_raw: bool
    epoch: int | None = None
    # Config path + overrides, stored in the result so a re-run can match it (``find_reusable``).
    spec: dict[str, str] = field(default_factory=dict)


def find_reusable(
    previous: list[dict[str, object]], label: str, expected: dict[str, object]
) -> dict[str, object] | None:
    """The earlier result with this label whose fields all equal ``expected``, else None."""
    for result in previous:
        if result.get("variant") == label and all(result.get(k) == v for k, v in expected.items()):
            return result
    return None


def load_previous(out_path: str) -> list[dict[str, object]]:
    if not os.path.exists(out_path):
        return []
    with open(out_path) as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, list) else []


def write_results(out_path: str, results: list[dict[str, object]]) -> None:
    # Write-then-rename: a session dying mid-write can't leave a truncated JSON.
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, out_path)


def run_variant(
    variant: Variant,
    ds: object,
    num_fid_batches: int,
    max_real_batches: int | None,
    seed: int,
) -> dict[str, object]:
    """Score one variant's checkpoint -> its results row (protocol fields + metrics)."""
    log = logging.getLogger(variant.label)
    epoch = resolve_epoch(variant.paths.weights_dir, variant.epoch)
    log.info("loading checkpoint from epoch %d", epoch)

    config = variant.config
    dataloader_train, dataloader_test = get_dataloaders(ds, config.data.batch_size)
    model = load_model(config, variant.paths, epoch, force_raw=variant.force_raw).to(config.device)
    params = sum(p.numel() for p in model.parameters())

    log.info("%s params, scoring FID/KID", f"{params:,}")
    # Reseed after load_model's random init: every variant gets the same sampling noise and the
    # same real-image subset (common random numbers), whatever its place in the --variant list.
    torch.manual_seed(seed)
    metrics = compute_fid_kid(
        config,
        dataloader_train,
        model,
        num_sample_batches=num_fid_batches,
        num_sampler_steps=variant.steps or 1000,
        max_real_batches=max_real_batches,
    )
    log.info("done -> %s", metrics)

    return {
        "variant": variant.label,
        "params": params,
        "sampler": config.sampler.type,
        "num_sampler_steps": variant.steps or 1000,
        "omega": config.omega_classifier_free,
        "epoch": epoch,
        "num_fid_batches": num_fid_batches,
        "max_real_batches": max_real_batches,
        "seed": seed,
        "spec": variant.spec,
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        required=True,
        dest="variants",
        metavar="LABEL=CONFIG_PATH[,key=value,...]",
    )
    add_root_argument(parser)
    parser.add_argument("--num-fid-batches", type=int, default=20)
    parser.add_argument("--max-real-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="compare_checkpoints.json")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    max_real_batches = (
        args.max_real_batches if args.max_real_batches is not None else args.num_fid_batches
    )

    variants = []
    for label, path, overrides in args.variants:
        config = load_config(path)
        steps = int(overrides["steps"]) if "steps" in overrides else None
        eta = float(overrides["eta"]) if "eta" in overrides else None
        omega = float(overrides["omega"]) if "omega" in overrides else None
        config = apply_eval_overrides(
            config, sampler=overrides.get("sampler"), eta=eta, omega=omega
        )
        force_raw = overrides.get("ema") == "false"
        epoch = int(overrides["epoch"]) if "epoch" in overrides else None
        if "run_name" in overrides:
            config.run_name = overrides["run_name"]
        if "num_ch" in overrides:
            if not isinstance(config.model, UNetModelConfig):
                raise ValueError(f"num_ch override only applies to a UNet model ({label!r})")
            config.model.num_ch = int(overrides["num_ch"])
        paths = Paths.from_config(args.root, path, config.run_name, tag=label)
        spec = {"config": path, **overrides}
        variants.append(Variant(label, config, paths, steps, force_raw, epoch, spec))

    ds = load_data(variants[0].paths.data_cache_dir)
    out_path = variants[0].paths.under_root(args.out)
    previous = [] if args.fresh else load_previous(out_path)

    results: list[dict[str, object]] = []
    for v in variants:
        expected: dict[str, object] = {
            "spec": v.spec,
            "num_fid_batches": args.num_fid_batches,
            "max_real_batches": max_real_batches,
            "seed": args.seed,
        }
        done = find_reusable(previous, v.label, expected)
        if done is not None:
            logging.info("%s: already scored in %s, reusing", v.label, out_path)
            results.append(done)
        else:
            results.append(run_variant(v, ds, args.num_fid_batches, max_real_batches, args.seed))
        write_results(out_path, results)

    print(f"\n{'=' * 80}\nRESULTS\n{'=' * 80}")
    print(
        f"{'variant':10} {'params':>10} {'sampler':6} {'steps':>6} "
        f"{'gen_s':>8} {'FID':>8} {'KID_mean':>10}"
    )
    for r in results:
        print(
            f"{r['variant']:10} {r['params']:>10,} {r['sampler']:6} {r['num_sampler_steps']:>6} "
            f"{r['generation_seconds']:>8.1f} {r['fid']:>8.2f} {r['kid_mean']:>10.4f}"
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
