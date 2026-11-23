import argparse
from typing import get_args

import torch

from ddpm.paths import Paths, add_root_argument
from ddpm.text import SamplerName, create_sampler
from ddpm.text.config import (
    MDLMObjectiveConfig,
    TextConfig,
    load_text_config,
)
from ddpm.text.data import Vocab, decode, download_tiny_shakespeare
from ddpm.text.eval import load_model


def generate(
    config: TextConfig,
    paths: Paths,
    checkpoint: str,
    num_samples: int,
    num_steps: int,
    seed: int,
    sampler_name: SamplerName,
) -> None:
    in_path = download_tiny_shakespeare(paths.text_data_dir)
    voc = Vocab()
    voc.build(in_path)

    assert voc.size() == 65, f"expected Tiny Shakespeare's 65 characters, got {voc.size()}"

    model, step = load_model(config, paths, checkpoint, voc.size())
    model.eval()

    generator = torch.Generator(device=config.device).manual_seed(seed)
    z = torch.full(
        (num_samples, config.data.seq_len), voc.mask_id, dtype=torch.long, device=generator.device
    )
    sampler = create_sampler(sampler_name)
    with torch.no_grad():
        generated = sampler.sample(
            model,
            z,
            num_steps,
            voc.mask_id,
            generator,
        )
    for i in range(generated.shape[0]):
        print(f"--- sample {i} ({sampler_name}, num_steps={num_steps}, step {step}) ---")
        print(decode(generated[i], voc))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    add_root_argument(parser)
    parser.add_argument("--checkpoint", default="best", help="best (the lowest val loss) or a step")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--num-steps", type=int, default=256, help="reverse steps, one model call each"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-name", type=str, default=None, help="checkpoint dir to load (default: config's)"
    )
    parser.add_argument(
        "--sampler",
        choices=get_args(SamplerName),
        default="ancestral",
        help="ancestral (a sample from the model), or confidence-based, sampled or greedy",
    )

    return parser


def config_from_argv(
    argv: list[str] | None = None,
) -> tuple[TextConfig, Paths, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_text_config(args.config)
    if args.run_name is not None:
        config.run_name = args.run_name
    if not isinstance(config.objective, MDLMObjectiveConfig):
        parser.error(
            "generate samples MDLM runs only; the AR baseline is a reference for MDLM's bound, "
            "with no sampler"
        )
    paths = Paths.from_config(args.root, args.config, config.run_name)
    return config, paths, args


if __name__ == "__main__":
    config, paths, args = config_from_argv()
    generate(
        config, paths, args.checkpoint, args.num_samples, args.num_steps, args.seed, args.sampler
    )
