import argparse

import torch

from ddpm.paths import Paths, add_root_argument
from ddpm.text.config import (
    TextConfig,
    load_text_config,
)
from ddpm.text.data import Vocab, decode, download_tiny_shakespeare
from ddpm.text.eval import load_model
from ddpm.text.sample import sample


def generate(config: TextConfig, paths: Paths, args: argparse.Namespace) -> None:
    in_path = download_tiny_shakespeare(paths.text_data_dir)
    voc = Vocab()
    voc.build(in_path)

    assert voc.size() == 65, f"expected Tiny Shakespeare's 65 characters, got {voc.size()}"

    model, step = load_model(config, paths, args.checkpoint, voc.size())
    model.eval()

    generator = torch.Generator(device=config.device).manual_seed(args.seed)

    with torch.no_grad():
        generated = sample(
            model,
            args.num_samples,
            config.data.seq_len,
            args.num_steps,
            voc.mask_id,
            generator,
        )
    for i in range(generated.shape[0]):
        print(f"--- sample {i} (num_steps={args.num_steps}, step {step}) ---")
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

    return parser


def config_from_argv(
    argv: list[str] | None = None,
) -> tuple[TextConfig, Paths, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_text_config(args.config)
    if args.run_name is not None:
        config.run_name = args.run_name
    if not config.objective.type == "mdlm":
        parser.error(
            "generate samples MDLM runs only; the AR baseline is a reference for MDLM's bound, "
            "with no sampler"
        )
    paths = Paths.from_config(args.root, args.config, config.run_name)
    return config, paths, args


if __name__ == "__main__":
    config, paths, args = config_from_argv()
    generate(config, paths, args)
