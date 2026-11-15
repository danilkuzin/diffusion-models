import argparse
import json
import os

import torch

from ddpm.paths import Paths, add_root_argument
from ddpm.text.config import TextConfig, load_text_config
from ddpm.text.data import Vocab, download_tiny_shakespeare, encode_file, split, val_chunks
from ddpm.text.model import TextDiT
from ddpm.text.objective import build_model, create_objective
from ddpm.text.train import VAL_SEED, validation_loss


def load_model(config: TextConfig, paths: Paths, checkpoint: str, V: int) -> tuple[TextDiT, int]:
    """`best.pt` or `weights_<checkpoint>.pt` -> (model, step). Built from `config`, not the
    checkpoint's, so a run trained with an architecture override fails in load_state_dict."""
    c_name = "best.pt" if checkpoint == "best" else f"weights_{checkpoint}.pt"
    state = torch.load(f"{paths.weights_dir}/{c_name}", weights_only=True, map_location="cpu")

    model = build_model(config, V).to(config.device)
    model.load_state_dict(state["model_state_dict"])

    return model, int(state["step"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument(
        "--run-name", default=None, help="checkpoint dir to load (default: config's)"
    )
    parser.add_argument("--checkpoint", default="best", help="best (the lowest val loss) or a step")
    parser.add_argument(
        "--tag",
        default=None,
        help="results-file suffix; required with --run-name or --checkpoint",
    )
    add_root_argument(parser)
    return parser


def config_from_argv(argv: list[str] | None = None) -> tuple[TextConfig, Paths, str]:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.run_name is not None or args.checkpoint != "best") and args.tag is None:
        parser.error(
            "--tag is required alongside --run-name or --checkpoint, "
            "so results don't overwrite the base config's eval_results file"
        )

    config = load_text_config(args.config)
    if args.run_name is not None:
        config.run_name = args.run_name

    paths = Paths.from_config(args.root, args.config, config.run_name, tag=args.tag)
    return config, paths, args.checkpoint


def main() -> None:
    config, paths, checkpoint = config_from_argv()

    in_path = download_tiny_shakespeare(paths.text_data_dir)
    voc = Vocab()
    voc.build(in_path)
    encoded = encode_file(in_path, voc, config.device)
    _, encoded_val = split(encoded, config.data.train_frac)

    diffusion_model, step = load_model(config, paths, checkpoint, voc.size())
    objective = create_objective(config, voc.mask_id)

    val_loss, val_stderr = validation_loss(diffusion_model, objective, encoded_val, config)

    results = {
        "run_name": config.run_name,
        "checkpoint": checkpoint,
        "step": step,
        "objective": config.objective.type,
        "K": config.eval.K,
        "seed": VAL_SEED,
        "n_chunks": val_chunks(encoded_val, config.data.seq_len).shape[0],
        "val_loss": {"mean": val_loss, "stderr": val_stderr},
    }

    os.makedirs(os.path.dirname(paths.eval_results_path), exist_ok=True)
    with open(paths.eval_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"val_loss {val_loss:.4f} ± {val_stderr:.4f}")
    print(f"wrote {paths.eval_results_path}")


if __name__ == "__main__":
    main()
