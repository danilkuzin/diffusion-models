import argparse
import json
import os
import re
import time
from collections import Counter
from typing import get_args

import torch

from ddpm.paths import Paths, add_root_argument
from ddpm.text import SamplerName, create_sampler
from ddpm.text.config import MDLMObjectiveConfig, TextConfig, load_text_config
from ddpm.text.data import Vocab, decode, download_tiny_shakespeare, encode_file, split, val_chunks
from ddpm.text.model import TextDiT
from ddpm.text.objective import build_model, create_objective
from ddpm.text.train import VAL_SEED, validation_loss


def words(text: str) -> list[str]:
    """Maximal runs of letters, lowercased."""
    return re.findall(r"[a-z]+", text.lower())


def word_validity(samples: list[str], vocab_words: set[str]) -> tuple[float, float]:
    """Per sample, the fraction of its words in `vocab_words`, the first and last dropped (the
    crop cuts them); samples with no word left are skipped -> (mean, stderr over samples)."""
    validities = []
    for s in samples:
        ws = words(s)[1:-1]
        if len(ws) == 0:
            continue
        validities.append(sum(w in vocab_words for w in ws) / len(ws))

    v = torch.tensor(validities, dtype=torch.float64)
    return v.mean().item(), (v.std() / len(v) ** 0.5).item()


def char_entropy(samples: list[str]) -> float:
    """Unigram character entropy of all samples pooled, -sum_c p_c ln p_c, per character."""
    counts = torch.tensor(list(Counter("".join(samples)).values()), dtype=torch.float64)
    p = counts / counts.sum()
    return -(p * p.log()).sum().item()


def sample_metrics(
    model: TextDiT,
    voc: Vocab,
    encoded_train: torch.Tensor,
    encoded_val: torch.Tensor,
    config: TextConfig,
    sampler_name: SamplerName,
    num_steps: int,
    num_samples: int,
    seed: int,
) -> dict[str, object]:
    """num_samples samples in batches of config.data.batch_size -> their word validity and
    entropy, beside num_samples val chunks' and the train text's."""
    sampler = create_sampler(sampler_name)
    generator = torch.Generator(device=config.device).manual_seed(seed)
    B, L = config.data.batch_size, config.data.seq_len

    model.eval()
    batches = []
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, num_samples, B):
            z = torch.full(
                (min(B, num_samples - i), L), voc.mask_id, dtype=torch.long, device=config.device
            )
            batches.append(sampler.sample(model, z, num_steps, voc.mask_id, generator))
    if config.device.startswith("cuda"):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    samples = [decode(row, voc) for row in torch.cat(batches)]

    chunks = val_chunks(encoded_val, L)
    assert num_samples <= chunks.shape[0], f"{chunks.shape[0]} val chunks, {num_samples} asked"
    perm = torch.randperm(chunks.shape[0], generator=torch.Generator().manual_seed(seed))
    val_texts = [decode(row, voc) for row in chunks[perm[:num_samples].to(chunks.device)]]

    train_text = decode(encoded_train, voc)
    vocab_words = set(words(train_text))
    validity, validity_stderr = word_validity(samples, vocab_words)
    val_validity, val_validity_stderr = word_validity(val_texts, vocab_words)

    return {
        "sampler": sampler_name,
        "num_steps": num_steps,
        "num_samples": num_samples,
        "seed": seed,
        "model_calls_per_sample": num_steps,
        "seconds": seconds,
        "word_validity": {"mean": validity, "stderr": validity_stderr},
        "entropy": char_entropy(samples),
        "examples": samples[:2],
        "reference": {
            "val_word_validity": {"mean": val_validity, "stderr": val_validity_stderr},
            "val_entropy": char_entropy(val_texts),
            "train_entropy": char_entropy([train_text]),
        },
    }


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
        help="results-file suffix; required with --run-name, --checkpoint or --sampler",
    )
    parser.add_argument(
        "--sampler",
        choices=get_args(SamplerName),
        default=None,
        help="also sample and score the samples: ancestral (a sample from the model), or "
        "confidence-based, sampled or greedy",
    )
    parser.add_argument("--num-samples", type=int, default=256, help="with --sampler")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=256,
        help="with --sampler: reverse steps, one model call each",
    )
    parser.add_argument("--seed", type=int, default=0, help="with --sampler: the sampling seed")
    add_root_argument(parser)
    return parser


def config_from_argv(
    argv: list[str] | None = None,
) -> tuple[TextConfig, Paths, argparse.Namespace]:
    parser = build_parser()
    args = parser.parse_args(argv)
    overridden = args.run_name is not None or args.checkpoint != "best" or args.sampler is not None
    if overridden and args.tag is None:
        parser.error(
            "--tag is required alongside --run-name, --checkpoint or --sampler, "
            "so results don't overwrite the base config's eval_results file"
        )

    config = load_text_config(args.config)
    if args.sampler is not None and not isinstance(config.objective, MDLMObjectiveConfig):
        parser.error("--sampler samples MDLM runs only; the AR baseline has no sampler")
    if args.run_name is not None:
        config.run_name = args.run_name

    paths = Paths.from_config(args.root, args.config, config.run_name, tag=args.tag)
    return config, paths, args


def main() -> None:
    config, paths, args = config_from_argv()

    in_path = download_tiny_shakespeare(paths.text_data_dir)
    voc = Vocab()
    voc.build(in_path)
    encoded = encode_file(in_path, voc, config.device)
    encoded_train, encoded_val = split(encoded, config.data.train_frac)

    diffusion_model, step = load_model(config, paths, args.checkpoint, voc.size())
    objective = create_objective(config, voc.mask_id)

    val_loss, val_stderr = validation_loss(diffusion_model, objective, encoded_val, config)

    results: dict[str, object] = {
        "run_name": config.run_name,
        "checkpoint": args.checkpoint,
        "step": step,
        "objective": config.objective.type,
        "K": config.eval.K,
        "seed": VAL_SEED,
        "n_chunks": val_chunks(encoded_val, config.data.seq_len).shape[0],
        "val_loss": {"mean": val_loss, "stderr": val_stderr},
    }
    if args.sampler is not None:
        results["samples"] = sample_metrics(
            diffusion_model,
            voc,
            encoded_train,
            encoded_val,
            config,
            args.sampler,
            args.num_steps,
            args.num_samples,
            args.seed,
        )

    os.makedirs(os.path.dirname(paths.eval_results_path), exist_ok=True)
    with open(paths.eval_results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"val_loss {val_loss:.4f} ± {val_stderr:.4f}")
    if args.sampler is not None:
        print(json.dumps(results["samples"], indent=2))
    print(f"wrote {paths.eval_results_path}")


if __name__ == "__main__":
    main()
