import argparse
import math
import os
import time
from typing import Any

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from ddpm.paths import Paths, add_root_argument, find_latest_checkpoint
from ddpm.text.config import (
    MDLMObjectiveConfig,
    TextConfig,
    TextTrainConfig,
    load_text_config,
)
from ddpm.text.data import (
    Vocab,
    download_tiny_shakespeare,
    encode_file,
    sample_batch,
    split,
    val_chunks,
)
from ddpm.text.model import TextDiT
from ddpm.text.objective import Objective, build_model, create_objective

PRINT_EVERY = 100
NUM_LOSS_BUCKETS = 10
VAL_SEED = 0


def lr_schedule(step: int, config_train: TextTrainConfig) -> float:
    """Linear warmup to lr, then cosine decay to min_lr_frac * lr at the last step."""
    if step < config_train.warmup_steps:
        return config_train.lr * (step + 1) / config_train.warmup_steps
    min_lr = config_train.min_lr_frac * config_train.lr
    return min_lr + 0.5 * (config_train.lr - min_lr) * (
        1
        + math.cos(
            math.pi
            * (step - config_train.warmup_steps)
            / (config_train.steps - config_train.warmup_steps)
        )
    )


def build_optimizer(model: TextDiT, config: TextConfig) -> torch.optim.Optimizer:
    """AdamW, with weight decay on matrices and embeddings only, not on biases or norms."""
    decay_params = []
    no_decay_params = []

    for _, param in model.named_parameters():
        if not param.requires_grad:  # the frozen Fourier W
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.train.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.train.lr,
        betas=(0.9, config.train.beta2),
    )

    return optimizer


def unigram_entropy(encoded: torch.Tensor, voc_size: int) -> float:
    """Entropy of the character frequencies, per character: the loss of a model that ignores
    context, for the NELBO too, since (1/t) * tL * H / L = H."""
    counts = torch.bincount(encoded.cpu(), minlength=voc_size).float()
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def bucket_means(loss_sum: torch.Tensor, loss_count: torch.Tensor) -> str:
    """Mean loss per tenth of t, '-' for an empty one."""
    means = loss_sum / loss_count.clamp(min=1)
    return " ".join(f"{m:.2f}" if c > 0 else "-" for m, c in zip(means, loss_count, strict=True))


def validation_loss(
    model: TextDiT, objective: Objective, encoded_val: torch.Tensor, config: TextConfig
) -> tuple[float, float]:
    """Val NELBO (MDLM) / NLL (the AR baseline) per character over every val chunk, K draws each,
    the same draws on every call -> (mean, stderr over chunks)."""

    generator = torch.Generator(device=encoded_val.device).manual_seed(VAL_SEED)
    B, L = config.data.batch_size, config.data.seq_len
    model.eval()
    total = []
    with torch.no_grad():
        chunks = val_chunks(encoded_val, L)  # (n, L)
        n, _ = chunks.shape
        for i in range(0, n, B):
            x = chunks[i : i + B]  # (B, L)
            total.append(
                torch.stack(
                    [
                        objective.loss_stats(x, generator, model)["loss_per_seq"]
                        for _ in range(config.eval.K)
                    ]
                ).mean(dim=0)
            )  # (B,)

    total_t = torch.cat(total)
    return total_t.mean().item(), total_t.std().item() / math.sqrt(n)


def apply_text_overrides(
    config: TextConfig,
    *,
    steps: int | None = None,
    lr: float | None = None,
    dropout: float | None = None,
    embed_dim: int | None = None,
    depth: int | None = None,
    heads: int | None = None,
    gate_init: float | None = None,
    no_time_cond: bool = False,
    run_name: str | None = None,
) -> TextConfig:
    """The CLI's training overrides. Each changes what gets trained, so `config_from_argv`
    requires a new `run_name` with any of them: resume is silent."""
    if steps is not None:
        config.train.steps = steps
    if lr is not None:
        config.train.lr = lr
    if dropout is not None:
        config.model.dropout = dropout
    if embed_dim is not None:
        config.model.embed_dim = embed_dim
    if depth is not None:
        config.model.depth = depth
    if heads is not None:
        config.model.heads = heads
    if gate_init is not None:
        config.model.gate_init = gate_init
    if no_time_cond:
        config.model.time_cond = False
    if run_name is not None:
        config.run_name = run_name
    return config


class TextDiffusionTrainer:
    """MDLM training, or the AR baseline's. Every `eval.every` steps: the val loss, and `best.pt`
    when it is the lowest so far; every `log.ckpt_every` steps a checkpoint. Resumes from the run's
    latest."""

    def __init__(
        self,
        model: TextDiT,
        encoded_train: torch.Tensor,
        encoded_val: torch.Tensor,
        writer: SummaryWriter,
        config: TextConfig,
        paths: Paths,
        objective: Objective,
        resume: bool = True,
    ):
        self.model = model
        self.encoded_train = encoded_train
        self.encoded_val = encoded_val
        self.writer = writer
        self.config = config
        self.paths = paths
        self.resume = resume

        self.optimizer = build_optimizer(self.model, config)
        # crop starts are drawn on the CPU; t and the masks on the device
        self.generator_cpu = torch.Generator().manual_seed(config.train.seed)
        self.generator_gpu = torch.Generator(device=config.device).manual_seed(
            config.train.seed + 1
        )
        self.objective = objective
        self.best_val_loss = math.inf

    def train_step(self) -> dict[str, Any]:
        self.model.train()
        x = sample_batch(
            self.encoded_train,
            self.config.data.batch_size,
            self.config.data.seq_len,
            self.generator_cpu,
        )
        stats = self.objective.loss_stats(x, self.generator_gpu, self.model)
        stats["loss"].backward()

        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.config.train.grad_clip
        )
        if not torch.isfinite(total_norm):
            raise RuntimeError(f"non-finite grad norm {total_norm}")
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        stats["grad_norm"] = total_norm
        return stats

    def load_checkpoint(self, path: str) -> int:
        """Restore into already-built objects; returns the number of steps done."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.generator_cpu.set_state(checkpoint["generator_cpu_state"])
        self.generator_gpu.set_state(checkpoint["generator_gpu_state"])
        self.best_val_loss = checkpoint["best_val_loss"]
        return int(checkpoint["step"])

    def save_checkpoint(self, path: str, step: int) -> None:
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "generator_cpu_state": self.generator_cpu.get_state(),
            "generator_gpu_state": self.generator_gpu.get_state(),
            "config": self.config.model_dump(),
            "best_val_loss": self.best_val_loss,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def train(self) -> None:
        start = 0
        if self.resume:
            if (latest := find_latest_checkpoint(self.paths.weights_dir)) is not None:
                start = self.load_checkpoint(latest[0])
                print(f"resumed from {latest[0]} at step {start}")

        tokens_per_step = self.config.data.batch_size * self.config.data.seq_len
        is_mdlm = isinstance(self.config.objective, MDLMObjectiveConfig)
        loss_sum = torch.zeros(NUM_LOSS_BUCKETS, device=self.config.device)
        loss_count = torch.zeros(NUM_LOSS_BUCKETS, device=self.config.device)

        tic = time.perf_counter()
        for step in range(start, self.config.train.steps):
            lr = lr_schedule(step, self.config.train)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            stats = self.train_step()

            done = step + 1
            self.writer.add_scalar("train/loss", stats["loss"], done)
            self.writer.add_scalar("train/grad_norm", stats["grad_norm"], done)
            self.writer.add_scalar("train/lr", lr, done)
            if is_mdlm:
                self.writer.add_scalar("train/masked_ce", stats["masked_ce"], done)

            if is_mdlm:
                bucket = (stats["t"] * NUM_LOSS_BUCKETS).long().clamp(max=NUM_LOSS_BUCKETS - 1)
                per_seq = stats["loss_per_seq"].detach()
                loss_sum.index_add_(0, bucket, per_seq)
                loss_count.index_add_(0, bucket, torch.ones_like(per_seq))

            if done % PRINT_EVERY == 0:
                tokens_per_s = PRINT_EVERY * tokens_per_step / (time.perf_counter() - tic)
                self.writer.add_scalar("train/tokens_per_s", tokens_per_s, done)
                line = (
                    f"step {done} loss {stats['loss'].item():.4f} lr {lr:.2e} "
                    f"grad_norm {stats['grad_norm'].item():.3f} "
                    f"tokens/s {tokens_per_s:.0f}"
                )
                if is_mdlm:
                    line += f" masked_ce {stats['masked_ce'].item():.4f}"
                print(line)
                if is_mdlm:
                    # Low-t buckets (few masks, lots of context) must sit well below high-t ones.
                    # A flat row means the model is not using its context.
                    means = loss_sum / loss_count.clamp(min=1)
                    print(f"  t-buckets 0.0-1.0: {bucket_means(loss_sum, loss_count)}")
                    for i, m in enumerate(means.tolist()):
                        self.writer.add_scalar(f"train/loss_bucket_{i}", m, done)
                    loss_sum.zero_()
                    loss_count.zero_()
                tic = time.perf_counter()
            if done % self.config.eval.every == 0 or done == self.config.train.steps:
                val_loss, val_stderr = validation_loss(
                    self.model, self.objective, self.encoded_val, self.config
                )
                print(f"step {done} val/loss {val_loss:.4f} val/stderr {val_stderr:.4f}")
                self.writer.add_scalar("val/loss", val_loss, done)
                self.writer.add_scalar("val/stderr", val_stderr, done)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f"{self.paths.weights_dir}/best.pt", done)

                tic = time.perf_counter()
            if done % self.config.log.ckpt_every == 0 or done == self.config.train.steps:
                self.save_checkpoint(f"{self.paths.weights_dir}/weights_{done}.pt", done)
                tic = time.perf_counter()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    add_root_argument(parser)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore any existing checkpoint in the run's weights dir and start from step 0",
    )
    parser.add_argument("--steps", type=int, default=None, help="override train.steps")
    parser.add_argument("--lr", type=float, default=None, help="override train.lr")
    parser.add_argument("--dropout", type=float, default=None, help="override model.dropout")
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=None,
        help="override model.embed_dim (e.g. 128 for a quick run)",
    )
    parser.add_argument("--depth", type=int, default=None, help="override model.depth")
    parser.add_argument("--heads", type=int, default=None, help="override model.heads")
    parser.add_argument(
        "--gate-init",
        type=float,
        default=None,
        help="override model.gate_init (0.0 is DiT's adaLN-Zero)",
    )
    parser.add_argument(
        "--no-time-cond",
        action="store_true",
        help="build the model without time conditioning (an ablation)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="override run_name; required alongside any other override so this run gets its own "
        "checkpoint directory instead of silently resuming into the base config's",
    )
    return parser


def config_from_argv(argv: list[str] | None = None) -> tuple[TextConfig, Paths, bool]:
    """Parse the CLI into (config, paths, resume). Separate from `__main__` so the
    `--run-name`-required rule is testable without a subprocess."""
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = (
        args.steps,
        args.lr,
        args.dropout,
        args.embed_dim,
        args.depth,
        args.heads,
        args.gate_init,
    )
    if (any(o is not None for o in overrides) or args.no_time_cond) and args.run_name is None:
        parser.error(
            "--run-name is required alongside any override, so this run doesn't silently "
            "resume into (and contaminate) the base config's checkpoint directory"
        )

    config = apply_text_overrides(
        load_text_config(args.config),
        steps=args.steps,
        lr=args.lr,
        dropout=args.dropout,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        gate_init=args.gate_init,
        no_time_cond=args.no_time_cond,
        run_name=args.run_name,
    )
    paths = Paths.from_config(args.root, args.config, config.run_name)
    return config, paths, not args.fresh


def main() -> None:
    config, paths, resume = config_from_argv()

    in_path = download_tiny_shakespeare(paths.text_data_dir)
    voc = Vocab()
    voc.build(in_path)
    encoded = encode_file(in_path, voc, config.device)
    encoded_train, encoded_val = split(encoded, config.data.train_frac)
    print(
        f"V={voc.size()} train={encoded_train.numel()} val={encoded_val.numel()} chars; "
        f"unigram floor {unigram_entropy(encoded_train, voc.size()):.3f} "
        f"(uniform {math.log(voc.size()):.3f})"
    )

    torch.manual_seed(config.train.seed)
    model = build_model(config, voc.size()).to(config.device)

    writer = SummaryWriter(log_dir=paths.log_dir)
    objective = create_objective(config, voc.mask_id)

    trainer = TextDiffusionTrainer(
        model, encoded_train, encoded_val, writer, config, paths, objective, resume
    )

    trainer.train()
    writer.close()


if __name__ == "__main__":
    main()
