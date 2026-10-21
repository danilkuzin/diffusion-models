# Tuning a small DDPM

This page records how the baseline config was chosen, and how much one FID score varies. The baseline is a 770K-parameter UNet trained for 200 epochs (78K steps), about 2% of Ho et al.'s parameters and 10% of their training steps. It scores **FID 35.77 ± 0.33**; the distance to their 3.17 is model size and compute. Every number below comes from a recorded run.

On the full protocol, the ladder below went from 70.51 (the corrected 16-channel config) to 45.90 (32 channels) to 35.78 (EMA and learning rate $10^{-3}$), and a retrained 5-seed run of the winner scored 35.77 ± 0.33. The fixes made before the ladder are in the [appendix](#appendix-before-the-ladder).

## Protocols

FID at these sample sizes is biased upwards, so scores are comparable only within one protocol.

| Protocol | Sampler | Generated images | FID std over 5 seeds | Used for |
|---|---|---|---|---|
| Full | DDPM, 1000 steps | 12,800 (100 batches), vs all 50,000 training images | 0.33 | Final numbers |
| 20-batch | DDPM, 1000 steps | 2,560, vs 2,560 training images | 0.87 | Early runs |
| Proxy | DDIM, $\eta = 0$, 50 steps | 1,280 | Not measured; one re-score moved by 1.0 | Ladder decisions |

A gap smaller than about $2\sigma$ of its protocol is not a result.

The proxy costs 1/200 of the full protocol's model calls: 50 sampling steps instead of 1000, on a tenth of the images. It is used only to rank runs. In every check against the full protocol it ranked the runs in the same order, but reported smaller gaps ([limitations](#limitations-and-open-questions)).

## The ladder

| Step | What changed | Result | Decision |
|---|---|---|---|
| Re-baseline | The corrected config: 16 channels (231K params), cosine schedule, lr $2 \times 10^{-4}$ | 68.59 (full) | The reference |
| Noise floor | Nothing trained; five sampling seeds on that checkpoint | std 0.87 (20-batch) | The threshold for the ladder |
| Learning rate | 30 epochs each at $10^{-4}$ / $2 \times 10^{-4}$ / $3 \times 10^{-4}$ / $10^{-3}$ | 214.8 / 173.2 / 143.6 / 145.8 (proxy) | $3 \times 10^{-4}$. At 200 epochs it tied $2 \times 10^{-4}$: 70.51 against 68.59 (full) |
| Width | 16 / 32 / 64 channels | 95.1 / 74.1 / 69.3 (proxy); 32 channels 45.90 (full) | 32 channels, on cost (below) |
| Length | 200 → 400 epochs | No gain (proxy); raw checkpoints differ by up to 18 FID | Stay at 200 |
| EMA | EMA 0.999 at lr $3 \times 10^{-4}$ and $10^{-3}$ | EMA better at every checkpoint; $10^{-3}$ better by 1.5–2.7 (proxy); 35.78 (full) | EMA on, lr $10^{-3}$ |
| Final | Retrain the winner from scratch, score 5 seeds | **35.77 ± 0.33** (full) | The baseline, [`configs/image/ddpm.yaml`](../configs/image/ddpm.yaml) |

### Width

| Channels | Parameters | Time per epoch | Proxy FID |
|---|---|---|---|
| 16 | 231K | 34 s | 95.1 |
| 32 | 770K | 47 s | 74.1 |
| 64 | 2.8M | 69 s | 69.3 |

From 16 to 32 channels, time per epoch grows by 37% and proxy FID falls by 22%. From 32 to 64, time grows by 46% and FID falls by 7%. The 32 → 64 gap, 4.9 FID, is close to the variation between raw checkpoints of one run (below), so the advantage of 64 channels is uncertain.

## EMA

Doubling the training length did not improve FID, but showed that the quality of raw weights varies from one checkpoint to the next. Both EMA runs were therefore scored on the raw and the EMA weights of the same checkpoints, so the weights are the only difference between rows (proxy FID):

| Weights | Epoch 150 | Epoch 175 | Epoch 200 | Spread |
|---|---|---|---|---|
| EMA, lr $3 \times 10^{-4}$ | 71.62 | 71.58 | 70.43 | 1.2 |
| Raw, lr $3 \times 10^{-4}$ | 79.64 | 87.67 | 76.90 | 10.8 |
| EMA, lr $10^{-3}$ | 69.16 | 68.91 | 68.98 | 0.3 |
| Raw, lr $10^{-3}$ | 72.66 | 91.45 | 77.26 | 18.8 |

- **EMA scores better at every checkpoint**, by 3.5 to 22.5 FID.
- **EMA removes the variation between checkpoints:** the spread over three checkpoints falls from 11–19 FID to 0.3–1.2, so a single EMA checkpoint represents its run.
- **With EMA, the higher learning rate is better.** On raw weights it has the widest spread, and the two learning rates cannot be separated.
- **Training runs agree.** The final config, retrained from scratch, scored within 0.1–0.5 FID of the first run at the same checkpoints, and 35.88 against 35.78 on the full protocol at seed 0.

## Sources of noise

| Source | Size (FID) | Measured on |
|---|---|---|
| Raw weights, checkpoint to checkpoint | 3–19 spread | Five checkpoints of one run; three checkpoints each of two runs (proxy) |
| Sampling, 20-batch protocol | std 0.87 | Five seeds, one checkpoint |
| Sampling, full protocol | std 0.33 | Five seeds, the final checkpoint |
| Training run to run, EMA weights | 0.1–0.5 | Two runs of the final config |

Compare EMA weights, never one raw checkpoint against another. On the full protocol, a gap under about 0.7 FID is noise.

## Limitations and open questions

- **Short learning-rate runs.** The 30-epoch ranking did not hold at 200 epochs. The learning rate was settled only on EMA weights.
- **The proxy understates gaps.** From 16 to 32 channels it measured 21.0 where the full protocol measured 24.6; for adding EMA and lr $10^{-3}$, about 4–5 against 10.1. Its sampling noise was not measured. Its scores predate a fix to DDIM's step grid, so they compare with each other, not with current DDIM numbers.
- **Flips and gradient clipping are not ablated.** Both entered the config together with a learning-rate change, and the result was 8 FID worse ([appendix](#appendix-before-the-ladder)). The learning rate explains about 2 FID of that. Training the final config without each of them would attribute the rest.
- **Not tried:** 64 channels at equal wall-clock time, EMA decay 0.9995, training past 200 epochs with EMA, dropout, learning-rate warmup, mixed precision.

## Reproduce

```bash
python -m ddpm.image.train -c configs/image/ddpm.yaml --root /data
python -m scripts.precompute_features -c configs/image/ddpm.yaml --root /data   # once, for eval
python -m ddpm.image.eval -c configs/image/ddpm.yaml --root /data --seeds 0-4
```

On a T4, training takes about 2.6 h (200 epochs at ~47 s, plus checkpoints and previews), and each full-protocol seed about 72 min. The five results files are in [`results/`](../results/). Ladder arms are `ddpm.image.train` overrides (`--lr`, `--num-epochs`, `--num-ch`, each with its own `--run-name`). The proxy is `ddpm.image.compare` with `sampler=ddim,eta=0.0,steps=50` and `--num-fid-batches 10`. The intermediate checkpoints are not published.

## Appendix: before the ladder

<details>
<summary>A misconfigured schedule, and a regression from three simultaneous changes</summary>

### A schedule that spent most of training on noise

The first baseline, a 231K-parameter UNet, used $\beta_t$ from 0.01 to 0.1, ten times Ho et al.'s $10^{-4}$ to 0.02. An input with $\bar\alpha_t < 0.01$ keeps less than a tenth of the image's amplitude; under this schedule, 77% of uniformly drawn training timesteps were like that. Fixing the two endpoints, then switching to cosine, changed nothing else:

| Schedule | $\bar\alpha_t$ halves at | Timesteps with $\bar\alpha_t < 0.01$ | FID (20-batch) |
|---|---|---|---|
| Linear, $\beta_t$ 0.01 → 0.1 | $t = 56$ | 77% | 111.66 |
| Linear, $\beta_t$ $10^{-4}$ → 0.02 | $t = 260$ | 33% | 92.84 |
| Cosine | $t = 496$ | 6.5% | 77.00 |

On the full protocol, the cosine run scores 60.54. The 20-batch protocol reads higher because of finite-sample bias ([evaluation](evaluation.md#calibration)).

The misconfigured run had a lower training loss than the corrected ones. Where $\bar\alpha_t \approx 0$, $x_t \approx \varepsilon$, so predicting the noise means returning the input: a near-zero loss that lowers the mean. **Mean training loss is not comparable across schedules.** The loss per tenth of the timestep range, which the trainer logs as `train/mse_loss_bucket_0` to `_9`, shows the problem: under this schedule, buckets 1–9 sit near zero.

Comparisons made against the misconfigured run partly measured the misconfiguration, and were discarded.

### Three changes at once

The config that started the ladder differs from the cosine run above in three settings: horizontal flips, gradient clipping at 1.0, and a learning rate of $2 \times 10^{-4}$ instead of $10^{-3}$. It scored 68.59 on the full protocol, against the cosine run's 60.54. The EMA runs on the ladder suggest the learning rate accounts for about 2 FID; the rest is unattributed between flips and clipping.

</details>
