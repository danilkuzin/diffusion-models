# Evaluation

Every score in this repo is FID (Heusel et al. 2017) and KID (Bińkowski et al. 2018) between generated images and the CIFAR-10 training set, on torchmetrics' Inception features. At the sample sizes used here, both rank runs scored under the same protocol; neither compares with published CIFAR-10 numbers.

## Metrics

FID fits a Gaussian to each set's 2048-d Inception features and takes the Fréchet distance between the two:

$$\mathrm{FID} = \lVert \mu_r - \mu_g \rVert^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2\,(\Sigma_r \Sigma_g)^{1/2}\right).$$

With $\mu$ and $\Sigma$ estimated from finite samples, FID is biased upwards: two independent samples of one distribution score above 0, and the bias grows as the samples shrink.

KID is the squared maximum mean discrepancy between the two feature sets under the polynomial kernel $k(x, y) = (x^\top y / d + 1)^3$, averaged over random subsets. It is unbiased, and comes with its spread across subsets (`kid_std` in the results).

The two can rank close variants differently: at 250 DDIM steps, FID ties DDIM with DDPM while KID ranks DDIM better ([README](../README.md#results)). Report both, and treat a claim that only one of them supports as a claim about that metric.

## Protocol

| | |
|---|---|
| Sampler | DDPM, all $T = 1000$ steps |
| Generated images | 100 batches of 128 = 12,800, with a fresh $x_T \sim \mathcal N(0, I)$ for every batch |
| Real images | All 50,000 training images |
| Weights | EMA, at the last checkpoint (epoch 200) |
| Seeds | 0–4, one results file each |
| Metrics | torchmetrics' `FrechetInceptionDistance` and `KernelInceptionDistance` at their defaults, on uint8 images |

Each results JSON records its protocol next to the metrics. The baseline's five files are in [`results/`](../results/).

The real images come from the training split, the convention in the DDPM literature. FID then measures distance to the data the model was fit on, and cannot tell a good model from one that reproduces its training set; the [memorisation check](#memorisation-check) covers that.

## Calibration

FID has no absolute scale, and its bias depends on sample size, so the scale is measured on this pipeline, at this protocol's sizes. `scripts/calibrate_fid.py` scores three pairs of image sets that involve no model. $n$ is the generated side's size: 12,800 for the full protocol, 2,560 for 20 batches.

| Comparison | Sets | FID, $n = 12{,}800$ | FID, $n = 2{,}560$ |
|---|---|---|---|
| Train vs train | `train[:n]` against the other $50{,}000 - n$ training images | 2.76 | 11.31 |
| Train vs test | All 10,000 test images against all 50,000 training images | 3.15 | 3.15 |
| Train vs noise | $n$ clamped Gaussian-noise images against all 50,000 training images | 451.38 | 453.29 |

The first two rows are the floor: pairs of samples from one distribution. The test row uses fixed sets, so it does not depend on $n$. The last row is the ceiling.

- **A perfect model would score about 3, not 0.** Its samples would be new draws from the data distribution, independent of the training set, like the test split. Two such samples differ only by finite-sample bias, which at the full protocol's sizes is 2.8–3.2 FID.
- **35.77 lies far from both ends:** about 13 times the floor, and under a tenth of the way to the ceiling, the range where FID discriminates between models.
- **Smaller samples raise the floor, not the ceiling.** At $n = 2{,}560$, two parts of the training set score 11.31, 8.5 FID above the full protocol's floor; the ceiling moves from 451 to 453. So scores from different budgets are never compared: one checkpoint scored 83.18 on the 20-batch protocol and 68.59 on the full one. `ddpm.image.compare`'s default also caps the real side at 2,560 images, which raises its floor further.

`python -m scripts.calibrate_fid -c configs/image/ddpm.yaml --root /data` reproduces the $n = 12{,}800$ column, and `--num-fid-batches 20` the other, in about 10 minutes on a GPU.

## Sampling noise

The same weights, scored at five sampling seeds:

| Protocol | Generated images | FID std | KID std |
|---|---|---|---|
| Full | 12,800 | 0.33 | 0.0003 |
| 20 batches, against 20 real batches | 2,560 | 0.87 | 0.00085 |

On the full protocol, a gap under about 0.7 FID ($2\sigma$) is not a result.

Sampling is not the largest source of noise. Raw (non-EMA) checkpoints of one run differ by 3–19 FID, while the EMA weights of two independent training runs differ by 0.1–0.5 ([tuning](tuning.md#sources-of-noise)). Compare EMA weights, or several checkpoints, never one raw checkpoint against another.

## Sample preparation

- **Pixel range.** Inception expects uint8 in $[0, 255]$, the format the real images arrive in. Samples in $[-1, 1]$ are clamped and mapped to $(x + 1) \cdot 127.5$ in uint8 (`reverse_transform`) before scoring. Floats in $[-1, 1]$ would give a plausible but meaningless score.
- **Starting noise per batch.** $x_T$ is redrawn for every batch. With one fixed $x_T$, a deterministic sampler (DDIM at $\eta = 0$) would return the same 128 images in every batch, and the score would cover 128 distinct samples instead of 12,800.

## Memorisation check

After FID and KID, `ddpm.image.eval` samples one more batch of 128 images and finds each sample's nearest training image by cosine similarity of ResNet18 features (ImageNet weights, the 512-d average-pooled layer). This extractor is smaller than FID's, so that the features of all 50,000 training images fit in memory. `scripts/precompute_features.py` builds this feature bank once, in dataset order, so each `nearest_idx` in the results JSON is a training-set index, and `nearest_sim` is its similarity.

A similarity near 1 means the sample nearly reproduces a training image. The check covers 128 samples, has no threshold, and works in an ImageNet feature space, so it detects only gross memorisation. A sample shown next to `train[nearest_idx]` says more than the similarity does.

## Comparing runs

`ddpm.image.eval` scores one config. `ddpm.image.compare` scores several variants side by side:

```bash
python -m ddpm.image.compare --root /data \
    --variant ddpm=configs/image/ddpm.yaml \
    --variant ddim_50=configs/image/ddpm.yaml,sampler=ddim,eta=0.0,steps=50 \
    --variant raw=configs/image/ddpm.yaml,ema=false \
    --num-fid-batches 100 --max-real-batches 391 --out compare_samplers.json
```

- **A variant** is a trained config plus eval-only overrides: `steps`, `sampler`, `eta`, `omega`, `ema=false`, `epoch`, `run_name`, `num_ch`.
- **Common random numbers.** Every variant is reseeded before scoring, so all the variants in one invocation share their starting noise and real images. Their differences are less noisy than independent scores would be.
- **The default budget is small:** 20 generated batches against 20 real ones. `--num-fid-batches 100 --max-real-batches 391` matches the full protocol (391 batches of 128 cover the 50,000 training images).
- **Resumable.** `--out` is rewritten after every variant, and a re-run reuses the rows already in it (`--fresh` re-scores everything).
- **Across invocations.** Reseeding makes every variant at a given seed draw the same starting noise and real images, whichever invocation it runs in, so rows from separate runs at one seed share them too. Timings don't carry over: the same 1000 DDPM calls took 70 minutes in one Colab session and 85 in another.
- **Cost is model calls, not steps.** Predictor–corrector makes two per step, and classifier-free guidance doubles every call. Compare samplers at equal calls.

Before calling a gap a result, check that both sides used the same protocol and EMA weights, and that the gap is above $2\sigma$ for that protocol.

The README's DDIM rows are one such run, on the full protocol at seed 0 ([`results/compare_ddim.json`](../results/compare_ddim.json)). Its DDPM row reproduces the baseline's seed-0 score, 35.88. Its timings include Inception's fixed cost per batch, so the speed-ups trail the step ratios: 13× for 20× fewer steps. The VP-SDE and flow rows are two more runs at seed 0, on their own checkpoints ([`results/compare_vpsde.json`](../results/compare_vpsde.json), [`results/compare_flow.json`](../results/compare_flow.json)); the VP run's DDPM row reproduces 35.88 again. All of these rows take their noise level from the DDPM baseline's seed stds, 0.33 FID and 0.0003 KID; their own were not measured.

### The README's runs

One `ddpm.image.compare` run per results table, each on the full protocol at seed 0. The VP run's `ddpm` variant is the anchor that reproduces the seed-0 DDPM row.

```bash
# The seed-0 DDPM and DDIM rows
python -m ddpm.image.compare --root /data \
    --variant ddpm=configs/image/ddpm.yaml \
    --variant ddim_10=configs/image/ddpm.yaml,sampler=ddim,eta=0.0,steps=10 \
    --variant ddim_20=configs/image/ddpm.yaml,sampler=ddim,eta=0.0,steps=20 \
    --variant ddim_50=configs/image/ddpm.yaml,sampler=ddim,eta=0.0,steps=50 \
    --variant ddim_100=configs/image/ddpm.yaml,sampler=ddim,eta=0.0,steps=100 \
    --variant ddim_250=configs/image/ddpm.yaml,sampler=ddim,eta=0.0,steps=250 \
    --variant ddimeta1_50=configs/image/ddpm.yaml,sampler=ddim,eta=1.0,steps=50 \
    --variant ddimeta1_100=configs/image/ddpm.yaml,sampler=ddim,eta=1.0,steps=100 \
    --num-fid-batches 100 --max-real-batches 391 --out compare_ddim.json

# The VP SDE rows
python -m ddpm.image.compare --root /data \
    --variant pfode_10=configs/image/vpsde.yaml,sampler=pfode,steps=10 \
    --variant pfode_20=configs/image/vpsde.yaml,sampler=pfode,steps=20 \
    --variant pfode_50=configs/image/vpsde.yaml,sampler=pfode,steps=50 \
    --variant pfode_100=configs/image/vpsde.yaml,sampler=pfode,steps=100 \
    --variant euler_vp_100=configs/image/vpsde.yaml,steps=100 \
    --variant pc_500=configs/image/vpsde.yaml,sampler=pc,steps=500 \
    --variant euler_vp_1000=configs/image/vpsde.yaml \
    --variant pfode_1000=configs/image/vpsde.yaml,sampler=pfode \
    --variant ddpm=configs/image/ddpm.yaml \
    --num-fid-batches 100 --max-real-batches 391 --out compare_vpsde.json

# The rectified-flow rows
python -m ddpm.image.compare --root /data \
    --variant flow_1=configs/image/flow.yaml,steps=1 \
    --variant flow_2=configs/image/flow.yaml,steps=2 \
    --variant flow_5=configs/image/flow.yaml,steps=5 \
    --variant flow_10=configs/image/flow.yaml,steps=10 \
    --variant flow_20=configs/image/flow.yaml,steps=20 \
    --variant flow_50=configs/image/flow.yaml,steps=50 \
    --variant flow_100=configs/image/flow.yaml,steps=100 \
    --variant flow_1000=configs/image/flow.yaml \
    --num-fid-batches 100 --max-real-batches 391 --out compare_flow.json
```
