# diffusion-models

A from-scratch PyTorch implementation of discrete-time diffusion models, DDPM and DDIM, on CIFAR-10. The code follows the papers and their reference implementations; the aim is readability and checkable math, not production code or competitive FID.

- [Tuning](docs/tuning.md): how the baseline was found, and how noisy its score is.
- [Evaluation](docs/evaluation.md): the FID protocol, its calibration, and how to compare runs.
- [Architecture](docs/architecture.md): the models, the training loop, and the math the tests check.

## Scope

On CIFAR-10, this release covers:

- DDPM training
- DDPM and DDIM sampling
- linear and cosine noise schedules
- UNet and DiT backbones
- EMA
- classifier-free guidance

The model has 770K parameters, about 2% of Ho et al.'s 35.7M, so that it trains on one consumer GPU or in Colab.

## Results

| Model | Sampler | Steps | Time | FID | KID | Notes |
|---|---|---|---|---|---|---|
| UNet, 770K | DDPM | 1000 | 1 | **35.77 ± 0.33** | **0.0257 ± 0.0003** | The baseline; mean ± std over seeds 0–4 |
| UNet, 770K | DDPM | 1000 | 1 | 35.88 | 0.0260 | Seed 0; the reference for the rows below |
| UNet, 770K | DDIM, $\eta = 0$ | 10 | 1/28 | 44.09 | 0.0385 | |
| UNet, 770K | DDIM, $\eta = 0$ | 20 | 1/22 | 38.75 | 0.0343 | |
| UNet, 770K | DDIM, $\eta = 0$ | 50 | 1/13 | 36.45 | 0.0294 | FID within noise of DDPM; KID's gap is 11 seed stds |
| UNet, 770K | DDIM, $\eta = 0$ | 100 | 1/8 | 35.84 | 0.0262 | Matches DDPM on both metrics |
| UNet, 770K | DDIM, $\eta = 0$ | 250 | 1/3.7 | 36.07 | 0.0238 | FID ties DDPM; KID ranks DDIM better |
| UNet, 770K | DDIM, $\eta = 1$ | 50 | 1/13 | 39.60 | 0.0377 | 3 FID behind $\eta = 0$ |
| UNet, 770K | DDIM, $\eta = 1$ | 100 | 1/8 | 35.93 | 0.0322 | Ties $\eta = 0$ on FID, not on KID |

- **Protocol.** EMA weights at epoch 200 of [`configs/image/ddpm.yaml`](configs/image/ddpm.yaml); 12,800 samples against all 50,000 training images. FID at this size is relative: it ranks runs in this repo, and doesn't compare with published numbers. For scale, two disjoint parts of the training set score 2.76, the test split 3.15, and pure noise 451 ([calibration](docs/evaluation.md#calibration)).
- **Noise.** Over sampling seeds, FID's std is 0.33 and KID's 0.0003, so a FID gap under about 0.7 is not a result. The DDIM rows are one run at seed 0, scored against the seed-0 DDPM row with the same starting noise.
- **Time** is sampling plus Inception features, relative to DDPM's 70 minutes in the same run.
- **Data:** the baseline's five results files and the DDIM run's [`compare_ddim.json`](results/compare_ddim.json) are in [`results/`](results/).

### Samples

![Samples from the baseline](docs/samples.png)

DDPM, 1000 steps, EMA weights at epoch 200.

## Method

**Model.** Every network, UNet or DiT, predicts the noise $\varepsilon_\theta(x_t, t, y)$ from a batch of noisy images $x_t$, the timestep $t$ and optional class labels $y$.

**Training.** Draw $t$ and $\varepsilon \sim \mathcal N(0, I)$, form $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$, and minimise $\lVert \varepsilon - \varepsilon_\theta(x_t, t) \rVert^2$.

**Schedule.** The schedule sets how fast $\bar\alpha_t$, the fraction of the signal kept, falls with $t$: linear (Ho et al.) or cosine (Nichol & Dhariwal).

**Sampling.** A sampler, DDPM or DDIM, applies $\varepsilon_\theta$ repeatedly on its own grid of timesteps, from pure noise $x_T$ to an image $x_0$.

[Architecture](docs/architecture.md) has the derivations and the figures.

## Usage

Install from `requirements.txt`, or use Docker, with the repo mounted at `/source` and a data folder at `/data`:

```bash
docker build -t diffusion-models .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD:/source -v <path_to_data_folder>:/data diffusion-models
```

Every entry point writes its artifacts under `--root`; the commands below use `/data`.

- **Tests:** `pytest`, on a CPU, in seconds.
- **Figures:** `make figures` redraws `docs/figures/` from the tests.

The commands behind the results table:

```bash
# Train the baseline
python -m ddpm.image.train -c configs/image/ddpm.yaml --root /data

# Build the memorisation check's feature bank, once, before any eval
python -m scripts.precompute_features -c configs/image/ddpm.yaml --root /data

# The baseline row: seeds 0-4
python -m ddpm.image.eval -c configs/image/ddpm.yaml --root /data --seeds 0-4

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

# The FID scale
python -m scripts.calibrate_fid -c configs/image/ddpm.yaml --root /data
```

## What's next

- **Comparisons on this baseline:** UNet against DiT ([`configs/image/dit.yaml`](configs/image/dit.yaml), 765K parameters on the UNet's recipe, not yet scored), classifier-free guidance and its $\omega$, EMA against raw weights.
- **Later releases:** score-based SDEs, flow matching, text diffusion.

## Papers

- Ho et al. 2020 (DDPM): training, ancestral sampling, the linear schedule.
- Song et al. 2021 (DDIM): the sampler.
- Nichol & Dhariwal 2021: the cosine schedule.
- Ho & Salimans 2022: classifier-free guidance.
- Peebles & Xie 2023 (DiT): the transformer backbone and adaLN-Zero.
- Saharia et al. 2022: static thresholding.
- Heusel et al. 2017 (FID) and Bińkowski et al. 2018 (KID).

## License

MIT
