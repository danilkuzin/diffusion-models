# diffusion-models

A from-scratch PyTorch implementation of diffusion models on CIFAR-10: discrete-time DDPM and DDIM, score-based SDEs, and rectified flow. The code follows the papers and their reference implementations; the aim is readability and checkable math, not production code or competitive FID.

- [Tuning](docs/tuning.md): how the baseline was found, and how noisy its score is.
- [Evaluation](docs/evaluation.md): the FID protocol, its calibration, and how to compare runs.
- [Architecture](docs/architecture.md): the models, forward processes and their samplers, the training loop, and the math the tests check.

## Scope

On CIFAR-10, this repo covers:

- three forward processes: discrete-time diffusion (linear and cosine schedules), score-based SDEs (VP, sub-VP and VE), and rectified flow
- their samplers: DDPM and DDIM; Euler–Maruyama, predictor–corrector and the probability-flow ODE; Euler on the flow
- UNet and DiT backbones
- EMA
- classifier-free guidance

The model has 770K parameters, about 2% of Ho et al.'s 35.7M, so that it trains on one consumer GPU or in Colab. The discrete, VP-SDE and flow results below train this same network on the same recipe. Sub-VP and VE are implemented and tested, not trained.

## Results

### Discrete time, [`configs/image/ddpm.yaml`](configs/image/ddpm.yaml)

| Sampler | Steps | Model calls | FID | KID | Notes |
|---|---|---|---|---|---|
| DDPM | 1000 | 1000 | **35.77 ± 0.33** | **0.0257 ± 0.0003** | The baseline; mean ± std over seeds 0–4 |
| DDPM | 1000 | 1000 | 35.88 | 0.0260 | Seed 0; the reference for every row below |
| DDIM, $\eta = 0$ | 10 | 10 | 44.09 | 0.0385 | |
| DDIM, $\eta = 0$ | 20 | 20 | 38.75 | 0.0343 | |
| DDIM, $\eta = 0$ | 50 | 50 | 36.45 | 0.0294 | FID within noise of DDPM; KID's gap is 11 seed stds |
| DDIM, $\eta = 0$ | 100 | 100 | 35.84 | 0.0262 | Matches DDPM on both metrics |
| DDIM, $\eta = 0$ | 250 | 250 | 36.07 | 0.0238 | FID ties DDPM; KID ranks DDIM better |
| DDIM, $\eta = 1$ | 50 | 50 | 39.60 | 0.0377 | 3 FID behind $\eta = 0$ |
| DDIM, $\eta = 1$ | 100 | 100 | 35.93 | 0.0322 | Ties $\eta = 0$ on FID, not on KID |

### VP SDE, [`configs/image/vpsde.yaml`](configs/image/vpsde.yaml)

| Sampler | Steps | Model calls | FID | KID | Notes |
|---|---|---|---|---|---|
| Probability-flow ODE | 10 | 10 | 172.41 | 0.1812 | DDIM at 10 steps: 44.09 |
| Probability-flow ODE | 20 | 20 | 88.74 | 0.0804 | |
| Probability-flow ODE | 50 | 50 | 50.82 | 0.0383 | |
| Probability-flow ODE | 100 | 100 | 42.66 | 0.0301 | |
| Probability-flow ODE | 1000 | 1000 | **37.60** | **0.0262** | The best VP row: 1.7 FID behind DDPM, KID tied |
| Euler–Maruyama | 100 | 100 | 68.79 | 0.0579 | The SDE degrades far faster than the ODE |
| Euler–Maruyama | 1000 | 1000 | 39.15 | 0.0290 | 1.55 FID behind the ODE at the same calls |
| Predictor–corrector | 500 | 1000 | 39.27 | 0.0279 | Ties Euler–Maruyama at the same calls |

### Rectified flow, [`configs/image/flow.yaml`](configs/image/flow.yaml)

| Sampler | Steps | Model calls | FID | KID | Notes |
|---|---|---|---|---|---|
| Euler | 1 | 1 | 392.06 | 0.4382 | One step lands on the predicted data mean |
| Euler | 2 | 2 | 198.30 | 0.2011 | |
| Euler | 5 | 5 | 67.63 | 0.0691 | |
| Euler | 10 | 10 | 48.68 | 0.0495 | The VP ODE at 10 steps: 172.41; DDIM: 44.09 |
| Euler | 20 | 20 | 42.29 | 0.0405 | |
| Euler | 50 | 50 | 38.97 | 0.0339 | |
| Euler | 100 | 100 | **37.97** | **0.0311** | Ties the VP ODE's 1000 calls; DDIM at 100: 35.84 |

- **Protocol.** EMA weights at epoch 200 of each table's config; 12,800 samples against all 50,000 training images. FID at this size is relative: it ranks runs in this repo, and doesn't compare with published numbers. For scale, two disjoint parts of the training set score 2.76, the test split 3.15, and pure noise 451 ([calibration](docs/evaluation.md#calibration)).
- **Noise.** Over sampling seeds, FID's std is 0.33 and KID's 0.0003, so a FID gap under about 0.7 is not a result.
- **Within a table, only the sampler differs.** Across tables the network and recipe are the same, but the forward process and the target differ, and so does the schedule: VP's $\beta$ is linear, the baseline's cosine. The hyperparameters were tuned on the discrete model only.
- **Cost** is model calls: predictor–corrector makes two per step. Sampling time is proportional to them, plus a fixed ~2 minutes for Inception features; DDPM's 1000 calls took 70–85 minutes.
- **Data:** the baseline's five results files and the runs' [`compare_ddim.json`](results/compare_ddim.json), [`compare_vpsde.json`](results/compare_vpsde.json) and [`compare_flow.json`](results/compare_flow.json) are in [`results/`](results/).

### Samples

![Samples from the baseline](docs/samples.png)

DDPM, 1000 steps, EMA weights at epoch 200.

## Method

Every network, UNet or DiT, maps noisy images $x_t$, the time $t$ and optional class labels $y$ to a tensor of $x_t$'s shape. Discrete diffusion ($x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \varepsilon$) and the score SDE ($x_t = a(t)\, x_0 + s(t)\, \varepsilon$) train it to predict the noise $\varepsilon$, and rectified flow ($x_t = (1 - t)\, \varepsilon + t\, x_0$) to predict the velocity $x_0 - \varepsilon$, each with a squared error. A sampler starts from pure noise and applies the network on its own grid of times. [Architecture](docs/architecture.md) has the derivations and the figures.

## Usage

Install from `requirements.txt`, or use Docker, with the repo mounted at `/source` and a data folder at `/data`:

```bash
docker build -t diffusion-models .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -v $PWD:/source -v <path_to_data_folder>:/data diffusion-models
```

Every entry point writes its artifacts under `--root`; the commands below use `/data`.

- **Tests:** `pytest`, on a CPU, in seconds.
- **Figures:** `make figures` redraws `docs/figures/` from the tests.

The commands behind the results tables, apart from the seed-0 rows, which come from one `ddpm.image.compare` run per table ([evaluation](docs/evaluation.md#the-readmes-runs)):

```bash
# Train the three checkpoints
python -m ddpm.image.train -c configs/image/ddpm.yaml --root /data
python -m ddpm.image.train -c configs/image/vpsde.yaml --root /data
python -m ddpm.image.train -c configs/image/flow.yaml --root /data

# Build the memorisation check's feature bank, once, before any eval
python -m scripts.precompute_features -c configs/image/ddpm.yaml --root /data

# The baseline row: seeds 0-4
python -m ddpm.image.eval -c configs/image/ddpm.yaml --root /data --seeds 0-4

# The FID scale
python -m scripts.calibrate_fid -c configs/image/ddpm.yaml --root /data
```

## What's next

- **Comparisons on this baseline:** UNet against DiT ([`configs/image/dit.yaml`](configs/image/dit.yaml), 765K parameters on the UNet's recipe, not yet scored), classifier-free guidance and its $\omega$, EMA against raw weights.
- **Later releases:** text diffusion.

## Papers

- Ho et al. 2020 (DDPM): training, ancestral sampling, the linear schedule.
- Song, Meng & Ermon 2021 (DDIM): the sampler.
- Song et al. 2021 (score SDE): the VP, sub-VP and VE SDEs, Euler–Maruyama, predictor–corrector and the probability-flow ODE.
- Liu et al. 2023 (rectified flow) and Lipman et al. 2023 (flow matching): the straight path and its velocity objective.
- Nichol & Dhariwal 2021: the cosine schedule.
- Ho & Salimans 2022: classifier-free guidance.
- Peebles & Xie 2023 (DiT): the transformer backbone and adaLN-Zero.
- Saharia et al. 2022: static thresholding.
- Heusel et al. 2017 (FID) and Bińkowski et al. 2018 (KID).

## License

MIT
