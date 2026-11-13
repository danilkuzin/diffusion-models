# Text diffusion

A masked diffusion language model (MDLM, Sahoo et al. 2024) over the characters of Tiny Shakespeare, and an autoregressive (AR) baseline on the same backbone. This page covers:
  * data
  * model
  * forward process and its loss
  * sampler
  * training
  * evaluation

## Data

Tiny Shakespeare, 1M characters. Vocabulary: 65 distinct characters, by descending frequency. 90% training / 10% validation.

Three reference losses:

| Model | Loss per character |
|---|---|
| Uniform over the 65 characters, $\ln 65$ | 4.174 |
| Unigram: the character frequencies, $H(X)$ | 3.309 |
| Bigram: the next character given the previous one, $H(X_i \mid X_{i-1})$ | 2.453 |

## The model contract

The model:

$$x_\theta(z_t, t) \;\to\; x,$$

where $z_t \in (B, L)$ is a partly masked sequence,
$x \in (B, L, V)$ are logits for the clean characters. [MASK] is an extra token id. $t \in (B,)$, $t_i \in [t_\text{min}, 1]$.

The AR baseline is the same network with a causal attention mask and no time input; it is a reference for MDLM's bound, with an exact likelihood.

## Model

DiT (Peebles & Xie 2023), [`configs/text/mdlm.yaml`](../configs/text/mdlm.yaml):

| | |
|---|---|
| Parameters | 16.6M |
| Context | 256 characters |
| Width, depth, heads | 384, 6, 6 |
| Attention | Bidirectional for MDLM, causal for AR |
| Conditioning | adaLN on Fourier features of $t$ (Tancik et al. 2020) |

The size follows nanoGPT's character-level Shakespeare model. The one deviation from DiT: the gates start at 0.5 rather than 0 ([below](#why-the-gates-start-open)).

## Forward process and loss

Each position is masked independently, with $\alpha_t = 1 - t$:

$$q(z_t^\ell \mid x^\ell) = \alpha_t\, \mathbb 1[z_t^\ell = x^\ell] + (1 - \alpha_t)\, \mathbb 1[z_t^\ell = \text{mask}].$$

Reverse step from $t$ to $s < t$: an unmasked position is copied; a masked one is unmasked with probability

$$\frac{\alpha_s - \alpha_t}{1 - \alpha_t} = \frac{t - s}{t},$$

to a draw from $x_\theta$, and otherwise stays masked.

Training minimises the continuous-time NELBO under the SUBS parameterisation (Sahoo et al. 2024), whose weight is $-\alpha_t' / (1 - \alpha_t) = 1/t$:

$$\mathcal L(x) = \frac1L\, \mathbb E_{t \sim U(0, 1]}\, \mathbb E_{z_t \sim q(\cdot \mid x)} \left[\frac1t \sum_{\ell:\ z_t^\ell = \text{mask}} -\log x_\theta^\ell(x^\ell)\right] \;\ge\; -\frac1L \log p_\theta(x).$$

The AR baseline minimises the exact next-character NLL:

$$\mathcal L_\text{AR}(x) = -\frac{1}{L - 1} \sum_{i=1}^{L-1} \log p_\theta(x_i \mid x_{<i}).$$

## Sampler

Ancestral sampling (Sahoo et al. 2024) starts from a fully masked sequence and applies the reverse step $N$ times, on the grid $t = 1, \frac{N-1}{N}, \dots, 0$. Positions unmasked in the same step are drawn independently, so sample quality falls as $N$ decreases: compare the [256-step](../results/samples_mdlm_256_steps.txt) and [16-step](../results/samples_mdlm_16_steps.txt) samples.

The AR baseline has no sampler.

## Training

AdamW with a cosine learning-rate schedule, no dropout, 30,000 steps: about 4 h on a T4. The checkpoint with the lowest validation loss is kept as `best.pt`.

## Evaluation

`ddpm.text.eval` computes the NELBO per character on the whole validation set, averaged over several draws of the masking noise, ± its standard error. The noise is fixed, so every checkpoint is scored on the same draws. MDLM's value is an upper bound on its NLL; the AR baseline's is exact.

## The release run

[`configs/text/mdlm.yaml`](../configs/text/mdlm.yaml). The validation loss:

| Step | 500 | 1000 | 2000 | 5000 | 10,000 | 20,000 | 30,000 |
|---|---|---|---|---|---|---|---|
| Val loss | 3.1675 | 2.4946 | 2.1959 | 1.9497 | 1.7510 | 1.6145 | **1.5718 ± 0.0196** |

The loss is still falling at the last step, which is `best.pt`.

## Why the gates start open

With adaLN-Zero (Peebles & Xie 2023), every gate starts at 0, and training remained at the unigram entropy: the loss of a model that predicts the character frequencies at every masked position,

$$\frac1L \cdot \frac1t \cdot tL \cdot H(X) = H(X) = 3.309.$$

Starting the gates at 0.5 (`gate_init`) removes the plateau. Validation loss after 2000 steps, on a smaller model and an earlier, noisier estimate:

| Gates at init | Seed 0 | Seed 1 |
|---|---|---|
| 0, adaLN-Zero | 3.275 | 3.305 |
| 0.5 | **3.159** | **3.105** |

## Tests

The tests check the $1/t$ weight, through a model with uniform predictions, which must score $\ln V$ per character at every $t$, and the AR model's causal mask, which must hide every later character.

## Reproduce

```bash
python -m ddpm.text.train -c configs/text/mdlm.yaml --root /data
python -m ddpm.text.eval -c configs/text/mdlm.yaml --root /data
python -m ddpm.text.generate -c configs/text/mdlm.yaml --root /data                 # 256 steps
python -m ddpm.text.generate -c configs/text/mdlm.yaml --root /data --num-steps 16

# The AR baseline: training and evaluation only
python -m ddpm.text.train -c configs/text/ar.yaml --root /data
python -m ddpm.text.eval -c configs/text/ar.yaml --root /data
```

## Limitations

- One run, one seed.
- The AR baseline is not trained in this release; nanoGPT's 1.47 is the only AR reference.
- Ancestral sampling only; sample quality is not measured.
- No dropout: the training loss ends about 0.25 below the validation loss.
