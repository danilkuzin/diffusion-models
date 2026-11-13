import math

import torch

from ddpm.text.loss import mdlm_loss
from ddpm.text.schedule import mask

L = 256
V = 65
mask_id = V


def uniform_se(t: float, V: int, L: int, n: int) -> float:
    r"""se = \ln V \sqrt{(1 - t) / (t L n)}: the standard error of the mean of n per-character
    NELBOs under zero logits, derived in the test below."""
    return math.log(V) * math.sqrt((1 - t) / (t * L)) / math.sqrt(n)


def test_uniform_identity_mdlm() -> None:
    """
    A model that knows nothing (zero logits, so a uniform prediction)
    scores ln V per character on average, at every t.

    It checks the 1/t weight and the division by L. With zero logits every masked position costs
    exactly ln V. So the per-character loss is K ln V / (t L), where K is the number of masked
    positions, and E[K] = tL makes the mean ln V.

    (1-t)/t grows like 1/t at small t, therefore se is higher and one fixed atol can't work across t
    """

    gen = torch.Generator()
    gen.manual_seed(1)
    n_draws = 1000
    logits = torch.zeros((n_draws, L, V), dtype=torch.float)

    for t in [0.2, 0.5, 0.9]:
        x = torch.randint(V, (n_draws, L), dtype=torch.long)
        t_batch = torch.full((n_draws,), t)
        _, is_masked = mask(x, t_batch, mask_id, gen)  # masks with probability t
        # The only randomness in this experiment is K, the number of masked positions: a count of
        # successes in L independent trials with probability t, so
        #
        #   K \sim Binomial(L, t),    E[K] = L t,    Var[K] = L t (1 - t).

        nelbo, _, _ = mdlm_loss(logits, x, is_masked, t_batch)
        # The per-character NELBO of one sequence is
        #
        #   X = \frac{1}{t} \frac{K \ln V}{L} = c K,    c = \frac{\ln V}{t L},
        #
        # where 1/t is the weight, K \ln V is the summed masked cross-entropy, and 1/L makes it
        # per character.
        #
        # Mean, where the t cancels, which is the uniform identity:
        #
        #   E[X] = c E[K] = \frac{\ln V}{t L} L t = \ln V.
        #
        # Variance and standard deviation of one sequence:
        #
        #   Var[X] = c^2 Var[K] = \frac{(\ln V)^2}{t^2 L^2} L t (1 - t)
        #          = \frac{(\ln V)^2 (1 - t)}{t L},
        #
        #   \sigma = \ln V \sqrt{\frac{1 - t}{t L}}.
        #
        # The test averages X over n sequences, each with its own mask. The mean of n independent
        # copies has variance Var[X] / n, so its standard deviation shrinks by \sqrt{n}:
        #
        #   se = \frac{\sigma}{\sqrt{n}} = \ln V \sqrt{\frac{1 - t}{t L n}}.
        se = uniform_se(t, V, L, n_draws)
        # A correct loss lands outside 4 se with probability P(|Z| > 4) = 2 \Phi(-4) ~ 6e-5
        # (Z standard normal: the mean of n = 1000 copies is close to Gaussian).
        assert abs(nelbo.mean() - math.log(V)) < 4 * se
