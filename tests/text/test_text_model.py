import torch

from ddpm.text.config import TextDiTModelConfig
from ddpm.text.model import TextDiT


def make_model(causal: bool, time_cond: bool) -> TextDiT:
    """A tiny TextDiT: V = 11 characters, L = 16 positions, width 32, 2 heads, depth 2."""
    return TextDiT(
        model_config=TextDiTModelConfig(embed_dim=32, heads=2, mlp_ratio=4.0, depth=2),
        V=11,
        L=16,
        causal=causal,
        time_cond=time_cond,
    )


def randomise_(model: TextDiT) -> None:
    """Every trainable weight ~ N(0, 0.1^2), in place. At init the output layer is zero, as in
    DiT, so every logit is 0 whatever the input, and a test comparing logits would pass without
    testing anything. The frozen Fourier projection W is left as it is."""
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                torch.nn.init.normal_(p, std=0.1)


def test_causal_no_future_leak() -> None:
    r"""
    The causal mask lets position i attend only to positions j \le i, so its logits are a
    function of the prefix alone:

      logits_i(x) = f_i(x_0, ..., x_i).

    Position i may see x_i itself, because its logits predict the next character, x_{i+1}.

    Changing one token, x_8, must leave logits_0, ..., logits_7 unchanged (up to float rounding,
    atol 1e-5). A missing or inverted mask lets them see x_8, and fails here.

    With random weights it must also change logits_8, ..., logits_15, the positions that see x_8.
    That second check keeps the first honest: a model that ignored its input altogether would
    pass the first one alone.
    """
    model = make_model(causal=True, time_cond=True)
    randomise_(model)
    model.eval()
    x1 = torch.randint(11, (3, 16), dtype=torch.long)
    x2 = torch.clone(x1)
    x2[:, 8] = (x1[:, 8] + 1) % 11  # a different character at position 8, in every row
    t = torch.rand((3,), dtype=torch.float)
    with torch.no_grad():
        logits1 = model(x1, t)
        logits2 = model(x2, t)

    assert torch.allclose(logits1[:, :8], logits2[:, :8], atol=1e-5)
    assert not torch.allclose(logits1[:, 8:], logits2[:, 8:], atol=1e-4)
