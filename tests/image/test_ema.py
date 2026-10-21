import torch

from ddpm.image.ema import EMA


def test_ema_update_matches_closed_form() -> None:
    """Weights held at p: shadow_n = p + decay**n * (shadow_0 - p)."""
    model = torch.nn.Linear(3, 2)
    decay, num_updates = 0.9, 10
    ema = EMA(model, decay)
    shadow_0 = {name: shadow.clone() for name, shadow in ema.shadow_weights.items()}
    p = {name: torch.randn_like(shadow) for name, shadow in shadow_0.items()}
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(p[name])

    for _ in range(num_updates):
        ema.update(model)

    for name, shadow in ema.shadow_weights.items():
        expected = p[name] + decay**num_updates * (shadow_0[name] - p[name])
        torch.testing.assert_close(shadow, expected)
