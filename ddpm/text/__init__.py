from typing import Literal, assert_never

from ddpm.text.sample import AncestralSampler, ConfidenceSampler

SamplerName = Literal["ancestral", "confidence", "confidence_greedy"]


def create_sampler(name: SamplerName) -> AncestralSampler | ConfidenceSampler:
    """The sampler named by generate's --sampler."""
    match name:
        case "ancestral":
            return AncestralSampler()
        case "confidence_greedy":
            return ConfidenceSampler(greedy=True)
        case "confidence":
            return ConfidenceSampler(greedy=False)
        case _:
            assert_never(name)
