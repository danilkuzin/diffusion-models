from pathlib import Path

import pytest

ARTIFACTS = Path(__file__).parent / "artifacts"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--viz", action="store_true")


@pytest.fixture
def viz(request: pytest.FixtureRequest) -> Path | None:
    """tests/artifacts/<test name>.png under --viz, else None."""
    if not request.config.getoption("--viz"):
        return None
    ARTIFACTS.mkdir(exist_ok=True)
    return ARTIFACTS / f"{request.node.name}.png"
