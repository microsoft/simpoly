import pathlib

import pytest
import torch


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``@pytest.mark.gpu`` tests when no CUDA device is available."""
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="requires CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def test_data_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="function")
def test_checkpoint_pt() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent / "checkpoints" / "vivace_v0.1.mliap.pt"


@pytest.fixture(scope="function")
def test_checkpoint_ase_pt() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent / "checkpoints" / "vivace_v0.1.pt"
