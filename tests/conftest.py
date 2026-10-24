import pytest
import torch

from adaptive_fsvod.smoke import make_synthetic


@pytest.fixture(scope="session", autouse=True)
def limit_cpu_threads():
    torch.set_num_threads(1)


@pytest.fixture
def corpus(tmp_path):
    return make_synthetic(tmp_path / "synthetic")
