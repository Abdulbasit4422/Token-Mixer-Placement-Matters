import random

import numpy as np
import torch

from token_mixer.reproducibility import seed_everything, seed_worker


def test_seed_everything_repeats_all_generators():
    seed_everything(42)
    first = (random.random(), np.random.rand(), torch.rand(1).item())

    seed_everything(42)
    second = (random.random(), np.random.rand(), torch.rand(1).item())

    assert first == second
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_seed_everything_returns_seeded_generator():
    generator = seed_everything(7)

    expected = torch.Generator().manual_seed(7)
    assert torch.equal(torch.rand(4, generator=generator), torch.rand(4, generator=expected))


def test_seed_worker_repeats_python_and_numpy_sequences():
    torch.manual_seed(123)
    seed_worker(0)
    first = (random.random(), np.random.rand())

    torch.manual_seed(123)
    seed_worker(0)
    second = (random.random(), np.random.rand())

    assert first == second
