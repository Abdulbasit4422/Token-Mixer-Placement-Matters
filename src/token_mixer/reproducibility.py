from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed process-wide random sources and return a seeded loader generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    return torch.Generator().manual_seed(seed)


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy in one DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
