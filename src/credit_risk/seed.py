"""Seeding. Call ``set_all_seeds(cfg.global_seed)`` at every entry point.

LightGBM has no global seed: pass ``**lightgbm_seed_params(seed)`` into each model.
Bootstrap streams should use their own generator, ``np.random.default_rng(seed + k)``.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    """Seed Python's ``random``, NumPy's legacy global RNG and ``PYTHONHASHSEED``.

    ``PYTHONHASHSEED`` only affects subprocesses started after this call.
    """
    random.seed(seed)
    # Deliberately the legacy global RNG: libraries called without random_state draw from it.
    np.random.seed(seed)  # noqa: NPY002
    os.environ["PYTHONHASHSEED"] = str(seed)


def lightgbm_seed_params(seed: int) -> dict[str, int | bool]:
    """LightGBM params for reproducible fits (``deterministic`` needs a ``force_*`` option)."""
    return {"seed": seed, "deterministic": True, "force_row_wise": True}
