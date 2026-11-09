"""Step 0.3 (E-02): one seed makes random/numpy draws and LightGBM params reproducible."""

import random

import numpy as np

from credit_risk.seed import lightgbm_seed_params, set_all_seeds


def _draws() -> tuple[float, list[float]]:
    # The legacy global RNG is exactly what set_all_seeds is meant to seed.
    return random.random(), np.random.rand(3).tolist()  # noqa: NPY002


def test_set_all_seeds_makes_draws_reproducible() -> None:
    set_all_seeds(42)
    first = _draws()
    set_all_seeds(42)
    assert _draws() == first


def test_different_seeds_give_different_draws() -> None:
    set_all_seeds(42)
    first = _draws()
    set_all_seeds(43)
    assert _draws() != first


def test_lightgbm_seed_params_are_deterministic() -> None:
    # PLAN §4: LightGBM's deterministic mode needs a force_* option.
    assert lightgbm_seed_params(42) == {
        "seed": 42,
        "deterministic": True,
        "force_row_wise": True,
    }
