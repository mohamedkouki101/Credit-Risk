"""T-05 synthetic data with a known answer (PLAN §9).

200k rows: x1, x2 ~ N(0, 1), x3 ~ Bernoulli(0.3), noise ~ N(0, 1) (unused by the truth).
True logit of bad = -2 + 0.8 x1 - 0.5 x2 + 0.7 x3; y ~ Bernoulli(sigmoid(logit)).
Rows are split in half: ``dev`` to fit, ``oot`` to score (no time structure here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

N_ROWS = 200_000
TRUE_INTERCEPT = -2.0
TRUE_COEF = {"x1": 0.8, "x2": -0.5, "x3": 0.7, "noise": 0.0}


@dataclass(frozen=True)
class Synthetic:
    dev: pd.DataFrame
    oot: pd.DataFrame
    features: tuple[str, ...] = ("x1", "x2", "x3", "noise")


def make_synthetic(n: int = N_ROWS, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.binomial(1, 0.3, size=n).astype(float),
            "noise": rng.normal(size=n),
        }
    )
    logit = TRUE_INTERCEPT + sum(c * df[k] for k, c in TRUE_COEF.items())
    df["true_p"] = 1 / (1 + np.exp(-logit))
    df["y"] = rng.binomial(1, df["true_p"])
    return df


@pytest.fixture(scope="session")
def synthetic() -> Synthetic:
    df = make_synthetic()
    half = len(df) // 2
    return Synthetic(
        dev=df.iloc[:half].reset_index(drop=True), oot=df.iloc[half:].reset_index(drop=True)
    )
