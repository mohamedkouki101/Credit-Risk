"""Environment smoke test (step 0.2, roadmap E-01).

Checks that the pinned library set works together before any real code is written:

1. OptBinning ``BinningProcess`` + ``Scorecard`` (PDO scaling, 600 points at 50:1
   good:bad odds, PDO 20) on ``make_classification`` data in a pandas DataFrame.
2. LightGBM with ``monotone_constraints``.
3. SHAP ``TreeExplainer`` on that LightGBM model.

Each step is timed and prints ``[timing] <step> <seconds>s``. The script exits
non-zero if any step raises or fails its sanity check. Run with:

    uv run python scripts/smoke_env.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import version
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from optbinning import BinningProcess, Scorecard
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from credit_risk.seed import set_all_seeds

SEED = 42
N_ROWS = 20_000
N_FEATURES = 10
PACKAGES = (
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "optbinning",
    "lightgbm",
    "shap",
    "pyarrow",
    "pandera",
    "pydantic",
    "fairlearn",
    "duckdb",
)

TIMINGS: dict[str, float] = {}


@contextmanager
def timed(step: str) -> Iterator[None]:
    """Time a block and print ``[timing] <step> <seconds>s``."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    TIMINGS[step] = round(elapsed, 3)
    print(f"[timing] {step} {elapsed:.3f}s", flush=True)


def make_data() -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic binary data. ``y = 1`` means bad (default), as in the project."""
    X, y = make_classification(
        n_samples=N_ROWS,
        n_features=N_FEATURES,
        n_informative=5,
        n_redundant=2,
        weights=[0.8, 0.2],
        random_state=SEED,
    )
    cols = [f"x{i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="bad")


def check_scorecard(X: pd.DataFrame, y: pd.Series) -> None:
    """Fit a PDO scorecard; higher score must mean lower risk."""
    binning = BinningProcess(variable_names=list(X.columns))
    card = Scorecard(
        binning_process=binning,
        estimator=LogisticRegression(max_iter=1000),
        scaling_method="pdo_odds",
        scaling_method_params={"pdo": 20, "odds": 50, "scorecard_points": 600},
    )
    card.fit(X, y)
    scores = card.score(X)
    proba_bad = card.predict_proba(X)[:, 1]
    assert np.isfinite(scores).all(), "non-finite scores"
    auc = roc_auc_score(y, proba_bad)
    assert auc > 0.6, f"scorecard AUC too low: {auc:.3f}"
    # Score is in "good" direction: it should rank bads low, so AUC(y, -score) > 0.5.
    assert roc_auc_score(y, -scores) > 0.6, "score direction is not good-high"
    print(f"  scorecard AUC={auc:.3f} score range=[{scores.min():.0f}, {scores.max():.0f}]")


def check_lightgbm(X: pd.DataFrame, y: pd.Series) -> lgb.LGBMClassifier:
    """Fit LightGBM with x0 forced increasing in bad-risk; verify monotonicity."""
    constraints = [1] + [0] * (N_FEATURES - 1)
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        monotone_constraints=constraints,
        random_state=SEED,
        verbose=-1,
    )
    model.fit(X, y)
    # Sweep x0 with other features fixed at a real row: predictions must not decrease.
    probe = pd.concat([X.iloc[[0]]] * 50, ignore_index=True)
    probe["x0"] = np.linspace(X["x0"].min(), X["x0"].max(), 50)
    sweep = model.predict_proba(probe)[:, 1]
    assert (np.diff(sweep) >= -1e-12).all(), "monotone constraint violated"
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    print(f"  lightgbm AUC={auc:.3f}")
    return model


def check_shap(model: lgb.LGBMClassifier, X: pd.DataFrame) -> None:
    """SHAP values plus base value must reproduce the raw log-odds output."""
    sample = X.iloc[:2000]
    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(sample))
    if values.ndim == 3:  # some versions return one array per class
        values = values[..., 1] if values.shape[-1] == 2 else values[1]
    base = np.ravel(explainer.expected_value)[-1]
    raw = model.predict(sample, raw_score=True)
    max_err = float(np.max(np.abs(values.sum(axis=1) + base - raw)))
    assert max_err < 1e-4, f"SHAP additivity error {max_err:.2e}"
    print(f"  shap additivity max error={max_err:.2e}")


def peak_rss_mb() -> float | None:
    """Peak resident memory in MB, or None where ``resource`` is unavailable (Windows)."""
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024, 1)


def run_step(name: str, fn: Callable[[], Any]) -> Any:
    with timed(name):
        return fn()


def main() -> int:
    set_all_seeds(SEED)
    X, y = make_data()
    run_step("optbinning_scorecard", lambda: check_scorecard(X, y))
    model = run_step("lightgbm_monotone", lambda: check_lightgbm(X, y))
    run_step("shap_tree_explainer", lambda: check_shap(model, X))

    summary = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {pkg: version(pkg) for pkg in PACKAGES},
        "timings_s": TIMINGS,
        "peak_rss_mb": peak_rss_mb(),
        "n_rows": N_ROWS,
    }
    print(json.dumps(summary, indent=2))
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
