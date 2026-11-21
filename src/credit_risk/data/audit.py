"""Feature leakage and missingness audit (D-05), on the **train window only**.

For every whitelisted feature:

- missing rate among good (``y = 0``) and bad (``y = 1``) loans, and the L6 flag
  (data plan 5.6): the larger rate is at least 1% and more than 3x the smaller one. The 3x
  ratio is the plan's heuristic (Unverified); the 1% floor stops a handful of missing
  cells from triggering it.
- information value (IV) on train, and IV per issue year inside train (stability).
- a mechanical decision, first rule that fires:
  1. constant on train -> drop;
  2. L6 flag -> drop (missingness encodes the target);
  3. IV above ``iv_leakage_flag`` (0.5) -> drop until a leakage audit clears it;
  4. otherwise keep (IV below ``iv_min`` is noted: the binning step may drop it).

IV uses a coarse, audit-only binning: numeric and ordinal features in up to 10 quantile
bins (each distinct value is its own bin when there are at most 10), categories as they
are, and missing as its own bin. The scorecard's own binning (M-03) is separate.
``IV = sum_i (g_i - b_i) * ln(g_i / b_i)`` with ``g_i``, ``b_i`` the bin's share of all
goods and bads; an empty cell counts as 0.5 loans so IV stays finite.
Validation, test and monitoring rows are never read here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_risk.config import FeatureSpec, MonthRange

MISSING_BIN = "__missing__"
L6_RATIO = 3.0
L6_FLOOR = 0.01
AUDIT_BINS = 10


def _bins(x: pd.Series, kind: str, n_bins: int) -> pd.Series:
    labels = pd.Series(MISSING_BIN, index=x.index, dtype=object)
    present = x.notna()
    if kind == "categorical":
        labels[present] = x[present].astype(str)
    elif x[present].nunique() <= n_bins:
        labels[present] = x[present].astype(float).astype(str)
    else:
        labels[present] = pd.qcut(x[present].astype(float), n_bins, duplicates="drop").astype(str)
    return labels


def information_value(x: pd.Series, y: pd.Series, kind: str, n_bins: int = AUDIT_BINS) -> float:
    """IV of ``x`` for target ``y`` (1 = bad). See the module docstring for the binning."""
    yv = pd.Series(np.asarray(y, dtype="float64"), index=x.index)
    keep = yv.notna()
    labels = _bins(x[keep], kind, n_bins)
    counts = pd.crosstab(labels, yv[keep].astype(int)).reindex(columns=[0, 1], fill_value=0)
    good = counts[0].to_numpy(dtype="float64")
    bad = counts[1].to_numpy(dtype="float64")
    total_good, total_bad = good.sum(), bad.sum()
    if total_good == 0 or total_bad == 0:
        return 0.0
    good = np.where(good == 0, 0.5, good) / total_good
    bad = np.where(bad == 0, 0.5, bad) / total_bad
    return float(np.sum((good - bad) * np.log(good / bad)))


def missing_by_target(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """(missing rate among goods, missing rate among bads)."""
    yv = np.asarray(y, dtype="float64")
    miss = x.isna().to_numpy()
    return float(miss[yv == 0].mean()), float(miss[yv == 1].mean())


def l6_flag(rate_good: float, rate_bad: float) -> bool:
    """True if missingness differs more than 3x between goods and bads (and is >= 1%)."""
    hi, lo = max(rate_good, rate_bad), min(rate_good, rate_bad)
    if hi < L6_FLOOR:
        return False
    return lo == 0 or hi / lo > L6_RATIO


def _decide(
    n_unique: int, flag: bool, iv: float, iv_min: float, iv_leakage_flag: float
) -> tuple[str, str]:
    if n_unique <= 1:
        return "drop", "constant in the core train population"
    if flag:
        return "drop", "L6: missing rate differs more than 3x between good and bad"
    if iv > iv_leakage_flag:
        return "drop", f"IV {iv:.3f} above the leakage flag {iv_leakage_flag}: audit first"
    note = " (weak: below iv_min, binning may drop it)" if iv < iv_min else ""
    return "keep", f"IV {iv:.3f}{note}"


def feature_audit(
    table: pd.DataFrame,
    specs: Sequence[FeatureSpec],
    train: MonthRange,
    iv_min: float,
    iv_leakage_flag: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit every feature on labelled core rows issued inside ``train``.

    Returns ``(audit, iv_by_year)``: one row per feature, and one row per (feature, year).
    """
    month = table["issue_d"].dt.strftime("%Y-%m")
    rows = (
        (table["segment"] == "core")
        & table["y"].notna()
        & (month >= train.start)
        & (month <= train.end)
    )
    tr = table[rows]
    y = tr["y"].astype("float64")
    year = tr["issue_d"].dt.year

    audit_rows, year_rows = [], []
    for spec in specs:
        x = tr[spec.name]
        rate_good, rate_bad = missing_by_target(x, y)
        flag = l6_flag(rate_good, rate_bad)
        iv = information_value(x, y, spec.kind)
        ivs = {}
        for yr in sorted(year.unique()):
            in_year = (year == yr).to_numpy()
            ivs[int(yr)] = information_value(x[in_year], y[in_year], spec.kind)
            year_rows.append({"feature": spec.name, "year": int(yr), "iv": ivs[int(yr)]})
        decision, reason = _decide(int(x.nunique()), flag, iv, iv_min, iv_leakage_flag)
        audit_rows.append(
            {
                "feature": spec.name,
                "kind": spec.kind,
                "n_train": len(x),
                "missing_good": rate_good,
                "missing_bad": rate_bad,
                "l6_flag": flag,
                "iv_train": iv,
                "iv_year_min": min(ivs.values()),
                "iv_year_max": max(ivs.values()),
                "decision": decision,
                "reason": reason,
            }
        )
    return pd.DataFrame(audit_rows), pd.DataFrame(year_rows, columns=["feature", "year", "iv"])


def render_audit_md(audit: pd.DataFrame, iv_by_year: pd.DataFrame) -> str:
    """Markdown for ``research/feature_audit.md``; every number comes from the frames."""
    lines = [
        "# Feature audit (D-05)",
        "",
        "Generated by `uv run python -m credit_risk data` from `results/feature_audit.csv`",
        "and `results/feature_iv_by_year.csv`. Do not edit by hand.",
        "",
        "Train window only (labelled core loans). Missing rates are shares among good and",
        "bad loans. L6 = missing rate differs more than 3x between them (and is at least 1%).",
        "IV uses a coarse audit binning (up to 10 quantile bins, missing as its own bin).",
        "",
        "| Feature | Kind | Missing (good) | Missing (bad) | L6 | IV train | IV by year"
        " (min to max) | Decision | Reason |",
        "|---|---|---:|---:|---|---:|---|---|---|",
    ]
    for r in audit.itertuples(index=False):
        lines.append(
            f"| {r.feature} | {r.kind} | {r.missing_good:.2%} | {r.missing_bad:.2%} | "
            f"{'yes' if r.l6_flag else 'no'} | {r.iv_train:.3f} | "
            f"{r.iv_year_min:.3f} to {r.iv_year_max:.3f} | **{r.decision}** | {r.reason} |"
        )
    wide = iv_by_year.pivot(index="feature", columns="year", values="iv")
    wide = wide.reindex(audit["feature"])
    years = [str(c) for c in wide.columns]
    lines += [
        "",
        "## IV by issue year (train)",
        "",
        "| Feature | " + " | ".join(years) + " |",
        "|---|" + "---:|" * len(years),
    ]
    for feature, vals in wide.iterrows():
        lines.append(f"| {feature} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    lines.append("")
    return "\n".join(lines)
