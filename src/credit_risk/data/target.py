"""Target construction and the per-vintage data profile (D-03, data plan 5.3a).

``build_target`` labels every row without dropping any:

- ``y`` (nullable ``Int8``): 1 = bad (charged off / default), 0 = good (fully paid). It is
  set **only** for rows in the modelling population with a determinate status, so loans
  outside it (other terms, joint applications, the unlabelled 2016-2018 monitoring
  vintages) can never be used as labelled data by mistake.
- ``segment``: ``"core"`` (Individual, 36-month, issued inside the population window) or
  ``"out_of_population"``.
- ``exclusion_reason``: ``None`` for modelling rows, else the first failing rule in the
  order ``application_type``, ``term``, ``issue_d_window``, ``indeterminate``.
- ``policy_flag``: the loan "does not meet the credit policy" (kept; SA-04 drops them).

Input contract: ``issue_d`` is already parsed to datetime and ``term`` is stripped
(cleaning rules 5.4.1-5.4.2), and footer rows are gone (rule 5.4.4).
An unknown or null ``loan_status`` raises (T-06): a new status must never silently
become "good".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_risk.config import TargetConfig

PROFILE_COLUMNS = [
    "quarter",
    "rows",
    "n_bad",
    "n_good",
    "n_indeterminate",
    "bad_rate",
    "indeterminate_share",
    "flagged",
    "missing_share",
]


def build_target(df: pd.DataFrame, cfg: TargetConfig) -> pd.DataFrame:
    """Return a copy of ``df`` with ``y``, ``segment``, ``exclusion_reason``, ``policy_flag``.

    Raises ``ValueError`` on a null or unmapped ``loan_status`` and ``TypeError`` if
    ``issue_d`` is not a datetime column.
    """
    if not pd.api.types.is_datetime64_any_dtype(df["issue_d"]):
        raise TypeError("issue_d must be parsed to datetime before build_target (rule 5.4.1)")
    status = df["loan_status"]
    if (n_null := int(status.isna().sum())) > 0:
        raise ValueError(
            f"{n_null} row(s) have null loan_status: drop footer rows first (rule 5.4.4)"
        )
    mapping = cfg.status_to_class()
    unknown = sorted(set(status.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"loan_status value(s) not in configs/target.yaml: {unknown}")

    cls = status.map(mapping)
    pop = cfg.population
    month = df["issue_d"].dt.strftime("%Y-%m")
    bad_app = (df["application_type"] != pop.application_type).to_numpy(dtype=bool)
    bad_term = (df["term"] != pop.term).to_numpy(dtype=bool)
    outside = ((month < pop.issue_d_start) | (month > pop.issue_d_end)).to_numpy(dtype=bool)
    indeterminate = (cls == "indeterminate").to_numpy(dtype=bool)

    # Assign from lowest to highest precedence so the first failing rule wins.
    reason = np.full(len(df), None, dtype=object)
    for mask, label in [
        (indeterminate, "indeterminate"),
        (outside, "issue_d_window"),
        (bad_term, "term"),
        (bad_app, "application_type"),
    ]:
        reason[mask] = label
    in_population = ~(bad_app | bad_term | outside)
    modelled = in_population & ~indeterminate

    out = df.copy()
    y = pd.Series(pd.NA, index=df.index, dtype="Int8")
    y[modelled & (cls == "bad").to_numpy(dtype=bool)] = 1
    y[modelled & (cls == "good").to_numpy(dtype=bool)] = 0
    out["y"] = y
    out["segment"] = np.where(in_population, "core", "out_of_population")
    out["exclusion_reason"] = pd.Series(reason, index=df.index, dtype=object)
    out["policy_flag"] = status.str.startswith(cfg.policy_flag_prefix).astype(bool)
    return out


def vintage_profile(
    targeted: pd.DataFrame, feature_columns: Sequence[str], flag_share: float
) -> pd.DataFrame:
    """Per issue quarter of the core population: counts, bad rate, indeterminate share.

    ``bad_rate`` = bad / (bad + good), indeterminate loans excluded (a share, not a %).
    ``missing_share`` = share of missing cells over ``feature_columns`` in that quarter.
    ``flagged`` is True when ``indeterminate_share > flag_share`` (data plan 5.3a: SA-01).
    """
    core = targeted[targeted["segment"] == "core"]
    quarter = core["issue_d"].dt.to_period("Q").astype(str)
    is_bad = core["y"].eq(1).fillna(False).astype(int)
    is_good = core["y"].eq(0).fillna(False).astype(int)
    is_indet = core["exclusion_reason"].eq("indeterminate").astype(int)
    n_missing = core[list(feature_columns)].isna().sum(axis=1)

    grouped = pd.DataFrame(
        {
            "quarter": quarter,
            "rows": 1,
            "n_bad": is_bad,
            "n_good": is_good,
            "n_indeterminate": is_indet,
            "n_missing": n_missing,
        }
    ).groupby("quarter", sort=True)
    prof = grouped.sum().reset_index()
    prof["bad_rate"] = prof["n_bad"] / (prof["n_bad"] + prof["n_good"])
    prof["indeterminate_share"] = prof["n_indeterminate"] / prof["rows"]
    prof["flagged"] = prof["indeterminate_share"] > flag_share
    prof["missing_share"] = prof["n_missing"] / (prof["rows"] * len(feature_columns))
    return prof[PROFILE_COLUMNS]


def render_profile_md(profile: pd.DataFrame, flag_share: float) -> str:
    """Markdown for ``reports/data_profile.md``; every number comes from ``profile``."""
    lines = [
        "# Data profile",
        "",
        "Generated by `uv run python -m credit_risk data` from `results/vintage_profile.csv`.",
        "Do not edit by hand.",
        "",
        "Core population (data plan 5.3a), by issue quarter. Bad rate = bad / (bad + good);",
        "indeterminate loans are excluded from it. Missing = share of missing feature cells.",
        "",
        "| Quarter | Rows | Bad | Good | Indeterminate | Bad rate | Indet. | Missing | Flag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in profile.itertuples(index=False):
        flag = "indeterminate > threshold" if r.flagged else ""
        lines.append(
            f"| {r.quarter} | {r.rows:,} | {r.n_bad:,} | {r.n_good:,} | "
            f"{r.n_indeterminate:,} | {r.bad_rate:.2%} | {r.indeterminate_share:.2%} | "
            f"{r.missing_share:.2%} | {flag} |"
        )
    flagged = profile.loc[profile["flagged"], "quarter"].tolist()
    lines += [
        "",
        f"Flagged vintages (indeterminate share > {flag_share:.1%}): "
        + (", ".join(flagged) if flagged else "none"),
        "",
    ]
    return "\n".join(lines)
