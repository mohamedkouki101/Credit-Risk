"""Cleaning rules of data plan 5.4 (D-04).

Rules are applied in this order, and every dropped or blanked row is counted:

1. (5.4.4) Drop rows with an empty ``loan_amnt``: the footer artefacts of LC CSVs. Done
   first so a footer row is not also counted as an unparsable date.
2. (5.4.1) Parse ``issue_d`` and ``earliest_cr_line`` with format ``%b-%Y`` (first of the
   month). Rows whose ``issue_d`` is missing or unparsable are dropped; rows whose
   ``earliest_cr_line`` is present but unparsable are dropped. A missing
   ``earliest_cr_line`` stays missing (credit age is then unknown, not wrong).
3. (5.4.2) Strip ``%`` from ``revol_util`` and ``int_rate`` (no-op when already numeric, as
   in Kaggle v3); strip whitespace from ``term``.
4. (5.4.3) Drop duplicate ``id`` (keep the first); assert ids are unique afterwards.
5. (5.4.5, data plan 5.2) ``dti`` outside [0, 100] -> missing plus ``dti_flag``;
   ``revol_util`` above 150 -> missing. Other range rules (``loan_amnt`` > 0,
   ``annual_inc`` >= 0, ``fico_mid`` in [300, 850]) are enforced by the pandera schema,
   so an impossible value fails loudly instead of being silently repaired.

Nothing is winsorised or capped here (rule 5.4.6): any capping is fitted on train only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DATE_FORMAT = "%b-%Y"
DTI_RANGE = (0.0, 100.0)
REVOL_UTIL_MAX = 150.0

DROP_RULES = (
    "footer_empty_loan_amnt",
    "issue_d_unparsed",
    "earliest_cr_line_unparsed",
    "duplicate_id",
)
FLAG_RULES = ("dti_out_of_range", "revol_util_over_150")


@dataclass(frozen=True)
class CleaningLog:
    """Rows dropped per rule (``drops``) and values blanked per rule (``flags``)."""

    raw_rows: int
    drops: dict[str, int] = field(default_factory=dict)
    flags: dict[str, int] = field(default_factory=dict)

    @property
    def total_dropped(self) -> int:
        return sum(self.drops.values())

    @property
    def clean_rows(self) -> int:
        return self.raw_rows - self.total_dropped

    def to_frame(self) -> pd.DataFrame:
        """One row per rule: ``rule``, ``kind`` (``drop``/``set_missing``), ``rows``."""
        rows = [(r, "drop", n) for r, n in self.drops.items()]
        rows += [(r, "set_missing", n) for r, n in self.flags.items()]
        return pd.DataFrame(rows, columns=["rule", "kind", "rows"])


def _to_number(s: pd.Series) -> pd.Series:
    """Percent strings like ``"29.7%"`` -> 29.7; numeric columns are returned as float64."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype("float64")
    text = s.astype("string").str.strip().str.rstrip("%")
    return pd.to_numeric(text, errors="raise").astype("float64")


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Apply the cleaning rules to a copy of ``raw`` (output of ``load_raw``).

    Returns the cleaned frame (fresh 0..n-1 index) and the log of drops and blanked values.
    ``len(out) == log.raw_rows - log.total_dropped`` always holds.
    """
    df = raw.copy()
    drops: dict[str, int] = {}
    flags: dict[str, int] = {}

    footer = df["loan_amnt"].isna()
    drops["footer_empty_loan_amnt"] = int(footer.sum())
    df = df[~footer]

    issue = pd.to_datetime(df["issue_d"], format=DATE_FORMAT, errors="coerce")
    bad_issue = issue.isna()
    drops["issue_d_unparsed"] = int(bad_issue.sum())
    earliest = pd.to_datetime(df["earliest_cr_line"], format=DATE_FORMAT, errors="coerce")
    bad_earliest = earliest.isna() & df["earliest_cr_line"].notna() & ~bad_issue
    drops["earliest_cr_line_unparsed"] = int(bad_earliest.sum())
    keep = ~(bad_issue | bad_earliest)
    df = df[keep].assign(issue_d=issue[keep], earliest_cr_line=earliest[keep])

    for col in ("revol_util", "int_rate"):
        if col in df:
            df[col] = _to_number(df[col])
    df["term"] = df["term"].str.strip()

    dup = df["id"].duplicated(keep="first")
    drops["duplicate_id"] = int(dup.sum())
    df = df[~dup]
    assert df["id"].is_unique, "duplicate ids remain after rule 5.4.3"

    dti_bad = df["dti"].notna() & ~df["dti"].between(*DTI_RANGE)
    flags["dti_out_of_range"] = int(dti_bad.sum())
    df["dti_flag"] = dti_bad.astype(bool)
    df.loc[dti_bad, "dti"] = float("nan")

    util_bad = df["revol_util"] > REVOL_UTIL_MAX
    flags["revol_util_over_150"] = int(util_bad.sum())
    df.loc[util_bad, "revol_util"] = float("nan")

    log = CleaningLog(raw_rows=len(raw), drops=drops, flags=flags)
    out = df.reset_index(drop=True)
    assert len(out) == log.clean_rows
    return out, log
