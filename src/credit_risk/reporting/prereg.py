"""Generated parts of ``PREREGISTRATION.md`` (P-01).

The split table (labelled loans and bads per split) comes from
``results/vintage_profile.csv`` and the config hashes from ``configs/``, so no number in
the pre-registration is typed by hand. Only counts are pre-registered, never outcome
rates or model metrics. Hashes normalise CRLF to LF so a Windows checkout gives the same
value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from credit_risk.config import MonthRange

BLOCK_START = "<!-- generated:prereg:start -->"
BLOCK_END = "<!-- generated:prereg:end -->"


def _quarter_months(quarter: str) -> tuple[str, str]:
    """``"2014Q3"`` -> ``("2014-07", "2014-09")``: first and last month of the quarter."""
    year, q = quarter.split("Q")
    first = 3 * (int(q) - 1) + 1
    return f"{year}-{first:02d}", f"{year}-{first + 2:02d}"


def split_counts(profile: pd.DataFrame, windows: Mapping[str, MonthRange]) -> pd.DataFrame:
    """Sum the per-quarter profile into one row per split.

    A quarter belongs to a split if it overlaps the split's window. The profile holds only
    population rows, so a quarter cut by the population start (2007Q2: June only) counts
    in full. Split windows themselves are whole quarters, so no quarter overlaps two.
    """
    months = profile["quarter"].map(_quarter_months)
    first = months.str[0]
    last = months.str[1]
    rows = []
    for name, win in windows.items():
        q = profile[(last >= win.start) & (first <= win.end)]
        rows.append(
            {
                "split": name,
                "window": f"{win.start} to {win.end}",
                "first_quarter": q["quarter"].min(),
                "last_quarter": q["quarter"].max(),
                "n_labelled": int(q["n_bad"].sum() + q["n_good"].sum()),
                "n_bad": int(q["n_bad"].sum()),
                "n_indeterminate": int(q["n_indeterminate"].sum()),
            }
        )
    return pd.DataFrame(rows)


def config_hashes(config_dir: Path) -> dict[str, str]:
    """SHA-256 of every ``*.yaml`` in ``config_dir`` with line endings normalised to LF."""
    out = {}
    for path in sorted(config_dir.glob("*.yaml")):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        out[path.name] = hashlib.sha256(data).hexdigest()
    return out


def render_block(counts: pd.DataFrame, hashes: Mapping[str, str]) -> str:
    """Markdown for the generated block: split counts and config hashes."""
    lines = [
        "| Split | Issue months | Labelled loans | Bad loans | Indeterminate (excluded) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in counts.itertuples(index=False):
        lines.append(
            f"| {r.split} | {r.window} | {r.n_labelled:,} | {r.n_bad:,} | {r.n_indeterminate:,} |"
        )
    lines += [
        "",
        "Config files at registration (SHA-256 of the file with LF line endings):",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    lines += [f"| `configs/{name}` | `{digest}` |" for name, digest in hashes.items()]
    return "\n".join(lines)


def fill_block(document: str, block: str) -> str:
    """Replace the text between the generated-block markers with ``block``."""
    if BLOCK_START not in document or BLOCK_END not in document:
        raise ValueError("generated-block marker not found")
    head, rest = document.split(BLOCK_START, 1)
    _, tail = rest.split(BLOCK_END, 1)
    return f"{head}{BLOCK_START}\n{block}\n{BLOCK_END}{tail}"
