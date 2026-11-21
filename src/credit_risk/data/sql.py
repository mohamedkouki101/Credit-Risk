"""Run the SQL files in ``sql/`` with DuckDB over in-memory pandas frames (D-04).

The SQL is the readable version of a table; the pandas code is the one the pipeline uses.
Tests assert they agree, so the SQL cannot rot.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from credit_risk.config import REPO_ROOT

SQL_DIR = REPO_ROOT / "sql"


def run_sql(name: str, **tables: pd.DataFrame) -> pd.DataFrame:
    """Run ``sql/<name>.sql`` with each keyword frame registered as a table of that name."""
    path = SQL_DIR / f"{name}.sql"
    if not path.is_file():
        raise FileNotFoundError(path)
    with duckdb.connect() as con:
        for table, frame in tables.items():
            con.register(table, frame)
        return con.execute(path.read_text(encoding="utf-8")).df()
