"""Fill the generated block of PREREGISTRATION.md (P-01).

Run after ``uv run python -m credit_risk data``:

    uv run python scripts/fill_prereg.py

Reads ``results/vintage_profile.csv`` and ``configs/``. Do not run it after the
``prereg-v1`` tag: the registered numbers are frozen (amend instead).
"""

from __future__ import annotations

import pandas as pd

from credit_risk.config import CONFIG_DIR, REPO_ROOT, load_all_configs, load_base_config
from credit_risk.reporting.prereg import config_hashes, fill_block, render_block, split_counts


def main() -> None:
    splits = load_all_configs().splits
    profile = pd.read_csv(REPO_ROOT / load_base_config().paths.results / "vintage_profile.csv")
    windows = {"train": splits.train, "valid": splits.valid, "test (OOT)": splits.test}
    block = render_block(split_counts(profile, windows), config_hashes(CONFIG_DIR))
    path = REPO_ROOT / "PREREGISTRATION.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(fill_block(text, block), encoding="utf-8", newline="\n")
    print(f"filled {path}")


if __name__ == "__main__":
    main()
