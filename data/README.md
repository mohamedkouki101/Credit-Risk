# Data

Nothing in `raw/`, `interim/` or `processed/` is committed (see `.gitignore`). Every file is rebuilt from a hash-checked download.

| Folder | Contents | Written by |
|---|---|---|
| `raw/` | Original Lending Club download, unchanged | `uv run python -m credit_risk data` (D-02) |
| `interim/` | Target-built, cleaned tables | `data` stage (D-03, D-04) |
| `processed/` | `train/valid/test/monitor.parquet` out-of-time splits | `features` stage (D-06) |

## Source and licence

Decided in step 0.4 (D-01); full record in `research/data_decision.md`.

- **Primary:** Kaggle `wordsforthewise/lending-club` (Version 3), file `accepted_2007_to_2018Q4.csv.gz`, licence **CC0** (screenshots in `research/licence_evidence/`). Original source: Lending Club.
  SHA-256 `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d`, 392,582,231 bytes.
- **Fallback / benchmark:** Zenodo "Lending Club loan dataset for granting models" v0.1, CC BY 4.0, md5 `b019384d6bc65bf2a3e839362e4ff502`.

Download needs a Kaggle API token in `%USERPROFILE%\.kaggle\kaggle.json` (never in this repo).

## Never in the repo

`kaggle.json`, `.env`, API keys, and any raw or derived loan-level file.
