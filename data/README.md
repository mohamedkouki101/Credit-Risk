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

## How to get the data

1. Create a free Kaggle account, then *Settings → API → Create New Token*. Save the downloaded `kaggle.json` to `%USERPROFILE%\.kaggle\kaggle.json` on Windows or `~/.kaggle/kaggle.json` on Linux/macOS. Never put it in this repo.
2. Run `uv sync`, then `uv run python -m credit_risk data` (or `make data`). The download step (`credit_risk.data.download.download`) fetches `accepted_2007_to_2018Q4.csv.gz` into `data/raw/` and checks its byte size and SHA-256 against `configs/base.yaml`.
3. If a file in `data/raw/` fails the check, `DataIntegrityError` is raised and the file is left alone; delete it and run again. A new download is written to `*.part` and renamed only after it passes, so a bad download never sits under the real name.

By hand: `uv run kaggle datasets download wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/raw`. The next pipeline run still checks the hash.

Zenodo fallback: `download("zenodo")` streams `LC_loans_granting_model_dataset.csv` from record 11295916 and checks its md5. No account needed.

`credit_risk.data.load.load_raw()` reads only the columns listed in `configs/features.yaml` (keys, feature sources, benchmark, fairness, outcome), each with an explicit dtype. Post-origination columns are never loaded.

## Never in the repo

`kaggle.json`, `.env`, API keys, and any raw or derived loan-level file.
