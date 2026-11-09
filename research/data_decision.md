# Data decision (gate D-01, step 0.4)

Date: 2026-11-09.

## Decision

**PRIMARY = full Lending Club file** (Kaggle `wordsforthewise/lending-club`, `accepted_2007_to_2018Q4.csv.gz`).
**Fallback / benchmark = Zenodo** "Lending Club loan dataset for granting models" v0.1 (CC BY 4.0, md5 `b019384d6bc65bf2a3e839362e4ff502`), used for the M-02 literature reproduction.

Rule applied (data plan §5.1): the licence field shows CC0, which allows research use, and the download works. So the full file is primary and the Zenodo variant rules (§5.3b) are not needed.

## Licence evidence

| Evidence | Where |
|---|---|
| Kaggle dataset page, License field = **CC0: Public Domain** | `research/licence_evidence/2026-11-09_kaggle_licence.png` |
| Kaggle file list: Version 3 (648.05 MB), 2 files, `accepted_2007_to_2018Q4.csv.gz` 392.58 MB | `research/licence_evidence/2026-11-09_kaggle_files.png` |
| Kaggle API at download time: `License(s): CC0-1.0` | CLI output, 2026-11-09 |

This verifies the CC0 tag that the data plan (§5.1) had only from a secondary catalogue (confidence Medium → High).

**Caveat (state it if asked):** CC0 is the licence chosen by the Kaggle uploader, who republished data Lending Club had made public. It is not a licence issued by Lending Club itself. The project uses the data for non-commercial research and education, never commits raw data, and credits Lending Club as the original source.

## Downloaded file

| Field | Value |
|---|---|
| Dataset | `wordsforthewise/lending-club`, Version 3 |
| File | `data/raw/accepted_2007_to_2018Q4.csv.gz` (git-ignored) |
| Command | `uv run --with six kaggle datasets download wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/raw` |
| Size | 392,582,231 bytes (matches the Kaggle API listing exactly) |
| Kaggle file creation date | 2019-12-17 |
| **SHA-256** | `55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d` |
| Integrity | Full gzip decompression succeeded (CRC verified) |
| Shape | 151 columns; 2,260,701 lines after the header. Lines, not loans: the exact loan count is measured in D-02/D-03 |
| Key columns present | `loan_status`, `issue_d` |

Hash computed with Python `hashlib` (SHA-256, 1 MiB chunks). D-02 (`download.py`) must verify this hash and raise `DataIntegrityError` on mismatch.

The rejected-loans file (`rejected_2007_to_2018Q4.csv.gz`, 255,470,782 bytes) is not downloaded. It is stretch-only.

## Tooling note

`kaggle` 2.2.5 (dev dependency) imports `six` without declaring it, so `uv run kaggle ...` failed with `ModuleNotFoundError: No module named 'six'`. The original download used `uv run --with six` as a stopgap. **Fixed 2026-11-09:** `six` (1.17.0) is now a dev dependency (Kouki chose this option), and plain `uv run kaggle datasets download wordsforthewise/lending-club -f accepted_2007_to_2018Q4.csv.gz -p data/raw` works. Remove `six` if a later `kaggle` release declares it.
