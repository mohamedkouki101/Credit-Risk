# Environment spike (step 0.2, roadmap E-01)

Date: 2026-11-08. Script: `scripts/smoke_env.py` (test: `tests/unit/test_smoke_env.py`, marked `slow`).
Numbers below are copied from the script's JSON output on each machine; they are environment timings on 20,000 synthetic rows, not model results.

## Verdict

- `uv run python scripts/smoke_env.py` exits 0 on **Windows 11** (Kouki's machine, native) and on **WSL2 Ubuntu 26.04**.
- **Risk R3 is cleared:** OptBinning 1.0.0 `BinningProcess` + `Scorecard` work with **pandas 3.1.1**. No `pandas<3` pin is needed.
- The three checks also assert behaviour, not just imports: scorecard scores are finite and rank bads low (PDO 20, 600 points at 50:1), the LightGBM monotone constraint holds on a 50-point sweep, and SHAP values + base value reproduce LightGBM's raw log-odds (max error 2.2e-14).

## Locked versions (`uv.lock`)

| Package | Version | Package | Version |
|---|---|---|---|
| Python | 3.12.14 | optbinning | 1.0.0 |
| pandas | 3.1.1 | lightgbm | 4.7.0 |
| numpy | 2.5.4 | shap | 0.54.1 |
| scipy | 1.18.1 | pyarrow | 26.0.1 |
| scikit-learn | 1.9.1 | pandera | 0.35.0 |
| pydantic | 2.14.3 | fairlearn | 0.15.0 |
| duckdb | 2.0.0 | cvxpy (via optbinning → ropwr) | 1.9.3 |

## Timings (seconds)

| Step | Windows 11 | WSL2 |
|---|---|---|
| OptBinning BinningProcess + Scorecard | 2.041 | 1.191 |
| LightGBM, monotone constraints | 0.301 | 0.189 |
| SHAP TreeExplainer (2,000 rows) | 0.335 | 0.192 |
| Peak RSS | not measured (`resource` is Unix-only) | 318 MB |

## Incompatibilities and notes

1. **LightGBM on Linux needs the system OpenMP library `libgomp.so.1`.** The WSL install lacked it (`OSError: libgomp.so.1: cannot open shared object file`). Windows wheels bundle OpenMP, and GitHub's `ubuntu-latest` runners ship libgomp, so neither Kouki's Windows setup nor CI is affected. Fix on WSL: `sudo apt-get install -y libgomp1`. For this spike the Ubuntu `libgomp1` package was unpacked into `~/.local/opt/libgomp` and loaded with `LD_LIBRARY_PATH`; no system change was made.
2. **CVXPY can't import the HiGHS solver** (`highspy` ABI mismatch: `undefined symbol _ZN5Highs13releaseMemoryEv`), seen on WSL. CVXPY arrives through `ropwr`, which OptBinning uses only for piecewise binning. Our binning uses OptBinning's default solver `cp` (OR-Tools 9.15), so this is harmless. It shows up as a noisy log line at import. If piecewise binning is ever needed, revisit.
3. **SHAP 0.54 + LightGBM binary:** TreeExplainer emits `UserWarning: ... shap values output has changed to a list of ndarray`. The explain module (X-01) must handle the per-class output shape explicitly, as `check_shap()` in the smoke script does.
4. **Separate virtualenvs for Windows and WSL.** The project `.venv/` belongs to Windows. WSL work sets `UV_PROJECT_ENVIRONMENT=~/.venvs/credit-risk` so the two don't overwrite each other. `uv.lock` is cross-platform and shared.
