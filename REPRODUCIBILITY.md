# Reproducibility Guide

## 1. Environment

The table/figure analysis is supported on Python 3.12. Create an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The frozen PPO run used Python 3.11.12. Its exact package versions are recorded inside `models/ppo_fixed_results.zip` as `runtime_versions.json` and mirrored in `requirements-ppo-frozen.txt`. Use a separate Python 3.11 environment for exact PPO inference:

```bash
python3.11 -m venv .venv-ppo
source .venv-ppo/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ppo-frozen.txt
```

## 2. Verify the public deposit

```bash
python scripts/verify_repository.py
python -m unittest discover -s tests -v
```

Expected result: every check prints `PASS`, and the test suite exits with status 0.

## 3. Reconstruct the licensed inputs

Open `notebooks/WRDS_reconstruction.ipynb`. Authenticate to WRDS using your own account and execute the notebook from top to bottom. The notebook creates the deterministic core under `publication_core_2024/`.

Required products for the downstream scripts are:

- `model_ready_primary_with_crsp_cash_2008_2024.parquet`
- `baseline_monthly_returns_2008_2024.parquet`
- `verified_monthly_total_return_panel_NO_FILL.parquet`
- `selector_expanding_predictions.parquet`
- `selector_expanding_weights.parquet`
- `selector_and_benchmark_oos_returns.parquet`

The notebook also creates the security-identity, target-alignment, leakage, return-quality, and feature-timing audits.

## 4. Construct the public-source bridge

```bash
python src/build_2026_live_extension.py
```

This creates `gsci_live_extension_2026/`. The article’s endpoint is May 2026. The script may retain a June 2026 provisional row, but downstream publication filters exclude it.

## 5. Re-evaluate the frozen PPO ensemble

```bash
python src/evaluate_2026_live_holdout.py
```

The ten seed models in `models/ppo_fixed_results.zip` are loaded in deterministic inference mode. There is no retraining, hyperparameter search, seed selection, or scaler refit on the temporal extension.

Acceptance gate: the 2019–2024 reconstructed PPO net-return path must match the frozen archived output to floating-point tolerance before the temporal extension is accepted.

## 6. Run the unified analysis

```bash
python src/run_full_publication_analysis.py
```

This rebuilds:

- transparent strategy weights and net returns;
- expanding and rolling selector results;
- ten-seed PPO ensemble results;
- performance and concentration summaries;
- circular block-bootstrap inference at 3-, 6-, and 12-month blocks;
- Holm-adjusted pairwise tests;
- the centered maximum-mean reality check;
- deflated-Sharpe diagnostics;
- transaction-cost and leave-one-year-out sensitivity;
- the March 2026 deletion diagnostic.

## 7. Generate LaTeX tables

```bash
python src/generate_publication_tables.py
python src/generate_supplement_tables.py
```

## 8. Expected headline values

All values below come from `outputs/performance_all_periods.csv` unless otherwise noted.

| Strategy, January 2019–May 2026 | CAGR | Excess Sharpe | Maximum drawdown |
|---|---:|---:|---:|
| Six-month momentum | 18.5643% | 1.11172 | −12.2683% |
| Expanding selector | 16.0076% | 0.75455 | −17.4691% |
| PPO ten-seed ensemble | 9.5584% | 0.43779 | −29.8914% |

The rolling 60-month selector has a 23.4% CAGR and 1.033 excess Sharpe. Its six-month-block comparison with six-month momentum has a two-sided centered bootstrap p-value of 0.323. The PPO ensemble’s March 2026 net return is 28.9155%, with a DBE weight of 68.2838%.

## 9. Integrity

`SHA256SUMS.txt` covers every deposited file except itself. Run:

```bash
sha256sum --check SHA256SUMS.txt
```

The verifier also checks the nested PPO ZIP for corruption and required seed files.
