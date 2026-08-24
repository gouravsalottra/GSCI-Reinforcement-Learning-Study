# Replication Package: Commodity Sector Rotation with Reinforcement Learning

This repository contains the replication materials for:

> Gourav Salotra and Eugene Pinsky (2026), “Do Reinforcement Learning Agents Improve Commodity Sector Rotation? Walk-Forward Evidence from Expert Selection, Strong Benchmarks, and a Frozen-Policy Temporal Extension,” *Risks*, 14(9), 188. <https://doi.org/10.3390/risks14090188>

## What is deposited

- Complete Python source for the transparent strategies, supervised expert selector, PPO experiment, temporal extension, inference, and publication tables.
- A stripped, executable WRDS notebook containing the CRSP queries and deterministic construction steps.
- The frozen design configuration, field dictionary, permanent-security-identifier map, alignment tests, and leakage tests.
- The ten fixed PPO model files and their training/runtime metadata in `models/ppo_fixed_results.zip`.
- Derived strategy-level results, weights, statistical-inference tables, concentration diagnostics, and audit outputs.
- All 18 publication figures in PDF and PNG format.
- Environment pins, integrity checks, and a repository-wide SHA-256 manifest.

The repository deliberately does **not** redistribute licensed CRSP security-level returns. Researchers with WRDS/CRSP access can reconstruct those inputs using `notebooks/WRDS_reconstruction.ipynb`. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the exact division between deposited and restricted material.

## Study design at a glance

| Item | Specification |
|---|---|
| Rotation assets | DBE, GLD, DBA, DBB |
| Investable benchmarks | GSG, DBC |
| Training targets | January 2008–December 2018 (132 months) |
| Original out-of-sample targets | January 2019–December 2024 (72 months) |
| Frozen-policy temporal extension | January 2025–May 2026 (17 months) |
| Unified out-of-sample evaluation | January 2019–May 2026 (89 months) |
| Primary cost | 10 basis points per unit of drift-adjusted one-half-L1 turnover |
| Primary rule benchmark | Top-two six-month momentum with inverse-volatility allocation |
| PPO evaluation | Ten prespecified seeds, ensemble weights, no post-extension retraining |

## Main findings reproduced by the deposited outputs

For January 2019–May 2026, six-month momentum achieved an 18.56% CAGR and 1.112 excess Sharpe, while the ten-seed PPO ensemble achieved a 9.56% CAGR and 0.438 excess Sharpe. The 60-month selector achieved a 23.4% CAGR, but its advantage over momentum was not statistically established (two-sided circular block-bootstrap \(p=0.323\)). The PPO temporal-extension gain is strongly concentrated in March 2026; the paper reports the deletion diagnostic rather than treating the episode as evidence of general dominance.

## Directory map

| Path | Contents |
|---|---|
| `src/` | Analysis and model code |
| `notebooks/` | Output-stripped WRDS reconstruction notebook |
| `config/` | Frozen confirmatory and publication configurations |
| `data/` | Input contract and licensed-data reconstruction instructions |
| `metadata/` | Instrument, timing, leakage, alignment, and bridge audits |
| `models/` | Ten frozen PPO models plus model metadata |
| `outputs/` | Published derived results and diagnostics |
| `figures/` | Publication figures in vector and raster formats |
| `scripts/` | Lightweight repository validation utilities |
| `tests/` | Output-contract tests that do not require CRSP access |
| `paper/` | Links and citation metadata for the published article |

## Quick verification without licensed data

Python 3.12 is suitable for the public audit. The archived PPO training run used Python 3.11.12 and the exact versions recorded in `models/ppo_fixed_results.zip`; use `requirements-ppo-frozen.txt` when reproducing PPO inference.

```bash
python scripts/verify_repository.py
python -m unittest discover -s tests -v
```

These commands verify the deposited output contract, headline values, model archive, expected figure set, file naming, and SHA-256 checksums. They do not require CRSP.

## Full reconstruction with WRDS/CRSP access

1. Create an environment and install `requirements.txt`.
2. Run `notebooks/WRDS_reconstruction.ipynb` using your own WRDS credentials.
3. Place the generated files in `publication_core_2024/` as documented in `data/README.md`.
4. Run `python src/build_2026_live_extension.py` to reconstruct the public-data bridge.
5. Run `python src/evaluate_2026_live_holdout.py` and then `python src/run_full_publication_analysis.py`.
6. Compare the regenerated results with `outputs/` and the manifest.

Detailed commands and expected products are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Terminology note

The article uses **frozen-policy temporal extension** for January 2025–May 2026. A few legacy filenames retain `post_freeze` because they are immutable identifiers from the frozen computational run; those names do not change the statistical status of the segment.

## License and citation

Code is released under the MIT License. Third-party data remain subject to their providers’ terms. Cite the article using [CITATION.cff](CITATION.cff) or the BibTeX entry in `paper/README.md`.

## Contact

For questions about reconstruction, open a GitHub issue in this repository and identify the file and validation step involved.
