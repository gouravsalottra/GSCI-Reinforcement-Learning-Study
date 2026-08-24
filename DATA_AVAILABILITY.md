# Data Availability and Licensing

## Deposited materials

This repository deposits the complete analysis code, frozen configurations, model archive, derived strategy-level results, portfolio weights, inference outputs, validation tests, metadata, and publication figures needed to audit the reported calculations.

## CRSP inputs: reconstructable but not redistributed

The security-level total returns and Treasury-bill returns used in the primary analysis were obtained through WRDS from the Center for Research in Security Prices (CRSP). They are licensed data and are therefore not redistributed in this public repository.

The exact reconstruction is provided in `notebooks/WRDS_reconstruction.ipynb`. It requests:

- CRSP CIZ monthly stock file: `crsp.msf_v2`, field `mthret`;
- CRSP monthly Treasury index: `crsp.mcti`, field `t30ret`;
- CRSP security-name history for identity verification;
- the exact permanent identifiers and dates listed below.

| Ticker | CRSP PERMNO | Role |
|---|---:|---|
| DBE | 91709 | Energy sleeve |
| GLD | 90448 | Gold sleeve |
| DBA | 91712 | Agriculture sleeve |
| DBB | 91715 | Base-metals sleeve |
| GSG | 91381 | Investable broad-commodity benchmark |
| DBC | 91129 | Investable broad-commodity benchmark |
| CPER | 13102 | Identity/robustness audit only |

Users must obtain WRDS/CRSP access through their institution and accept the provider’s terms. The code never embeds credentials. The notebook is output-stripped and prompts for the user’s own authenticated WRDS session.

## Public temporal-extension sources

The frozen-policy temporal extension uses:

- Nasdaq daily historical closing prices for DBE, GLD, DBA, DBB, GSG, and DBC;
- Kenneth R. French’s monthly factor file for the monthly risk-free return;
- FRED `DGS1MO` only for a separately labeled provisional June 2026 cash proxy, which is excluded from the article’s publication endpoint.

The public-source retrieval and transformation logic is in `src/build_2026_live_extension.py`. Source URLs are stated directly in the script and in `data/README.md`.

CRSP total returns are used through December 2025. Nasdaq closing-price returns are used for January–May 2026 after overlap auditing against CRSP during 2024–2025. The published endpoint is May 2026; the provisional June row is not part of any headline result.

## Deposited derived outputs

`outputs/` contains the published strategy-level returns, portfolio weights, performance summaries, bootstrap results, model-selection diagnostics, concentration measures, transaction-cost sensitivity, and leave-one-year-out results. `metadata/` contains the identity, timing, leakage, and source-bridge audits.

These derived research outputs are supplied for result verification. They are not a substitute for a CRSP license and must not be used to reconstruct or redistribute the underlying vendor database.

## Reproducibility scope

There are two supported verification modes:

1. **Public audit:** verifies all deposited result files, headline statistics, frozen models, figures, and hashes without licensed data.
2. **Full reconstruction:** reruns the entire pipeline after an authorized user generates the required CRSP inputs with the supplied WRDS notebook.

## Citation

When using the code or derived outputs, cite:

Salotra, G.; Pinsky, E. (2026). Do Reinforcement Learning Agents Improve Commodity Sector Rotation? Walk-Forward Evidence from Expert Selection, Strong Benchmarks, and a Frozen-Policy Temporal Extension. *Risks*, 14(9), 188. <https://doi.org/10.3390/risks14090188>.
