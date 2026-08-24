# Input Data Contract

This directory contains documentation only because CRSP inputs are licensed and cannot be publicly redistributed.

## Authorized reconstruction

Run `notebooks/WRDS_reconstruction.ipynb` with an authorized WRDS account. It queries CRSP monthly total returns (`crsp.msf_v2.mthret`), the CRSP 30-day Treasury return (`crsp.mcti.t30ret`), and security-name history for the PERMNOs in `metadata/frozen_instrument_map.csv`.

Do not commit WRDS credentials, cookies, tokens, or the resulting security-level CRSP extracts.

## Public source endpoints

- Nasdaq historical API: `https://api.nasdaq.com/api/quote/{ticker}/historical`
- Kenneth French factor file: `https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip`
- FRED one-month Treasury series: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO`

`src/build_2026_live_extension.py` documents request parameters, transformations, month-end aggregation, and source-bridge audits.

## Expected local directories

After reconstruction, the project root should contain:

```text
publication_core_2024/      # licensed CRSP-derived inputs; never commit
gsci_live_extension_2026/   # reconstructed public-source extension
```

Both paths are excluded by `.gitignore`.
