from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reproduced_outputs"
CORE = ROOT / "publication_core_2024"
LIVE = ROOT / "gsci_live_extension_2026"
OUT = ROOT / "reproduced_tables" / "supplement"
OUT.mkdir(parents=True, exist_ok=True)


def latex_table(
    frame: pd.DataFrame,
    filename: str,
    caption: str,
    label: str,
    float_format: str = "%.4f",
) -> None:
    text = frame.to_latex(
        index=False,
        longtable=True,
        escape=True,
        caption=caption,
        label=label,
        float_format=float_format,
    )
    (OUT / filename).write_text(
        "\\begingroup\\tiny\\setlength{\\tabcolsep}{2pt}\n"
        + text
        + "\n\\endgroup\n"
    )


instrument = pd.read_csv(CORE / "frozen_instrument_map.csv")
instrument["company_names"] = instrument["company_names"].str.slice(0, 38)
instrument = instrument[
    [
        "ticker",
        "permno",
        "company_names",
        "first_name_date",
        "last_name_date",
        "share_codes",
        "identity_status",
    ]
].rename(
    columns={
        "company_names": "company",
        "first_name_date": "first_name",
        "last_name_date": "last_name",
        "share_codes": "share_code",
        "identity_status": "status",
    }
)
latex_table(
    instrument,
    "s1_instrument_map.tex",
    "Frozen permanent-identifier map and name-history validation.",
    "tab:s1-instruments",
)

coverage = pd.read_csv(CORE / "legacy_ciz_coverage_comparison.csv")
coverage = coverage[
    [
        "source",
        "ticker",
        "permno",
        "first_valid_return",
        "last_valid_return",
        "valid_returns",
        "missing_returns",
        "duplicate_months",
    ]
].rename(
    columns={
        "first_valid_return": "first",
        "last_valid_return": "last",
        "valid_returns": "valid",
        "missing_returns": "missing",
        "duplicate_months": "duplicates",
    }
)
latex_table(
    coverage,
    "s2_coverage.tex",
    "Legacy CRSP and CIZ monthly-return coverage.",
    "tab:s2-coverage",
)

quality = pd.read_csv(CORE / "corrected_return_quality_summary.csv")
quality = quality.rename(
    columns={
        "first_valid_month": "first",
        "last_valid_month": "last",
        "valid_months": "months",
        "internal_missing_months": "gaps",
        "zero_returns": "zeros",
        "minimum_return": "min_return",
        "maximum_return": "max_return",
        "returns_le_minus_100pct": "le_minus_100",
        "absolute_returns_gt_50pct": "abs_gt_50",
    }
)
latex_table(
    quality,
    "s3_return_quality.tex",
    "Return-quality diagnostics for the selected instruments.",
    "tab:s3-quality",
)

features = pd.read_csv(CORE / "feature_and_timing_dictionary.csv")
features = features[["column", "definition", "information_available"]].rename(
    columns={
        "column": "feature",
        "information_available": "available",
    }
)
features["feature"] = features["feature"].str.replace("state__", "", regex=False)
features["definition"] = (
    features["definition"]
    .str.replace("Compounded return over months ", "Compound ", regex=False)
    .str.replace("Monthly standard deviation over months ", "Volatility ", regex=False)
)
latex_table(
    features,
    "s4_features.tex",
    "Feature definitions and timing.",
    "tab:s4-features",
)

leakage = pd.read_csv(CORE / "leakage_test_results.csv")
leakage = leakage[["test", "status", "detail"]]
leakage["detail"] = leakage["detail"].astype(str).str.slice(0, 55)
latex_table(
    leakage,
    "s5_leakage.tex",
    "Feature, target, scaling, and leakage tests.",
    "tab:s5-leakage",
)

performance = pd.read_csv(ANALYSIS / "performance_all_periods.csv")
latex_table(
    performance[
        [
            "period",
            "strategy",
            "months",
            "cagr",
            "annual_volatility",
            "excess_sharpe",
            "maximum_drawdown",
            "average_monthly_turnover",
        ]
    ],
    "s6_all_performance.tex",
    "Performance across all reported periods.",
    "tab:s6-performance",
)

costs = pd.read_csv(ANALYSIS / "transaction_cost_sensitivity.csv")
latex_table(
    costs[
        [
            "strategy",
            "cost_bps",
            "cagr",
            "excess_sharpe",
            "maximum_drawdown",
            "average_monthly_turnover",
        ]
    ],
    "s7_costs.tex",
    "Transaction-cost sensitivity for active strategies.",
    "tab:s7-costs",
)

pairwise = pd.read_csv(ANALYSIS / "pairwise_block_bootstrap.csv")
latex_table(
    pairwise,
    "s8_pairwise.tex",
    "All pairwise circular block-bootstrap results against 6-month momentum.",
    "tab:s8-pairwise",
)

selector_pairwise = pd.read_csv(
    ANALYSIS / "selector_pairwise_block_bootstrap.csv"
)
latex_table(
    selector_pairwise,
    "s9_selector_pairwise.tex",
    "All supervised-selector pairwise block-bootstrap results.",
    "tab:s9-selector-pairwise",
)

rolling = pd.read_csv(ANALYSIS / "selector_rolling_window_inference.csv")
latex_table(
    rolling,
    "s10_rolling.tex",
    "Rolling-selector inference across window and block lengths.",
    "tab:s10-rolling",
)

reality = pd.read_csv(ANALYSIS / "centered_max_mean_reality_check.csv")
reality = reality.drop(columns=["candidate_set"])
latex_table(
    reality,
    "s11_reality.tex",
    "Centered max-mean reality check.",
    "tab:s11-reality",
)

dsr = pd.read_csv(ANALYSIS / "deflated_sharpe_frozen_set.csv")
latex_table(
    dsr.drop(columns=["scope_warning"]),
    "s12_dsr.tex",
    "Deflated Sharpe diagnostic for the explicitly frozen strategy set.",
    "tab:s12-dsr",
)

concentration = pd.read_csv(ANALYSIS / "portfolio_concentration.csv")
concentration = concentration.rename(
    columns={
        "training_window_months": "window",
        "average_hhi": "avg_hhi",
        "effective_assets": "eff_assets",
        "average_max_weight": "avg_max_w",
        "months_max_weight_gt_50pct": "months_max_gt50",
        "average_weight_DBE": "w_DBE",
        "average_weight_GLD": "w_GLD",
        "average_weight_DBA": "w_DBA",
        "average_weight_DBB": "w_DBB",
        "average_weight_CASH": "w_cash",
    }
)
latex_table(
    concentration,
    "s13_concentration.tex",
    "Concentration and average portfolio weights.",
    "tab:s13-concentration",
)

leave = pd.read_csv(ANALYSIS / "leave_one_year_out.csv")
latex_table(
    leave,
    "s14_leave_year.tex",
    "Leave-one-calendar-year-out comparisons with 6-month momentum.",
    "tab:s14-leave-year",
)

annual = pd.read_csv(ANALYSIS / "annual_excess_contributions.csv")
latex_table(
    annual,
    "s15_annual.tex",
    "Annual return contributions relative to 6-month momentum.",
    "tab:s15-annual",
)

selector_frequency = pd.read_csv(ANALYSIS / "selector_expert_frequency.csv")
latex_table(
    selector_frequency,
    "s16_selector_frequency.tex",
    "Expert-selection frequency.",
    "tab:s16-selector-frequency",
)

ppo_contrib = pd.read_csv(ANALYSIS / "ppo_monthly_asset_contributions.csv")
ppo_contrib = ppo_contrib.loc[
    ppo_contrib["target_month"].between("2025-01-31", "2026-05-31")
]
latex_table(
    ppo_contrib,
    "s17_ppo_holdout.tex",
    "PPO asset contributions during the frozen-policy temporal extension.",
    "tab:s17-ppo-holdout",
)

bridge = pd.read_csv(LIVE / "nasdaq_vs_crsp_monthly_audit.csv")
bridge = bridge.rename(
    columns={
        "overlap_months": "months",
        "mean_absolute_difference": "mean_abs_diff",
        "maximum_absolute_difference": "max_abs_diff",
        "non_december_months": "non_dec_months",
        "non_december_mean_absolute_difference": "non_dec_mean_abs",
        "non_december_maximum_absolute_difference": "non_dec_max_abs",
    }
)
latex_table(
    bridge,
    "s18_bridge.tex",
    "Nasdaq-to-CRSP source-bridge overlap audit.",
    "tab:s18-bridge",
)

hashes = pd.read_csv(ANALYSIS / "sha256_manifest.csv")
latex_table(
    hashes,
    "s19_hashes.tex",
    "SHA-256 manifest for the final analysis outputs.",
    "tab:s19-hashes",
)

print(f"Wrote supplement tables to {OUT}")
