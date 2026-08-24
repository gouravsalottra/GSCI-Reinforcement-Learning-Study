from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reproduced_outputs"
OUT = ROOT / "reproduced_tables" / "main"
OUT.mkdir(parents=True, exist_ok=True)


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def num(value: float) -> str:
    return f"{value:.3f}"


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content)


performance = pd.read_csv(ANALYSIS / "performance_all_periods.csv")
combined = performance.loc[
    performance["period"].eq("Combined OOS 2019-May 2026")
].set_index("strategy")
ordered = [
    ("absolute_momentum_12m_cash", "Absolute momentum/cash"),
    ("momentum_6m_top2_invvol", "6-month momentum"),
    ("supervised_selector", "Expanding selector"),
    ("momentum_12m_top2_invvol", "12-month momentum"),
    ("inverse_vol_12m", "Inverse volatility"),
    ("equal_weight", "Equal weight"),
    ("DBC_buy_and_hold", "DBC"),
    ("GSG_buy_and_hold", "GSG"),
    ("ppo_10_seed_ensemble", "PPO ensemble"),
]
rows = []
for strategy, label in ordered:
    row = combined.loc[strategy]
    rows.append(
        f"{label} & {pct(row.cagr)} & {num(row.excess_sharpe)} & "
        f"{pct(row.annual_volatility)} & {pct(row.maximum_drawdown)} & "
        f"{pct(row.total_return)} & {num(row.annualized_turnover)} \\\\"
    )
write(
    "table_main_performance.tex",
    r"""\begin{table}[H]
\caption{Unified out-of-sample performance, January 2019--May 2026.}
\label{tab:main-performance}
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrr}
\toprule
Strategy & CAGR & Excess Sharpe & Volatility & Max. DD & Total return & Annual turnover\\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\begin{minipage}{\textwidth}\footnotesize
\emph{Notes:} Returns are net of 10 basis points per unit of one-half drift-adjusted turnover for active strategies. Sharpe ratios use contemporaneous monthly Treasury-bill returns. Passive ETF benchmarks have zero modeled rebalancing turnover.
\end{minipage}
\end{table}
""",
)

period_labels = {
    "Original OOS 2019-2024": "Original OOS",
    "Temporal extension 2025-May 2026": "Temporal extension",
    "Combined OOS 2019-May 2026": "Combined",
    "Excluding March 2026": "Excl. March 2026",
}
period_strategies = [
    ("momentum_6m_top2_invvol", "6-month momentum"),
    ("supervised_selector", "Expanding selector"),
    ("ppo_10_seed_ensemble", "PPO ensemble"),
]
rows = []
for period, short in period_labels.items():
    subset = performance.loc[performance["period"].eq(period)].set_index("strategy")
    for strategy, label in period_strategies:
        row = subset.loc[strategy]
        rows.append(
            f"{short} & {label} & {int(row.months)} & {pct(row.cagr)} & "
            f"{num(row.excess_sharpe)} & {pct(row.maximum_drawdown)} \\\\"
        )
write(
    "table_period_robustness.tex",
    r"""\begin{table}[H]
\caption{Temporal robustness and the March 2026 concentration diagnostic.}
\label{tab:period-robustness}
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrrrr}
\toprule
Period & Strategy & Months & CAGR & Excess Sharpe & Max. DD\\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
""",
)

rolling = pd.read_csv(ANALYSIS / "selector_rolling_window_sensitivity.csv")
inference = pd.read_csv(ANALYSIS / "selector_rolling_window_inference.csv")
inference = inference.loc[inference["block_length"].eq(6)].set_index(
    "training_window_months"
)
rows = []
for _, row in rolling.iterrows():
    inf = inference.loc[int(row.training_window_months)]
    rows.append(
        f"{int(row.training_window_months)} & {pct(row.cagr)} & "
        f"{num(row.excess_sharpe)} & {pct(row.maximum_drawdown)} & "
        f"{pct(inf.annualized_mean_difference)} & "
        f"[{pct(inf.ci_2_5)}, {pct(inf.ci_97_5)}] & "
        f"{inf.two_sided_centered_p_value:.3f} \\\\"
    )
write(
    "table_rolling_selector.tex",
    r"""\begin{table}[H]
\caption{Rolling-window selector sensitivity and comparison with 6-month momentum.}
\label{tab:rolling-selector}
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{rrrrrrr}
\toprule
Window & CAGR & Excess Sharpe & Max. DD & Mean diff. & 95\% CI & $p$\\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\begin{minipage}{\textwidth}\footnotesize
\emph{Notes:} Window is the number of trailing training months. Mean differences, confidence intervals, and two-sided centered circular block-bootstrap $p$-values compare each selector with the prespecified 6-month momentum benchmark using six-month blocks and 5,000 replications.
\end{minipage}
\end{table}
""",
)

selector_inference = pd.read_csv(
    ANALYSIS / "selector_pairwise_block_bootstrap.csv"
)
selector_inference = selector_inference.loc[
    selector_inference["block_length"].eq(6)
].set_index("benchmark")
inference_order = [
    ("equal_weight", "Equal weight"),
    ("inverse_vol_12m", "Inverse volatility"),
    ("momentum_6m_top2_invvol", "6-month momentum"),
    ("absolute_momentum_12m_cash", "Absolute momentum/cash"),
    ("DBC_buy_and_hold", "DBC"),
    ("GSG_buy_and_hold", "GSG"),
    ("ppo_10_seed_ensemble", "PPO ensemble"),
]
rows = []
for strategy, label in inference_order:
    row = selector_inference.loc[strategy]
    rows.append(
        f"{label} & {pct(row.annualized_mean_difference)} & "
        f"[{pct(row.ci_2_5)}, {pct(row.ci_97_5)}] & "
        f"{row.two_sided_centered_p_value:.3f} & "
        f"{row.holm_adjusted_p_value:.3f} \\\\"
    )
write(
    "table_selector_inference.tex",
    r"""\begin{table}[H]
\caption{Expanding selector pairwise inference, January 2019--May 2026.}
\label{tab:selector-inference}
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrr}
\toprule
Benchmark & Annualized mean diff. & 95\% CI & $p$ & Holm $p$\\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\begin{minipage}{\textwidth}\footnotesize
\emph{Notes:} Positive differences favor the supervised selector. Inference uses 5,000 circular moving-block bootstrap replications with a six-month block. Holm adjustment covers the displayed frozen pairwise family.
\end{minipage}
\end{table}
""",
)

concentration = pd.read_csv(ANALYSIS / "portfolio_concentration.csv")
concentration = concentration.loc[
    concentration["period"].eq("Combined OOS 2019-May 2026")
].set_index("strategy")
concentration_order = [
    ("momentum_6m_top2_invvol", "6-month momentum"),
    ("supervised_selector", "Expanding selector"),
    ("ppo_10_seed_ensemble", "PPO ensemble"),
]
rows = []
for strategy, label in concentration_order:
    row = concentration.loc[strategy]
    rows.append(
        f"{label} & {row.average_hhi:.3f} & {row.effective_assets:.2f} & "
        f"{pct(row.average_max_weight)} & {pct(row.average_weight_DBE)} & "
        f"{pct(row.average_weight_GLD)} & {pct(row.average_weight_DBA)} & "
        f"{pct(row.average_weight_DBB)} & {pct(row.average_weight_CASH)} \\\\"
    )
write(
    "table_concentration.tex",
    r"""\begin{table}[H]
\caption{Portfolio concentration and average allocation.}
\label{tab:concentration}
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lrrrrrrrr}
\toprule
Strategy & HHI & Effective assets & Max wt. & DBE & GLD & DBA & DBB & Cash\\
\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
""",
)

write(
    "table_data_design.tex",
    r"""\begin{table}[H]
\caption{Investable instruments, permanent identifiers, and roles.}
\label{tab:data-design}
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{llrl}
\toprule
Sleeve & Ticker & CRSP PERMNO & Empirical role\\
\midrule
Energy & DBE & 91709 & Portfolio asset\\
Gold & GLD & 90448 & Portfolio asset\\
Agriculture & DBA & 91712 & Portfolio asset\\
Base metals & DBB & 91715 & Portfolio asset\\
Broad commodities & GSG & 91381 & Investable benchmark\\
Broad commodities & DBC & 91129 & Investable benchmark\\
Cash & 30-day T-bill & -- & Risk-free return / defensive asset\\
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}
""",
)

print(f"Wrote publication tables to {OUT}")
