from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "publication_core_2024"
LIVE = ROOT / "gsci_live_extension_2026"
OUT = ROOT / "reproduced_outputs"
OUT.mkdir(exist_ok=True)

ASSETS = ["DBE", "GLD", "DBA", "DBB"]
PORTFOLIO_ASSETS = ASSETS + ["CASH"]
FEATURES = [
    f"state__{ticker.lower()}__momentum_{window}m"
    for ticker in ASSETS
    for window in [6, 12]
] + [
    f"state__{ticker.lower()}__volatility_12m"
    for ticker in ASSETS
]
EXPERTS = [
    "equal_weight",
    "inverse_vol_12m",
    "momentum_6m_top2_invvol",
    "momentum_12m_top2_invvol",
    "absolute_momentum_12m_cash",
]
PRIMARY_BENCHMARK = "momentum_6m_top2_invvol"
MODEL_SEED = 20260722
GBM_PARAMETERS = {
    "n_estimators": 50,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_samples_leaf": 10,
    "subsample": 1.0,
    "loss": "squared_error",
}
COST_GRID = [0, 5, 10, 25]
BLOCK_LENGTHS = [3, 6, 12]
BOOTSTRAP_REPLICATIONS = 5000
BOOTSTRAP_SEED = 20260722
ORIGINAL_END = pd.Timestamp("2024-12-31")
PUBLICATION_END = pd.Timestamp("2026-05-31")


def build_weights(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def empty() -> pd.DataFrame:
        return pd.DataFrame(0.0, index=panel.index, columns=PORTFOLIO_ASSETS)

    output: dict[str, pd.DataFrame] = {}
    weights = empty()
    weights[ASSETS] = 0.25
    output["equal_weight"] = weights

    inverse_vol = pd.DataFrame(
        {
            asset: 1.0
            / panel[f"state__{asset.lower()}__volatility_12m"].clip(lower=1e-8)
            for asset in ASSETS
        },
        index=panel.index,
    )
    weights = empty()
    weights[ASSETS] = inverse_vol.div(inverse_vol.sum(axis=1), axis=0)
    output["inverse_vol_12m"] = weights

    for window in [6, 12]:
        weights = empty()
        for state_month, row in panel.iterrows():
            momentum = pd.Series(
                {
                    asset: row[f"state__{asset.lower()}__momentum_{window}m"]
                    for asset in ASSETS
                }
            )
            selected = momentum.nlargest(2).index.tolist()
            inverse_selected_vol = pd.Series(
                {
                    asset: 1.0
                    / max(
                        row[f"state__{asset.lower()}__volatility_12m"], 1e-8
                    )
                    for asset in selected
                }
            )
            allocation = inverse_selected_vol / inverse_selected_vol.sum()
            weights.loc[state_month, selected] = allocation.values
        output[f"momentum_{window}m_top2_invvol"] = weights

    weights = empty()
    for state_month, row in panel.iterrows():
        momentum = pd.Series(
            {
                asset: row[f"state__{asset.lower()}__momentum_12m"]
                for asset in ASSETS
            }
        )
        winner = momentum.idxmax()
        if momentum[winner] > 0:
            weights.loc[state_month, winner] = 1.0
        else:
            weights.loc[state_month, "CASH"] = 1.0
    output["absolute_momentum_12m_cash"] = weights
    return output


def drift_adjusted_turnover(
    weights: pd.DataFrame, state_returns: pd.DataFrame
) -> pd.Series:
    turnover = pd.Series(0.0, index=weights.index, name="turnover")
    for position in range(1, len(weights)):
        state = weights.index[position]
        previous = weights.index[position - 1]
        end_values = weights.loc[previous] * (1.0 + state_returns.loc[state])
        drifted = end_values / end_values.sum()
        turnover.loc[state] = 0.5 * (weights.loc[state] - drifted).abs().sum()
    return turnover


def performance(
    monthly_returns: pd.Series,
    cash: pd.Series,
    turnover: pd.Series | None = None,
) -> dict[str, float | int]:
    returns = pd.Series(monthly_returns, dtype=float).dropna()
    cash = pd.Series(cash, dtype=float).reindex(returns.index)
    if cash.isna().any():
        raise RuntimeError("Missing cash in performance calculation.")
    months = len(returns)
    wealth = np.concatenate(
        [[1.0], np.cumprod(1.0 + returns.to_numpy(float))]
    )
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    ending_wealth = wealth[-1]
    excess = returns - cash
    excess_vol = excess.std(ddof=1)
    average_turnover = (
        0.0 if turnover is None else turnover.reindex(returns.index).mean()
    )
    return {
        "months": months,
        "total_return": ending_wealth - 1.0,
        "cagr": ending_wealth ** (12.0 / months) - 1.0,
        "arithmetic_annual_return": 12.0 * returns.mean(),
        "annual_volatility": np.sqrt(12.0) * returns.std(ddof=1),
        "excess_sharpe": (
            np.sqrt(12.0) * excess.mean() / excess_vol
            if excess_vol > 0
            else np.nan
        ),
        "maximum_drawdown": drawdown.min(),
        "monthly_hit_rate_over_cash": (returns > cash).mean(),
        "average_monthly_turnover": average_turnover,
        "annualized_turnover": 12.0 * average_turnover,
    }


def circular_indices(
    observations: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    blocks = math.ceil(observations / block_length)
    starts = rng.integers(0, observations, size=blocks)
    indices: list[int] = []
    for start in starts:
        indices.extend(
            ((np.arange(start, start + block_length)) % observations).tolist()
        )
    return np.asarray(indices[:observations], dtype=int)


def pairwise_bootstrap(
    candidate: pd.Series,
    benchmark: pd.Series,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    difference = (candidate - benchmark).to_numpy(float)
    observed = difference.mean()
    centered = difference - observed
    rng = np.random.default_rng(seed)
    boot_mean = np.empty(BOOTSTRAP_REPLICATIONS)
    boot_centered = np.empty(BOOTSTRAP_REPLICATIONS)
    for replication in range(BOOTSTRAP_REPLICATIONS):
        indices = circular_indices(len(difference), block_length, rng)
        boot_mean[replication] = difference[indices].mean()
        boot_centered[replication] = centered[indices].mean()
    p_value = (
        1.0 + np.sum(np.abs(boot_centered) >= abs(observed))
    ) / (BOOTSTRAP_REPLICATIONS + 1.0)
    return {
        "block_length": block_length,
        "replications": BOOTSTRAP_REPLICATIONS,
        "annualized_mean_difference": 12.0 * observed,
        "ci_2_5": 12.0 * np.percentile(boot_mean, 2.5),
        "ci_97_5": 12.0 * np.percentile(boot_mean, 97.5),
        "two_sided_centered_p_value": p_value,
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count)
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, (count - rank) * p_values[position])
        adjusted[position] = min(running, 1.0)
    return adjusted


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    original = pd.read_parquet(
        CORE / "model_ready_primary_with_crsp_cash_2008_2024.parquet"
    ).sort_index()
    holdout = pd.read_parquet(
        LIVE / "model_ready_post_freeze_holdout_2025_2026.parquet"
    ).sort_index()
    holdout = holdout.loc[
        holdout["publication_status"].eq("PRIMARY_ELIGIBLE")
    ].drop(columns=["publication_status"])
    panel = pd.concat([original, holdout]).sort_index()
    panel["target_month"] = pd.to_datetime(panel["target_month"])
    if panel["target_month"].max() != PUBLICATION_END:
        raise RuntimeError("Publication endpoint is not May 2026.")

    monthly = pd.read_parquet(
        LIVE / "extended_monthly_returns_through_2026_06.parquet"
    ).sort_index()
    state_returns = monthly[ASSETS + ["CASH"]].reindex(panel.index).astype(float)
    historical_cash = pd.read_parquet(
        CORE / "crsp_30day_tbill_monthly_return.parquet"
    ).iloc[:, 0]
    historical_cash.index = pd.to_datetime(historical_cash.index)
    state_returns["CASH"] = state_returns["CASH"].fillna(
        historical_cash.reindex(state_returns.index)
    )
    if state_returns.isna().any().any():
        raise RuntimeError("State returns contain missing values.")

    target_returns = pd.DataFrame(index=panel.index)
    for asset in ASSETS:
        target_returns[asset] = panel[f"target_next__{asset.lower()}"]
    target_returns["CASH"] = panel["target_next__cash"]

    expert_weights = build_weights(panel)
    expert_gross: dict[str, pd.Series] = {}
    expert_turnover: dict[str, pd.Series] = {}
    expert_net: dict[int, dict[str, pd.Series]] = {
        cost: {} for cost in COST_GRID
    }
    for expert, weights in expert_weights.items():
        if (weights.sum(axis=1) - 1.0).abs().max() > 1e-12:
            raise RuntimeError(f"Invalid weights for {expert}.")
        gross = (weights * target_returns).sum(axis=1)
        turnover = drift_adjusted_turnover(weights, state_returns)
        expert_gross[expert] = gross
        expert_turnover[expert] = turnover
        for cost in COST_GRID:
            expert_net[cost][expert] = gross - (cost / 10_000.0) * turnover

    # Exact historical baseline reproduction gate.
    saved_weights = pd.read_parquet(
        CORE / "baseline_target_weights_2008_2024.parquet"
    )
    baseline_audit_rows = []
    for expert in EXPERTS:
        saved = (
            saved_weights.loc[saved_weights["strategy"].eq(expert)]
            .set_index("state_month")[PORTFOLIO_ASSETS]
            .sort_index()
        )
        saved.index = pd.to_datetime(saved.index)
        error = (
            expert_weights[expert].loc[saved.index] - saved
        ).abs().to_numpy().max()
        baseline_audit_rows.append(
            {
                "strategy": expert,
                "historical_weight_maximum_error": error,
                "status": "PASS" if error <= 1e-12 else "FAIL",
            }
        )
    baseline_audit = pd.DataFrame(baseline_audit_rows)
    baseline_audit.to_csv(OUT / "baseline_reproduction_audit.csv", index=False)
    if not baseline_audit["status"].eq("PASS").all():
        raise RuntimeError("Historical baseline reproduction failed.")

    expert_gross_frame = pd.DataFrame(expert_gross)
    prediction_states = panel.index[
        panel["target_month"].between(pd.Timestamp("2018-12-31"), PUBLICATION_END)
    ]
    selector_rows = []
    selector_weight_rows = []
    for fold_number, state_month in enumerate(prediction_states, start=1):
        target_month = panel.loc[state_month, "target_month"]
        training_index = panel.index[panel["target_month"] < target_month]
        x_train_raw = panel.loc[training_index, FEATURES]
        x_test_raw = panel.loc[[state_month], FEATURES]
        fold_mean = x_train_raw.mean()
        fold_std = x_train_raw.std(ddof=0)
        x_train = (x_train_raw - fold_mean) / fold_std
        x_test = (x_test_raw - fold_mean) / fold_std
        predictions: dict[str, float] = {}
        actual: dict[str, float] = {}
        for expert_number, expert in enumerate(EXPERTS):
            model = GradientBoostingRegressor(
                random_state=MODEL_SEED + expert_number, **GBM_PARAMETERS
            )
            model.fit(x_train, expert_gross_frame.loc[training_index, expert])
            predictions[expert] = float(model.predict(x_test)[0])
            actual[expert] = float(expert_gross_frame.loc[state_month, expert])
        selected = max(
            EXPERTS, key=lambda expert: (predictions[expert], -EXPERTS.index(expert))
        )
        actual_best = max(
            EXPERTS, key=lambda expert: (actual[expert], -EXPERTS.index(expert))
        )
        selector_rows.append(
            {
                "fold_number": fold_number,
                "state_month": state_month,
                "target_month": target_month,
                "training_rows": len(training_index),
                "training_first_target": panel.loc[
                    training_index, "target_month"
                ].min(),
                "training_last_target": panel.loc[
                    training_index, "target_month"
                ].max(),
                "selected_expert": selected,
                "actual_best_expert": actual_best,
                "selected_expert_hit": selected == actual_best,
                "selected_actual_gross_return": actual[selected],
                "oracle_actual_gross_return": actual[actual_best],
                "oracle_gap": actual[actual_best] - actual[selected],
                **{f"predicted__{key}": value for key, value in predictions.items()},
                **{f"actual__{key}": value for key, value in actual.items()},
            }
        )
        selector_weight_rows.append(
            {
                "state_month": state_month,
                "target_month": target_month,
                "selected_expert": selected,
                **expert_weights[selected].loc[state_month].to_dict(),
            }
        )
    selector_predictions = (
        pd.DataFrame(selector_rows).set_index("state_month").sort_index()
    )
    selector_weights = (
        pd.DataFrame(selector_weight_rows).set_index("state_month").sort_index()
    )

    # Exact selector decision/weight reproduction gate.
    saved_predictions = pd.read_parquet(
        CORE / "selector_expanding_predictions.parquet"
    ).sort_index()
    saved_selector_weights = pd.read_parquet(
        CORE / "selector_expanding_weights.parquet"
    ).sort_index()
    historical_selector = selector_predictions.loc[saved_predictions.index]
    decisions_equal = historical_selector["selected_expert"].equals(
        saved_predictions["selected_expert"]
    )
    prediction_error = (
        historical_selector[
            [f"predicted__{expert}" for expert in EXPERTS]
        ]
        - saved_predictions[[f"predicted__{expert}" for expert in EXPERTS]]
    ).abs().to_numpy().max()
    selector_weight_error = (
        selector_weights.loc[saved_selector_weights.index, PORTFOLIO_ASSETS]
        - saved_selector_weights[PORTFOLIO_ASSETS]
    ).abs().to_numpy().max()
    selector_audit = pd.DataFrame(
        [
            {
                "historical_selected_experts_exact": decisions_equal,
                "historical_prediction_maximum_error": prediction_error,
                "historical_weight_maximum_error": selector_weight_error,
                "status": (
                    "PASS"
                    if decisions_equal
                    and selector_weight_error <= 1e-12
                    and prediction_error <= 1e-10
                    else "FAIL"
                ),
            }
        ]
    )
    selector_audit.to_csv(OUT / "selector_reproduction_audit.csv", index=False)
    if selector_audit.loc[0, "status"] != "PASS":
        raise RuntimeError("Historical selector reproduction failed.")

    selector_state_returns = state_returns.reindex(selector_weights.index)
    selector_target_returns = target_returns.reindex(selector_weights.index)
    selector_turnover = drift_adjusted_turnover(
        selector_weights[PORTFOLIO_ASSETS], selector_state_returns
    )
    selector_gross = (
        selector_weights[PORTFOLIO_ASSETS] * selector_target_returns
    ).sum(axis=1)
    selector_net = {
        cost: selector_gross - (cost / 10_000.0) * selector_turnover
        for cost in COST_GRID
    }

    # Rolling-window selector sensitivity. The estimator and expert set remain
    # frozen; only the amount of trailing training history changes.
    rolling_rows = []
    rolling_return_series: dict[int, pd.Series] = {}
    for training_window in [60, 96, 120]:
        rolling_weight_rows = []
        for state_month in prediction_states:
            target_month = panel.loc[state_month, "target_month"]
            eligible = panel.index[panel["target_month"] < target_month]
            training_index = eligible[-training_window:]
            if len(training_index) < training_window:
                continue
            x_train_raw = panel.loc[training_index, FEATURES]
            x_test_raw = panel.loc[[state_month], FEATURES]
            fold_mean = x_train_raw.mean()
            fold_std = x_train_raw.std(ddof=0)
            x_train = (x_train_raw - fold_mean) / fold_std
            x_test = (x_test_raw - fold_mean) / fold_std
            predictions = {}
            for expert_number, expert in enumerate(EXPERTS):
                model = GradientBoostingRegressor(
                    random_state=MODEL_SEED + expert_number, **GBM_PARAMETERS
                )
                model.fit(
                    x_train, expert_gross_frame.loc[training_index, expert]
                )
                predictions[expert] = float(model.predict(x_test)[0])
            selected = max(
                EXPERTS,
                key=lambda expert: (
                    predictions[expert],
                    -EXPERTS.index(expert),
                ),
            )
            rolling_weight_rows.append(
                {
                    "state_month": state_month,
                    "target_month": target_month,
                    "selected_expert": selected,
                    **expert_weights[selected].loc[state_month].to_dict(),
                }
            )
        rolling_weights = (
            pd.DataFrame(rolling_weight_rows)
            .set_index("state_month")
            .sort_index()
        )
        rolling_turnover = drift_adjusted_turnover(
            rolling_weights[PORTFOLIO_ASSETS],
            state_returns.reindex(rolling_weights.index),
        )
        rolling_gross = (
            rolling_weights[PORTFOLIO_ASSETS]
            * target_returns.reindex(rolling_weights.index)
        ).sum(axis=1)
        rolling_net = rolling_gross - 0.001 * rolling_turnover
        rolling_oos_states = rolling_weights.index[
            rolling_weights["target_month"].between(
                pd.Timestamp("2019-01-31"), PUBLICATION_END
            )
        ]
        rolling_target_months = pd.DatetimeIndex(
            rolling_weights.loc[rolling_oos_states, "target_month"]
        )
        rolling_return_series[training_window] = rolling_net.loc[
            rolling_oos_states
        ].set_axis(rolling_target_months)
        rolling_output = pd.DataFrame(
            {
                "selector_net_10bps": rolling_return_series[training_window],
                "momentum_6m_top2_invvol_net_10bps": expert_net[10][
                    PRIMARY_BENCHMARK
                ]
                .loc[rolling_oos_states]
                .set_axis(rolling_target_months),
                "cash": panel.loc[
                    rolling_oos_states, "target_next__cash"
                ].to_numpy(),
            },
            index=rolling_target_months,
        )
        rolling_output.index.name = "target_month"
        rolling_output.to_csv(
            OUT / f"selector_rolling_{training_window}m_monthly_returns.csv"
        )
        rolling_cash = panel.loc[
            rolling_oos_states, "target_next__cash"
        ].set_axis(rolling_target_months)
        rolling_turnover_target = rolling_turnover.loc[
            rolling_oos_states
        ].set_axis(rolling_target_months)
        rolling_rows.append(
            {
                "training_window_months": training_window,
                "first_target": rolling_target_months.min(),
                "last_target": rolling_target_months.max(),
                **performance(
                    rolling_return_series[training_window],
                    rolling_cash,
                    rolling_turnover_target,
                ),
            }
        )
        rolling_weights.to_parquet(
            OUT / f"selector_rolling_{training_window}m_weights.parquet"
        )
    pd.DataFrame(rolling_rows).to_csv(
        OUT / "selector_rolling_window_sensitivity.csv", index=False
    )
    rolling_inference_rows = []
    momentum_target_returns = expert_net[10][PRIMARY_BENCHMARK].loc[
        panel.index[
            panel["target_month"].between(
                pd.Timestamp("2019-01-31"), PUBLICATION_END
            )
        ]
    ]
    momentum_target_returns.index = pd.DatetimeIndex(
        panel.loc[momentum_target_returns.index, "target_month"]
    )
    for training_window, rolling_returns in rolling_return_series.items():
        for block_length in BLOCK_LENGTHS:
            rolling_inference_rows.append(
                {
                    "training_window_months": training_window,
                    **pairwise_bootstrap(
                        rolling_returns,
                        momentum_target_returns.reindex(rolling_returns.index),
                        block_length,
                        BOOTSTRAP_SEED
                        + 7000
                        + training_window
                        + block_length,
                    ),
                }
            )
    pd.DataFrame(rolling_inference_rows).to_csv(
        OUT / "selector_rolling_window_inference.csv", index=False
    )

    # Unified returns include all frozen methods and strong investable benchmarks.
    oos_states = panel.index[
        panel["target_month"].between(pd.Timestamp("2019-01-31"), PUBLICATION_END)
    ]
    target_months = pd.DatetimeIndex(panel.loc[oos_states, "target_month"])
    unified = pd.DataFrame(index=target_months)
    unified.index.name = "target_month"
    for expert in EXPERTS:
        unified[expert] = expert_net[10][expert].loc[oos_states].to_numpy()
    selector_oos_states = selector_predictions.index[
        selector_predictions["target_month"].between(
            pd.Timestamp("2019-01-31"), PUBLICATION_END
        )
    ]
    unified["supervised_selector"] = selector_net[10].loc[
        selector_oos_states
    ].to_numpy()
    frozen = pd.read_parquet(
        LIVE / "frozen_strategy_returns_2019_2026_05.parquet"
    ).sort_index()
    unified["ppo_10_seed_ensemble"] = frozen.loc[
        unified.index, "ppo_10_seed_ensemble"
    ]
    unified["GSG_buy_and_hold"] = panel.loc[
        oos_states, "benchmark_next__gsg"
    ].to_numpy()
    unified["DBC_buy_and_hold"] = panel.loc[
        oos_states, "benchmark_next__dbc"
    ].to_numpy()
    unified["cash_30day_tbill"] = panel.loc[
        oos_states, "target_next__cash"
    ].to_numpy()
    unified["sample_segment"] = np.where(
        unified.index <= ORIGINAL_END,
        "ORIGINAL_OOS",
        "FROZEN_POLICY_TEMPORAL_EXTENSION",
    )
    unified.to_csv(OUT / "unified_monthly_returns_2019_2026_05.csv")
    unified.to_parquet(OUT / "unified_monthly_returns_2019_2026_05.parquet")

    # Save all weights and selector diagnostics.
    weight_frames = []
    for name, weights in expert_weights.items():
        frame = weights.loc[oos_states].copy()
        frame["strategy"] = name
        frame["target_month"] = target_months
        frame["state_month"] = frame.index
        weight_frames.append(frame.reset_index(drop=True))
    selector_frame = selector_weights.loc[selector_oos_states].copy()
    selector_frame["strategy"] = "supervised_selector"
    selector_frame["state_month"] = selector_frame.index
    weight_frames.append(selector_frame.reset_index(drop=True))
    all_weights = pd.concat(weight_frames, ignore_index=True)
    all_weights.to_parquet(OUT / "all_active_strategy_weights.parquet", index=False)
    selector_predictions.to_parquet(
        OUT / "selector_expanding_predictions_through_may_2026.parquet"
    )
    selector_weights.to_parquet(
        OUT / "selector_expanding_weights_through_may_2026.parquet"
    )

    turnover_map = {
        **{
            name: series.loc[oos_states].set_axis(target_months)
            for name, series in expert_turnover.items()
        },
        "supervised_selector": selector_turnover.loc[
            selector_oos_states
        ].set_axis(target_months),
    }
    ppo_weights = pd.read_parquet(
        LIVE / "ppo_weights_2019_2026_05.parquet"
    ).sort_index()
    ppo_turnover = pd.read_parquet(
        LIVE / "frozen_strategy_returns_2019_2026_05.parquet"
    )
    # PPO turnover is reconstructed from the saved target weights.
    ppo_w = ppo_weights.reindex(target_months)
    ppo_w_columns = [column for column in ppo_w.columns if column in PORTFOLIO_ASSETS]
    if len(ppo_w_columns) != 5:
        raise RuntimeError(f"Unexpected PPO weight columns: {ppo_weights.columns}")
    ppo_state_w = ppo_w[ppo_w_columns].copy()
    ppo_state_w.columns = PORTFOLIO_ASSETS
    ppo_state_w.index = oos_states
    turnover_map["ppo_10_seed_ensemble"] = drift_adjusted_turnover(
        ppo_state_w, state_returns.loc[oos_states]
    ).set_axis(target_months)

    # Period performance.
    period_definitions = {
        "Original OOS 2019-2024": (
            pd.Timestamp("2019-01-31"),
            pd.Timestamp("2024-12-31"),
        ),
        "Temporal extension 2025-May 2026": (
            pd.Timestamp("2025-01-31"),
            PUBLICATION_END,
        ),
        "Combined OOS 2019-May 2026": (
            pd.Timestamp("2019-01-31"),
            PUBLICATION_END,
        ),
        "Excluding 2020-2021": (
            pd.Timestamp("2019-01-31"),
            PUBLICATION_END,
        ),
        "Excluding March 2026": (
            pd.Timestamp("2019-01-31"),
            PUBLICATION_END,
        ),
    }
    return_columns = [
        column
        for column in unified.columns
        if column not in ["sample_segment"]
    ]
    performance_rows = []
    for period, (start, end) in period_definitions.items():
        period_index = unified.loc[start:end].index
        if period == "Excluding 2020-2021":
            period_index = period_index[~period_index.year.isin([2020, 2021])]
        if period == "Excluding March 2026":
            period_index = period_index[
                period_index != pd.Timestamp("2026-03-31")
            ]
        for strategy in return_columns:
            row = performance(
                unified.loc[period_index, strategy],
                unified.loc[period_index, "cash_30day_tbill"],
                turnover_map[strategy].loc[period_index]
                if strategy in turnover_map
                else None,
            )
            performance_rows.append(
                {"period": period, "strategy": strategy, **row}
            )
    performance_table = pd.DataFrame(performance_rows)
    performance_table.to_csv(OUT / "performance_all_periods.csv", index=False)

    # Transaction cost sensitivity for the active rules and selector.
    cost_rows = []
    full_index = unified.index
    for cost in COST_GRID:
        for expert in EXPERTS:
            returns = expert_net[cost][expert].loc[oos_states].set_axis(target_months)
            cost_rows.append(
                {
                    "strategy": expert,
                    "cost_bps": cost,
                    **performance(
                        returns,
                        unified["cash_30day_tbill"],
                        turnover_map[expert],
                    ),
                }
            )
        selector_returns = selector_net[cost].loc[
            selector_oos_states
        ].set_axis(target_months)
        cost_rows.append(
            {
                "strategy": "supervised_selector",
                "cost_bps": cost,
                **performance(
                    selector_returns,
                    unified["cash_30day_tbill"],
                    turnover_map["supervised_selector"],
                ),
            }
        )
        # PPO saved evaluation is 10 bps; infer gross exactly from saved turnover.
        ppo_net_10 = unified["ppo_10_seed_ensemble"]
        ppo_gross = ppo_net_10 + 0.001 * turnover_map["ppo_10_seed_ensemble"]
        ppo_returns = ppo_gross - (cost / 10_000.0) * turnover_map[
            "ppo_10_seed_ensemble"
        ]
        cost_rows.append(
            {
                "strategy": "ppo_10_seed_ensemble",
                "cost_bps": cost,
                **performance(
                    ppo_returns,
                    unified["cash_30day_tbill"],
                    turnover_map["ppo_10_seed_ensemble"],
                ),
            }
        )
    pd.DataFrame(cost_rows).to_csv(
        OUT / "transaction_cost_sensitivity.csv", index=False
    )

    # Pairwise inference against the frozen primary benchmark.
    comparison_set = [
        strategy
        for strategy in return_columns
        if strategy not in [PRIMARY_BENCHMARK, "cash_30day_tbill"]
    ]
    bootstrap_rows = []
    for block_length in BLOCK_LENGTHS:
        for number, candidate in enumerate(comparison_set):
            result = pairwise_bootstrap(
                unified[candidate],
                unified[PRIMARY_BENCHMARK],
                block_length,
                BOOTSTRAP_SEED + 100 * block_length + number,
            )
            bootstrap_rows.append(
                {
                    "candidate": candidate,
                    "benchmark": PRIMARY_BENCHMARK,
                    **result,
                }
            )
    bootstrap = pd.DataFrame(bootstrap_rows)
    primary_mask = bootstrap["block_length"].eq(6)
    bootstrap.loc[primary_mask, "holm_adjusted_p_value"] = holm_adjust(
        bootstrap.loc[primary_mask, "two_sided_centered_p_value"].to_numpy()
    )
    bootstrap.to_csv(OUT / "pairwise_block_bootstrap.csv", index=False)

    selector_pairwise_rows = []
    selector_benchmarks = [
        "equal_weight",
        "inverse_vol_12m",
        "momentum_6m_top2_invvol",
        "momentum_12m_top2_invvol",
        "absolute_momentum_12m_cash",
        "GSG_buy_and_hold",
        "DBC_buy_and_hold",
        "ppo_10_seed_ensemble",
    ]
    for block_length in BLOCK_LENGTHS:
        for number, benchmark in enumerate(selector_benchmarks):
            selector_pairwise_rows.append(
                {
                    "candidate": "supervised_selector",
                    "benchmark": benchmark,
                    **pairwise_bootstrap(
                        unified["supervised_selector"],
                        unified[benchmark],
                        block_length,
                        BOOTSTRAP_SEED + 3000 + 100 * block_length + number,
                    ),
                }
            )
    selector_pairwise = pd.DataFrame(selector_pairwise_rows)
    selector_primary_mask = selector_pairwise["block_length"].eq(6)
    selector_pairwise.loc[
        selector_primary_mask, "holm_adjusted_p_value"
    ] = holm_adjust(
        selector_pairwise.loc[
            selector_primary_mask, "two_sided_centered_p_value"
        ].to_numpy()
    )
    selector_pairwise.to_csv(
        OUT / "selector_pairwise_block_bootstrap.csv", index=False
    )

    # Frozen-set centered max-mean reality check.
    differences = pd.DataFrame(
        {
            strategy: unified[strategy] - unified[PRIMARY_BENCHMARK]
            for strategy in comparison_set
        }
    )
    observed_means = differences.mean()
    observed_max = observed_means.max()
    reality_rows = []
    for block_length in BLOCK_LENGTHS:
        centered = differences - differences.mean()
        rng = np.random.default_rng(BOOTSTRAP_SEED + 1000 + block_length)
        boot_max = np.empty(BOOTSTRAP_REPLICATIONS)
        for replication in range(BOOTSTRAP_REPLICATIONS):
            indices = circular_indices(len(centered), block_length, rng)
            boot_max[replication] = centered.iloc[indices].mean().max()
        p_value = (
            1.0 + np.sum(boot_max >= observed_max)
        ) / (BOOTSTRAP_REPLICATIONS + 1.0)
        reality_rows.append(
            {
                "benchmark": PRIMARY_BENCHMARK,
                "candidate_count": len(comparison_set),
                "candidate_set": " | ".join(comparison_set),
                "observed_best_candidate": observed_means.idxmax(),
                "observed_max_annualized_mean_difference": 12.0 * observed_max,
                "block_length": block_length,
                "replications": BOOTSTRAP_REPLICATIONS,
                "centered_max_mean_p_value": p_value,
            }
        )
    pd.DataFrame(reality_rows).to_csv(
        OUT / "centered_max_mean_reality_check.csv", index=False
    )

    # Deflated Sharpe Ratio diagnostic for the explicitly frozen strategy set.
    # This does not pretend to correct for every abandoned historical experiment.
    from scipy.stats import kurtosis, norm, skew

    trial_count = len(return_columns) - 1  # exclude cash
    euler_gamma = 0.5772156649015329
    expected_max_standard_normal = (
        (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trial_count)
        + euler_gamma
        * norm.ppf(1.0 - 1.0 / (trial_count * math.e))
    )
    dsr_rows = []
    for strategy in [column for column in return_columns if column != "cash_30day_tbill"]:
        excess = (
            unified[strategy] - unified["cash_30day_tbill"]
        ).to_numpy(float)
        monthly_sharpe = excess.mean() / excess.std(ddof=1)
        sharpe_standard_error = math.sqrt(
            (
                1.0
                - skew(excess, bias=False) * monthly_sharpe
                + (kurtosis(excess, fisher=False, bias=False) - 1.0)
                * monthly_sharpe**2
                / 4.0
            )
            / (len(excess) - 1.0)
        )
        null_max_sharpe = expected_max_standard_normal / math.sqrt(
            len(excess) - 1.0
        )
        dsr_probability = norm.cdf(
            (monthly_sharpe - null_max_sharpe) / sharpe_standard_error
        )
        dsr_rows.append(
            {
                "strategy": strategy,
                "observations": len(excess),
                "frozen_trial_count": trial_count,
                "annualized_excess_sharpe": np.sqrt(12.0) * monthly_sharpe,
                "estimated_skewness": skew(excess, bias=False),
                "estimated_kurtosis": kurtosis(
                    excess, fisher=False, bias=False
                ),
                "deflated_sharpe_probability": dsr_probability,
                "scope_warning": (
                    "Corrects only for the explicitly frozen strategy set, "
                    "not all historical exploratory trials"
                ),
            }
        )
    pd.DataFrame(dsr_rows).to_csv(
        OUT / "deflated_sharpe_frozen_set.csv", index=False
    )

    # Leave-one-year-out and annual excess contribution.
    annual_rows = []
    leave_rows = []
    for strategy in comparison_set:
        difference = unified[strategy] - unified[PRIMARY_BENCHMARK]
        for year, values in difference.groupby(difference.index.year):
            annual_rows.append(
                {
                    "strategy": strategy,
                    "year": year,
                    "months": len(values),
                    "annualized_arithmetic_excess": 12.0 * values.mean(),
                    "compounded_excess_wealth_difference": (
                        np.prod(1.0 + unified.loc[values.index, strategy])
                        - np.prod(
                            1.0 + unified.loc[values.index, PRIMARY_BENCHMARK]
                        )
                    ),
                }
            )
        for year in sorted(unified.index.year.unique()):
            index = unified.index[unified.index.year != year]
            leave_rows.append(
                {
                    "strategy": strategy,
                    "excluded_year": year,
                    "months": len(index),
                    "annualized_mean_difference": 12.0
                    * (
                        unified.loc[index, strategy]
                        - unified.loc[index, PRIMARY_BENCHMARK]
                    ).mean(),
                    "candidate_cagr": performance(
                        unified.loc[index, strategy],
                        unified.loc[index, "cash_30day_tbill"],
                    )["cagr"],
                    "benchmark_cagr": performance(
                        unified.loc[index, PRIMARY_BENCHMARK],
                        unified.loc[index, "cash_30day_tbill"],
                    )["cagr"],
                }
            )
    pd.DataFrame(annual_rows).to_csv(
        OUT / "annual_excess_contributions.csv", index=False
    )
    pd.DataFrame(leave_rows).to_csv(
        OUT / "leave_one_year_out.csv", index=False
    )

    # Selection signal diagnostics.
    selector_eval = selector_predictions.loc[selector_oos_states]
    ic_rows = []
    for state, row in selector_eval.iterrows():
        predicted = [row[f"predicted__{expert}"] for expert in EXPERTS]
        actual = [row[f"actual__{expert}"] for expert in EXPERTS]
        ic_rows.append(
            {
                "state_month": state,
                "target_month": row["target_month"],
                "spearman_ic": spearmanr(predicted, actual).statistic,
            }
        )
    selector_diagnostics = pd.DataFrame(
        [
            {
                "months": len(selector_eval),
                "best_expert_hit_rate": selector_eval[
                    "selected_expert_hit"
                ].mean(),
                "mean_oracle_gap_monthly": selector_eval["oracle_gap"].mean(),
                "mean_cross_sectional_spearman_ic": pd.DataFrame(ic_rows)[
                    "spearman_ic"
                ].mean(),
                "average_monthly_turnover": turnover_map[
                    "supervised_selector"
                ].mean(),
            }
        ]
    )
    selector_diagnostics.to_csv(
        OUT / "selector_signal_diagnostics.csv", index=False
    )
    (
        selector_eval["selected_expert"]
        .value_counts()
        .reindex(EXPERTS, fill_value=0)
        .rename_axis("expert")
        .reset_index(name="selected_months")
        .assign(selection_fraction=lambda frame: frame["selected_months"] / len(selector_eval))
        .to_csv(OUT / "selector_expert_frequency.csv", index=False)
    )

    # Concentration and attribution.
    concentration_rows = []
    for strategy, weights in {
        **{name: frame.loc[oos_states] for name, frame in expert_weights.items()},
        "supervised_selector": selector_weights.loc[
            selector_oos_states, PORTFOLIO_ASSETS
        ],
        "ppo_10_seed_ensemble": ppo_state_w,
    }.items():
        aligned_weights = weights.copy()
        aligned_weights.index = target_months
        hhi = (aligned_weights**2).sum(axis=1)
        max_weight = aligned_weights.max(axis=1)
        for period, index in {
            "Combined OOS 2019-May 2026": target_months,
            "Temporal extension 2025-May 2026": target_months[
                target_months >= pd.Timestamp("2025-01-31")
            ],
        }.items():
            concentration_rows.append(
                {
                    "strategy": strategy,
                    "period": period,
                    "months": len(index),
                    "average_hhi": hhi.loc[index].mean(),
                    "effective_assets": 1.0 / hhi.loc[index].mean(),
                    "average_max_weight": max_weight.loc[index].mean(),
                    "months_max_weight_gt_50pct": int(
                        (max_weight.loc[index] > 0.5).sum()
                    ),
                    **{
                        f"average_weight_{asset}": aligned_weights.loc[
                            index, asset
                        ].mean()
                        for asset in PORTFOLIO_ASSETS
                    },
                }
            )
    pd.DataFrame(concentration_rows).to_csv(
        OUT / "portfolio_concentration.csv", index=False
    )

    ppo_weight_target = ppo_state_w.copy()
    ppo_weight_target.index = target_months
    asset_returns_by_target = target_returns.loc[oos_states].copy()
    asset_returns_by_target.index = target_months
    ppo_contribution = ppo_weight_target * asset_returns_by_target
    ppo_contribution["gross_return"] = ppo_contribution.sum(axis=1)
    ppo_contribution["turnover"] = turnover_map["ppo_10_seed_ensemble"]
    ppo_contribution["cost_10bps"] = (
        0.001 * turnover_map["ppo_10_seed_ensemble"]
    )
    ppo_contribution["net_return"] = unified["ppo_10_seed_ensemble"]
    ppo_contribution.to_csv(OUT / "ppo_monthly_asset_contributions.csv")

    # Compact publication figures.
    plt.style.use("seaborn-v0_8-whitegrid")
    display_names = {
        "momentum_6m_top2_invvol": "6m momentum",
        "supervised_selector": "Supervised selector",
        "ppo_10_seed_ensemble": "PPO (10-seed ensemble)",
        "GSG_buy_and_hold": "GSG",
        "DBC_buy_and_hold": "DBC",
        "equal_weight": "Equal weight",
    }
    key = list(display_names)
    wealth = (1.0 + unified[key]).cumprod()
    wealth.columns = [display_names[column] for column in wealth.columns]
    ax = wealth.plot(figsize=(10.5, 5.8), linewidth=1.8)
    ax.set_title("Growth of $1: Unified Out-of-Sample Evidence")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("")
    ax.legend(ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(OUT / "figure_cumulative_wealth.pdf")
    plt.savefig(OUT / "figure_cumulative_wealth.png", dpi=240)
    plt.close()

    drawdowns = wealth / wealth.cummax() - 1.0
    ax = drawdowns.plot(figsize=(10.5, 5.8), linewidth=1.6)
    ax.set_title("Out-of-Sample Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("")
    ax.legend(ncol=2, frameon=True)
    plt.tight_layout()
    plt.savefig(OUT / "figure_drawdowns.pdf")
    plt.savefig(OUT / "figure_drawdowns.png", dpi=240)
    plt.close()

    selected_annual = pd.DataFrame(annual_rows)
    selected_annual = selected_annual.loc[
        selected_annual["strategy"].isin(
            ["supervised_selector", "ppo_10_seed_ensemble"]
        )
    ]
    pivot = selected_annual.pivot(
        index="year", columns="strategy", values="annualized_arithmetic_excess"
    )
    pivot.columns = [display_names[column] for column in pivot.columns]
    ax = pivot.plot.bar(figsize=(10.5, 5.8), width=0.78)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Annualized Mean Return Difference vs 6m Momentum")
    ax.set_ylabel("Annualized difference")
    ax.set_xlabel("")
    ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(OUT / "figure_annual_excess.pdf")
    plt.savefig(OUT / "figure_annual_excess.png", dpi=240)
    plt.close()

    # Final machine-readable decision summary.
    combined = performance_table.loc[
        performance_table["period"].eq("Combined OOS 2019-May 2026")
    ].set_index("strategy")
    temporal_extension = performance_table.loc[
        performance_table["period"].eq("Temporal extension 2025-May 2026")
    ].set_index("strategy")
    march = pd.Timestamp("2026-03-31")
    decision = {
        "publication_sample": {
            "first_target": str(unified.index.min().date()),
            "last_target": str(unified.index.max().date()),
            "months": len(unified),
            "temporal_extension_months": int(
                (unified.index >= pd.Timestamp("2025-01-31")).sum()
            ),
        },
        "combined_primary_findings": {
            strategy: {
                "cagr": float(combined.loc[strategy, "cagr"]),
                "excess_sharpe": float(combined.loc[strategy, "excess_sharpe"]),
                "maximum_drawdown": float(
                    combined.loc[strategy, "maximum_drawdown"]
                ),
            }
            for strategy in [
                PRIMARY_BENCHMARK,
                "supervised_selector",
                "ppo_10_seed_ensemble",
                "GSG_buy_and_hold",
                "DBC_buy_and_hold",
            ]
        },
        "temporal_extension_findings": {
            strategy: {
                "cagr": float(temporal_extension.loc[strategy, "cagr"]),
                "excess_sharpe": float(
                    temporal_extension.loc[strategy, "excess_sharpe"]
                ),
            }
            for strategy in [
                PRIMARY_BENCHMARK,
                "supervised_selector",
                "ppo_10_seed_ensemble",
                "GSG_buy_and_hold",
                "DBC_buy_and_hold",
            ]
        },
        "march_2026": {
            "ppo_net_return": float(
                unified.loc[march, "ppo_10_seed_ensemble"]
            ),
            "momentum_net_return": float(
                unified.loc[march, PRIMARY_BENCHMARK]
            ),
            "ppo_dbe_weight": float(ppo_weight_target.loc[march, "DBE"]),
            "dbe_return": float(asset_returns_by_target.loc[march, "DBE"]),
        },
        "reproduction_gates": {
            "baselines": baseline_audit.to_dict(orient="records"),
            "selector": selector_audit.to_dict(orient="records"),
        },
    }
    (OUT / "final_decision_summary.json").write_text(
        json.dumps(decision, indent=2, default=str)
    )

    manifest_rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "sha256_manifest.csv":
            manifest_rows.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(OUT / "sha256_manifest.csv", index=False)

    print("=" * 100)
    print("FULL PUBLICATION ANALYSIS COMPLETE")
    print("=" * 100)
    print("Rows:", len(unified))
    print("Target range:", unified.index.min().date(), "to", unified.index.max().date())
    print("\nCombined performance:")
    print(
        combined.loc[
            [
                PRIMARY_BENCHMARK,
                "supervised_selector",
                "ppo_10_seed_ensemble",
                "GSG_buy_and_hold",
                "DBC_buy_and_hold",
            ],
            ["cagr", "excess_sharpe", "maximum_drawdown"],
        ].to_string()
    )
    print("\nOutput:", OUT)


if __name__ == "__main__":
    main()
