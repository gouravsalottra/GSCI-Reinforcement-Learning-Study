from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "publication_core_2024"
EXTENSION = ROOT / "gsci_live_extension_2026"
PPO_RESULTS_ZIP = ROOT / "models" / "ppo_fixed_results.zip"

sys.path.insert(0, str(ROOT / "src"))
from ppo_experiment import (  # noqa: E402
    ASSETS,
    FEATURE_COLUMNS,
    PORTFOLIO_ASSETS,
    TARGET_COLUMNS,
    ExperimentConfig,
    evaluate_policy,
    paired_block_bootstrap,
    performance_metrics,
)


def momentum_weights(panel: pd.DataFrame) -> np.ndarray:
    rows = []
    for _, row in panel.iterrows():
        momentum = np.array(
            [row[f"state__{asset.lower()}__momentum_6m"] for asset in ASSETS]
        )
        volatility = np.array(
            [row[f"state__{asset.lower()}__volatility_12m"] for asset in ASSETS]
        )
        selected = np.argsort(-momentum, kind="stable")[:2]
        weights = np.zeros(len(ASSETS))
        weights[selected] = 1.0 / volatility[selected]
        weights /= weights.sum()
        rows.append(weights)
    return np.asarray(rows)


def momentum_returns(
    panel: pd.DataFrame,
    weights: np.ndarray,
    monthly_asset_returns: pd.DataFrame,
    cost_bps: int = 10,
) -> pd.DataFrame:
    target_returns = panel[
        [f"target_next__{asset.lower()}" for asset in ASSETS]
    ].to_numpy(float)
    gross = np.sum(weights * target_returns, axis=1)
    turnover = np.zeros(len(panel))
    turnover[0] = np.nan

    for position in range(1, len(panel)):
        state_month = panel.index[position]
        realized_state_returns = monthly_asset_returns.loc[
            state_month, ASSETS
        ].to_numpy(float)
        pretrade = weights[position - 1] * (1.0 + realized_state_returns)
        pretrade /= pretrade.sum()
        turnover[position] = 0.5 * np.abs(weights[position] - pretrade).sum()

    return pd.DataFrame(
        {
            "state_month": panel.index,
            "target_month": panel["target_month"].to_numpy(),
            "gross_return": gross,
            "turnover": turnover,
            "net_return": gross - (cost_bps / 10_000.0) * turnover,
        }
    ).set_index("target_month")


def metric_row(
    period: str,
    strategy: str,
    returns: pd.Series,
    cash: pd.Series,
    turnover: pd.Series | None = None,
) -> dict[str, object]:
    aligned = pd.concat(
        [returns.rename("returns"), cash.rename("cash")], axis=1
    ).dropna()
    turnover_array = (
        turnover.reindex(aligned.index).to_numpy(float)
        if turnover is not None
        else None
    )
    return {
        "period": period,
        "strategy": strategy,
        **performance_metrics(
            aligned["returns"].to_numpy(float),
            aligned["cash"].to_numpy(float),
            turnover_array,
        ),
    }


def main() -> None:
    original = pd.read_parquet(
        DATA / "model_ready_primary_with_crsp_cash_2008_2024.parquet"
    ).sort_index()
    holdout = pd.read_parquet(
        EXTENSION / "model_ready_post_freeze_holdout_2025_2026.parquet"
    ).sort_index()
    holdout = holdout.loc[
        holdout["publication_status"] == "PRIMARY_ELIGIBLE"
    ].copy()
    monthly_returns = pd.read_parquet(
        EXTENSION / "extended_monthly_returns_through_2026_06.parquet"
    ).sort_index()

    train = original.loc[original["sample_split"] == "TRAIN"].copy()
    original_oos = original.loc[original["sample_split"] == "OOS"].copy()
    full_oos = pd.concat(
        [
            original_oos[FEATURE_COLUMNS + TARGET_COLUMNS + ["target_month"]],
            holdout[FEATURE_COLUMNS + TARGET_COLUMNS + ["target_month"]],
        ]
    )

    train_mean = train[FEATURE_COLUMNS].mean()
    train_std = train[FEATURE_COLUMNS].std(ddof=0)
    train_features = (
        (train[FEATURE_COLUMNS] - train_mean) / train_std
    ).to_numpy(np.float32)
    full_features = (
        (full_oos[FEATURE_COLUMNS] - train_mean) / train_std
    ).to_numpy(np.float32)
    train_returns = train[TARGET_COLUMNS].to_numpy(float)
    full_returns = full_oos[TARGET_COLUMNS].to_numpy(float)

    with tempfile.TemporaryDirectory(prefix="ppo_live_models_") as temp_dir:
        with zipfile.ZipFile(PPO_RESULTS_ZIP) as archive:
            model_members = [
                name for name in archive.namelist() if name.startswith("models/")
            ]
            for name in model_members:
                archive.extract(name, temp_dir)
        models = [
            PPO.load(
                Path(temp_dir) / "models" / f"ppo_seed_{seed}.zip",
                device="cpu",
            )
            for seed in range(1001, 1011)
        ]

        ppo_monthly, ppo_weights = evaluate_policy(
            models=models,
            warmup_features=train_features[-1],
            warmup_returns=train_returns[-1],
            oos_features=full_features,
            oos_returns=full_returns,
            state_months=pd.DatetimeIndex(full_oos.index),
            target_months=pd.DatetimeIndex(full_oos["target_month"]),
            cost_bps=10,
        )

    ppo_monthly = ppo_monthly.set_index("target_month").sort_index()
    ppo_weights = ppo_weights.set_index("target_month").sort_index()

    with zipfile.ZipFile(PPO_RESULTS_ZIP) as archive:
        with archive.open("ppo_ensemble_monthly_all_costs.parquet") as stream:
            saved = pd.read_parquet(stream)
    saved = saved.loc[saved["cost_bps"] == 10].set_index("target_month")
    reproduction_error = float(
        np.max(
            np.abs(
                ppo_monthly.loc[saved.index, "net_return"].to_numpy()
                - saved["net_return"].to_numpy()
            )
        )
    )
    if reproduction_error > 1e-15:
        raise RuntimeError(f"Frozen PPO reproduction failed: {reproduction_error}")

    all_states = pd.concat([original, holdout]).sort_index()
    momentum_w = momentum_weights(all_states)
    momentum_monthly = momentum_returns(
        all_states, momentum_w, monthly_returns, cost_bps=10
    )

    holdout_index = pd.DatetimeIndex(holdout["target_month"])
    full_index = pd.DatetimeIndex(full_oos["target_month"])
    result = pd.DataFrame(index=full_index)
    result.index.name = "target_month"
    result["ppo_10_seed_ensemble"] = ppo_monthly.loc[
        full_index, "net_return"
    ].to_numpy()
    result["momentum_6m_top2_invvol"] = momentum_monthly.loc[
        full_index, "net_return"
    ].to_numpy()
    result["cash"] = full_oos["target_next__cash"].to_numpy()
    result["gsg"] = np.concatenate(
        [
            original_oos["benchmark_next__gsg"].to_numpy(),
            holdout["benchmark_next__gsg"].to_numpy(),
        ]
    )
    result["dbc"] = np.concatenate(
        [
            original_oos["benchmark_next__dbc"].to_numpy(),
            holdout["benchmark_next__dbc"].to_numpy(),
        ]
    )
    result["sample_segment"] = np.where(
        result.index <= pd.Timestamp("2024-12-31"),
        "ORIGINAL_OOS",
        "FROZEN_POLICY_TEMPORAL_EXTENSION",
    )
    result.to_csv(EXTENSION / "frozen_strategy_returns_2019_2026_05.csv")
    result.to_parquet(EXTENSION / "frozen_strategy_returns_2019_2026_05.parquet")
    ppo_weights.to_parquet(EXTENSION / "ppo_weights_2019_2026_05.parquet")

    strategies = {
        "ppo_10_seed_ensemble": result["ppo_10_seed_ensemble"],
        "momentum_6m_top2_invvol": result["momentum_6m_top2_invvol"],
        "GSG_buy_and_hold": result["gsg"],
        "DBC_buy_and_hold": result["dbc"],
    }
    metric_rows = []
    for period, dates in {
        "original_OOS_2019_2024": (
            pd.Timestamp("2019-01-31"),
            pd.Timestamp("2024-12-31"),
        ),
        "temporal_extension_2025_01_2026_05": (
            pd.Timestamp("2025-01-31"),
            pd.Timestamp("2026-05-31"),
        ),
        "combined_2019_01_2026_05": (
            pd.Timestamp("2019-01-31"),
            pd.Timestamp("2026-05-31"),
        ),
    }.items():
        subset_index = result.loc[dates[0] : dates[1]].index
        for strategy, returns in strategies.items():
            turnover = (
                ppo_monthly["turnover"]
                if strategy == "ppo_10_seed_ensemble"
                else (
                    momentum_monthly["turnover"]
                    if strategy == "momentum_6m_top2_invvol"
                    else None
                )
            )
            metric_rows.append(
                metric_row(
                    period,
                    strategy,
                    returns.loc[subset_index],
                    result.loc[subset_index, "cash"],
                    turnover,
                )
            )
    pd.DataFrame(metric_rows).to_csv(
        EXTENSION / "frozen_strategy_performance_2019_2026_05.csv", index=False
    )

    holdout_result = result.loc[holdout_index].copy()
    sensitivity_rows = []
    for label, frame in {
        "all_17_months": holdout_result,
        "excluding_2026_03": holdout_result.drop(pd.Timestamp("2026-03-31")),
        "calendar_2025": holdout_result.loc["2025-01-31":"2025-12-31"],
        "2026_Jan_to_May": holdout_result.loc["2026-01-31":"2026-05-31"],
    }.items():
        for strategy in ["ppo_10_seed_ensemble", "momentum_6m_top2_invvol"]:
            turnover = (
                ppo_monthly["turnover"]
                if strategy == "ppo_10_seed_ensemble"
                else momentum_monthly["turnover"]
            )
            sensitivity_rows.append(
                metric_row(
                    label,
                    strategy,
                    frame[strategy],
                    frame["cash"],
                    turnover,
                )
            )
    pd.DataFrame(sensitivity_rows).to_csv(
        EXTENSION / "post_freeze_holdout_sensitivity.csv", index=False
    )

    bootstrap_config = ExperimentConfig(
        bootstrap_replications=5_000,
        bootstrap_block_length=3,
        bootstrap_seed=20260723,
    )
    bootstrap = paired_block_bootstrap(
        holdout_result["ppo_10_seed_ensemble"].to_numpy(),
        holdout_result["momentum_6m_top2_invvol"].to_numpy(),
        holdout_result["cash"].to_numpy(),
        bootstrap_config,
    )
    bootstrap.update(
        {
            "candidate": "ppo_10_seed_ensemble",
            "benchmark": "momentum_6m_top2_invvol",
            "period": "2025-01-31 through 2026-05-31",
        }
    )
    pd.DataFrame([bootstrap]).to_csv(
        EXTENSION / "post_freeze_ppo_vs_momentum_bootstrap.csv", index=False
    )

    diagnostics = {
        "frozen_ppo_reproduction_maximum_error": reproduction_error,
        "post_freeze_months": len(holdout_result),
        "post_freeze_start": str(holdout_result.index.min().date()),
        "post_freeze_end": str(holdout_result.index.max().date()),
        "no_retraining_or_hyperparameter_changes": True,
        "march_2026_ppo_return": float(
            holdout_result.loc[
                pd.Timestamp("2026-03-31"), "ppo_10_seed_ensemble"
            ]
        ),
        "march_2026_momentum_return": float(
            holdout_result.loc[
                pd.Timestamp("2026-03-31"), "momentum_6m_top2_invvol"
            ]
        ),
    }
    (EXTENSION / "live_evaluation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )

    print("\nPOST-FREEZE HOLDOUT RETURNS")
    print(holdout_result.to_string())
    print("\nPERFORMANCE")
    print(
        pd.read_csv(
            EXTENSION / "frozen_strategy_performance_2019_2026_05.csv"
        ).to_string(index=False)
    )
    print("\nSENSITIVITY")
    print(
        pd.read_csv(
            EXTENSION / "post_freeze_holdout_sensitivity.csv"
        ).to_string(index=False)
    )
    print("\nBOOTSTRAP")
    print(
        pd.read_csv(
            EXTENSION / "post_freeze_ppo_vs_momentum_bootstrap.csv"
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
