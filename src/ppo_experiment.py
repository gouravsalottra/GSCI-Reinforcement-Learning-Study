from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed


ASSETS = ["DBE", "GLD", "DBA", "DBB"]
PORTFOLIO_ASSETS = ASSETS + ["CASH"]
FEATURE_COLUMNS = [
    f"state__{ticker.lower()}__{feature}"
    for ticker in ASSETS
    for feature in ("momentum_6m", "momentum_12m", "volatility_12m")
]
TARGET_COLUMNS = [f"target_next__{ticker.lower()}" for ticker in ASSETS] + [
    "target_next__cash"
]


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = tuple(range(1001, 1011))
    transaction_cost_bps: int = 10
    cost_sensitivity_bps: tuple[int, ...] = (0, 5, 10, 25)
    episode_length: int = 24
    total_timesteps: int = 50_000
    reward_scale: float = 100.0
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.20
    ent_coef: float = 0.005
    vf_coef: float = 0.50
    max_grad_norm: float = 0.50
    policy_width: int = 32
    bootstrap_replications: int = 5_000
    bootstrap_block_length: int = 6
    bootstrap_seed: int = 20260723


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax_weights(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    # PPO acts in the recommended normalized [-1, 1] box. Multiplying by five
    # preserves a wide simplex range before the softmax transformation.
    logits = 5.0 * action
    shifted = logits - np.max(logits)
    exponentiated = np.exp(shifted)
    weights = exponentiated / exponentiated.sum()
    return weights.astype(np.float64)


def drift_weights(weights: np.ndarray, realized_returns: np.ndarray) -> np.ndarray:
    end_values = weights * (1.0 + realized_returns)
    denominator = float(end_values.sum())
    if denominator <= 0:
        raise RuntimeError("Portfolio value became nonpositive.")
    return end_values / denominator


class CommodityPortfolioEnv(gym.Env):
    """Train-only episodic environment with long-only simplex allocations."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        next_returns: np.ndarray,
        episode_length: int,
        cost_rate: float,
        reward_scale: float,
    ) -> None:
        super().__init__()
        self.features = np.asarray(features, dtype=np.float32)
        self.next_returns = np.asarray(next_returns, dtype=np.float64)
        self.episode_length = int(episode_length)
        self.cost_rate = float(cost_rate)
        self.reward_scale = float(reward_scale)

        if self.features.ndim != 2 or self.next_returns.ndim != 2:
            raise ValueError("Features and returns must be two-dimensional.")
        if len(self.features) != len(self.next_returns):
            raise ValueError("Feature and return row counts differ.")
        if self.features.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError("Unexpected state-feature count.")
        if self.next_returns.shape[1] != len(PORTFOLIO_ASSETS):
            raise ValueError("Unexpected portfolio-return count.")
        if len(self.features) < self.episode_length:
            raise ValueError("Episode length exceeds the training sample.")

        observation_size = self.features.shape[1] + len(PORTFOLIO_ASSETS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(PORTFOLIO_ASSETS),),
            dtype=np.float32,
        )

        self._row = 0
        self._episode_stop = 0
        self._pretrade_weights = np.full(
            len(PORTFOLIO_ASSETS), 1.0 / len(PORTFOLIO_ASSETS), dtype=np.float64
        )

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [self.features[self._row], self._pretrade_weights.astype(np.float32)]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        latest_start = len(self.features) - self.episode_length
        self._row = int(self.np_random.integers(0, latest_start + 1))
        self._episode_stop = self._row + self.episode_length
        self._pretrade_weights = np.full(
            len(PORTFOLIO_ASSETS), 1.0 / len(PORTFOLIO_ASSETS), dtype=np.float64
        )
        return self._observation(), {"start_row": self._row}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        weights = softmax_weights(action)
        turnover = 0.5 * float(np.abs(weights - self._pretrade_weights).sum())
        realized = self.next_returns[self._row]
        gross_return = float(weights @ realized)
        net_return = gross_return - self.cost_rate * turnover
        cash_return = float(realized[-1])

        if net_return <= -1.0 or cash_return <= -1.0:
            raise RuntimeError("Invalid log-return input.")

        reward = self.reward_scale * (
            math.log1p(net_return) - math.log1p(cash_return)
        )
        self._pretrade_weights = drift_weights(weights, realized)
        self._row += 1

        truncated = self._row >= self._episode_stop
        terminated = False
        info = {
            "weights": weights,
            "turnover": turnover,
            "gross_return": gross_return,
            "net_return": net_return,
            "cash_return": cash_return,
        }

        if truncated:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            observation = self._observation()

        return observation, float(reward), terminated, truncated, info


def load_and_validate_data(data_dir: Path) -> dict[str, Any]:
    input_hash_path = data_dir / "input_sha256.json"
    expected_hashes = json.loads(input_hash_path.read_text())
    for filename, expected in expected_hashes.items():
        actual = sha256_file(data_dir / filename)
        if actual != expected:
            raise RuntimeError(
                f"Input hash mismatch for {filename}: {actual} != {expected}"
            )

    panel = pd.read_parquet(
        data_dir / "model_ready_primary_with_crsp_cash_2008_2024.parquet"
    ).sort_index()
    raw_returns = pd.read_parquet(
        data_dir / "verified_monthly_total_return_panel_NO_FILL.parquet"
    ).sort_index()
    baselines = pd.read_parquet(
        data_dir / "baseline_monthly_returns_2008_2024.parquet"
    ).sort_index()

    panel.index = pd.to_datetime(panel.index)
    raw_returns.index = pd.to_datetime(raw_returns.index)
    baselines.index = pd.to_datetime(baselines.index)
    panel["target_month"] = pd.to_datetime(panel["target_month"])
    baselines["target_month"] = pd.to_datetime(baselines["target_month"])

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise RuntimeError(f"Data-contract failure: {name}: {detail}")

    record("panel_rows", len(panel) == 204, f"observed={len(panel)} expected=204")
    record(
        "training_rows",
        int((panel["sample_split"] == "TRAIN").sum()) == 132,
        f"observed={int((panel['sample_split'] == 'TRAIN').sum())} expected=132",
    )
    record(
        "oos_rows",
        int((panel["sample_split"] == "OOS").sum()) == 72,
        f"observed={int((panel['sample_split'] == 'OOS').sum())} expected=72",
    )
    record(
        "feature_count",
        len(FEATURE_COLUMNS) == 12 and set(FEATURE_COLUMNS).issubset(panel.columns),
        f"features={len(FEATURE_COLUMNS)}",
    )
    record(
        "no_missing_model_values",
        not panel[FEATURE_COLUMNS + TARGET_COLUMNS].isna().any().any(),
        str(panel[FEATURE_COLUMNS + TARGET_COLUMNS].isna().sum().to_dict()),
    )
    record(
        "target_range",
        panel["target_month"].min() == pd.Timestamp("2008-01-31")
        and panel["target_month"].max() == pd.Timestamp("2024-12-31"),
        f"{panel['target_month'].min()} to {panel['target_month'].max()}",
    )
    record(
        "split_boundary",
        panel.loc[panel["sample_split"] == "TRAIN", "target_month"].max()
        == pd.Timestamp("2018-12-31")
        and panel.loc[panel["sample_split"] == "OOS", "target_month"].min()
        == pd.Timestamp("2019-01-31"),
        "TRAIN ends 2018-12; OOS begins 2019-01",
    )
    record(
        "monthly_contiguity",
        panel["target_month"].equals(
            pd.Series(
                pd.date_range("2008-01-31", "2024-12-31", freq="ME"),
                index=panel.index,
                name="target_month",
            )
        ),
        "Target months must be contiguous.",
    )

    alignment_errors: dict[str, float] = {}
    for ticker in ASSETS:
        expected = raw_returns[ticker].reindex(panel["target_month"]).to_numpy()
        actual = panel[f"target_next__{ticker.lower()}"].to_numpy(dtype=float)
        alignment_errors[ticker] = float(np.max(np.abs(expected - actual)))
    record(
        "target_alignment",
        max(alignment_errors.values()) <= 1e-15,
        json.dumps(alignment_errors, sort_keys=True),
    )
    record(
        "baseline_alignment",
        panel.index.equals(baselines.index)
        and panel["target_month"].equals(baselines["target_month"]),
        "Baseline and model panels share state and target dates.",
    )

    return {
        "panel": panel,
        "raw_returns": raw_returns,
        "baselines": baselines,
        "checks": checks,
    }


def configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed, using_cuda=False)


def train_model(
    seed: int,
    train_features: np.ndarray,
    train_returns: np.ndarray,
    config: ExperimentConfig,
) -> PPO:
    configure_determinism(seed)
    env = CommodityPortfolioEnv(
        features=train_features,
        next_returns=train_returns,
        episode_length=config.episode_length,
        cost_rate=config.transaction_cost_bps / 10_000.0,
        reward_scale=config.reward_scale,
    )
    env.reset(seed=seed)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        normalize_advantage=True,
        policy_kwargs={
            "activation_fn": torch.nn.Tanh,
            "net_arch": {
                "pi": [config.policy_width, config.policy_width],
                "vf": [config.policy_width, config.policy_width],
            },
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=config.total_timesteps, progress_bar=False)
    return model


def policy_weights(
    model: PPO, standardized_features: np.ndarray, pretrade: np.ndarray
) -> np.ndarray:
    observation = np.concatenate(
        [standardized_features.astype(np.float32), pretrade.astype(np.float32)]
    ).astype(np.float32)
    action, _ = model.predict(observation, deterministic=True)
    return softmax_weights(action)


def evaluate_policy(
    models: list[PPO],
    warmup_features: np.ndarray,
    warmup_returns: np.ndarray,
    oos_features: np.ndarray,
    oos_returns: np.ndarray,
    state_months: pd.DatetimeIndex,
    target_months: pd.DatetimeIndex,
    cost_bps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pretrade = np.full(
        len(PORTFOLIO_ASSETS), 1.0 / len(PORTFOLIO_ASSETS), dtype=np.float64
    )

    warmup_actions = [
        policy_weights(model, warmup_features, pretrade) for model in models
    ]
    warmup_weights = np.mean(warmup_actions, axis=0)
    pretrade = drift_weights(warmup_weights, warmup_returns)

    rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    cost_rate = cost_bps / 10_000.0

    for position in range(len(oos_features)):
        component_weights = [
            policy_weights(model, oos_features[position], pretrade) for model in models
        ]
        weights = np.mean(component_weights, axis=0)
        weights = weights / weights.sum()
        turnover = 0.5 * float(np.abs(weights - pretrade).sum())
        gross_return = float(weights @ oos_returns[position])
        net_return = gross_return - cost_rate * turnover

        rows.append(
            {
                "state_month": state_months[position],
                "target_month": target_months[position],
                "gross_return": gross_return,
                "turnover": turnover,
                "net_return": net_return,
                "cash_return": float(oos_returns[position, -1]),
            }
        )
        weight_rows.append(
            {
                "state_month": state_months[position],
                "target_month": target_months[position],
                **{
                    asset: float(weights[index])
                    for index, asset in enumerate(PORTFOLIO_ASSETS)
                },
            }
        )
        pretrade = drift_weights(weights, oos_returns[position])

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def maximum_drawdown(returns: np.ndarray) -> float:
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    running_peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / running_peak - 1.0))


def performance_metrics(
    returns: np.ndarray, cash: np.ndarray, turnover: np.ndarray | None = None
) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    cash = np.asarray(cash, dtype=float)
    months = len(returns)
    ending_wealth = float(np.prod(1.0 + returns))
    excess = returns - cash
    excess_std = float(np.std(excess, ddof=1))
    drawdown = maximum_drawdown(returns)
    cagr = ending_wealth ** (12.0 / months) - 1.0
    return {
        "months": int(months),
        "cagr": float(cagr),
        "arithmetic_annual_return": float(12.0 * np.mean(returns)),
        "annual_volatility": float(np.sqrt(12.0) * np.std(returns, ddof=1)),
        "excess_sharpe": (
            float(np.sqrt(12.0) * np.mean(excess) / excess_std)
            if excess_std > 0
            else np.nan
        ),
        "maximum_drawdown": drawdown,
        "calmar": float(cagr / abs(drawdown)) if drawdown < 0 else np.nan,
        "total_return": float(ending_wealth - 1.0),
        "average_monthly_turnover": (
            float(np.mean(turnover)) if turnover is not None else 0.0
        ),
        "annualized_turnover": (
            float(12.0 * np.mean(turnover)) if turnover is not None else 0.0
        ),
    }


def circular_block_indices(
    observations: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    blocks_needed = math.ceil(observations / block_length)
    starts = rng.integers(0, observations, size=blocks_needed)
    indices: list[int] = []
    for start in starts:
        indices.extend(((np.arange(start, start + block_length)) % observations).tolist())
    return np.asarray(indices[:observations], dtype=int)


def paired_block_bootstrap(
    candidate: np.ndarray,
    benchmark: np.ndarray,
    cash: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float]:
    difference = candidate - benchmark
    observed_mean = float(np.mean(difference))
    centered = difference - observed_mean
    observed_sharpe_difference = (
        performance_metrics(candidate, cash)["excess_sharpe"]
        - performance_metrics(benchmark, cash)["excess_sharpe"]
    )
    rng = np.random.default_rng(config.bootstrap_seed)
    boot_mean = np.empty(config.bootstrap_replications)
    boot_centered = np.empty(config.bootstrap_replications)
    boot_sharpe = np.empty(config.bootstrap_replications)

    for replication in range(config.bootstrap_replications):
        indices = circular_block_indices(
            len(difference), config.bootstrap_block_length, rng
        )
        boot_mean[replication] = np.mean(difference[indices])
        boot_centered[replication] = np.mean(centered[indices])
        boot_sharpe[replication] = (
            performance_metrics(candidate[indices], cash[indices])["excess_sharpe"]
            - performance_metrics(benchmark[indices], cash[indices])["excess_sharpe"]
        )

    p_value = (
        1.0
        +
        np.sum(np.abs(boot_centered) >= abs(observed_mean))
    ) / (config.bootstrap_replications + 1.0)
    return {
        "annualized_mean_difference": 12.0 * observed_mean,
        "annualized_mean_ci_2_5": 12.0 * float(np.percentile(boot_mean, 2.5)),
        "annualized_mean_ci_97_5": 12.0 * float(np.percentile(boot_mean, 97.5)),
        "two_sided_centered_p_value": float(p_value),
        "observed_sharpe_difference": float(observed_sharpe_difference),
        "sharpe_difference_ci_2_5": float(np.percentile(boot_sharpe, 2.5)),
        "sharpe_difference_ci_97_5": float(np.percentile(boot_sharpe, 97.5)),
        "block_length": config.bootstrap_block_length,
        "replications": config.bootstrap_replications,
    }


def write_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append(
                {
                    "filename": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(root / "sha256_manifest.csv", index=False)
    return manifest


def run_experiment(
    data_dir: Path,
    output_zip: Path,
    config: ExperimentConfig | None = None,
) -> Path:
    config = config or ExperimentConfig()
    loaded = load_and_validate_data(data_dir)
    panel: pd.DataFrame = loaded["panel"]
    baselines: pd.DataFrame = loaded["baselines"]
    checks: list[dict[str, Any]] = loaded["checks"]

    train = panel.loc[panel["sample_split"] == "TRAIN"].copy()
    oos = panel.loc[panel["sample_split"] == "OOS"].copy()
    train_mean = train[FEATURE_COLUMNS].mean()
    train_std = train[FEATURE_COLUMNS].std(ddof=0)
    if (train_std <= 0).any():
        raise RuntimeError("Nonpositive training feature standard deviation.")

    train_features = ((train[FEATURE_COLUMNS] - train_mean) / train_std).to_numpy(
        dtype=np.float32
    )
    oos_features = ((oos[FEATURE_COLUMNS] - train_mean) / train_std).to_numpy(
        dtype=np.float32
    )
    train_returns = train[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    oos_returns = oos[TARGET_COLUMNS].to_numpy(dtype=np.float64)

    check_env(
        CommodityPortfolioEnv(
            train_features,
            train_returns,
            config.episode_length,
            config.transaction_cost_bps / 10_000.0,
            config.reward_scale,
        ),
        warn=True,
    )

    warmup_features = train_features[-1]
    warmup_returns = train_returns[-1]
    state_months = pd.DatetimeIndex(oos.index)
    target_months = pd.DatetimeIndex(oos["target_month"])

    work_root = Path(tempfile.mkdtemp(prefix="ppo_fixed_results_"))
    results_dir = work_root / "ppo_fixed_results"
    models_dir = results_dir / "models"
    models_dir.mkdir(parents=True)

    (results_dir / "ppo_config.json").write_text(
        json.dumps(asdict(config), indent=2)
    )
    pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "training_mean": train_mean.values,
            "training_std_ddof0": train_std.values,
        }
    ).to_csv(results_dir / "train_only_scaler.csv", index=False)
    pd.DataFrame(checks).to_csv(results_dir / "data_contract_checks.csv", index=False)

    models: list[PPO] = []
    seed_monthly_frames: list[pd.DataFrame] = []
    seed_weight_frames: list[pd.DataFrame] = []
    seed_performance_rows: list[dict[str, Any]] = []

    for seed in config.seeds:
        print(f"Training fixed PPO seed {seed}...", flush=True)
        model = train_model(seed, train_features, train_returns, config)
        model.save(models_dir / f"ppo_seed_{seed}")
        models.append(model)
        monthly, weights = evaluate_policy(
            [model],
            warmup_features,
            warmup_returns,
            oos_features,
            oos_returns,
            state_months,
            target_months,
            config.transaction_cost_bps,
        )
        monthly["seed"] = seed
        weights["seed"] = seed
        seed_monthly_frames.append(monthly)
        seed_weight_frames.append(weights)
        metrics = performance_metrics(
            monthly["net_return"].to_numpy(),
            monthly["cash_return"].to_numpy(),
            monthly["turnover"].to_numpy(),
        )
        metrics["seed"] = seed
        seed_performance_rows.append(metrics)

    seed_monthly = pd.concat(seed_monthly_frames, ignore_index=True)
    seed_weights = pd.concat(seed_weight_frames, ignore_index=True)
    seed_performance = pd.DataFrame(seed_performance_rows).sort_values("seed")
    seed_monthly.to_parquet(results_dir / "ppo_seed_monthly.parquet", index=False)
    seed_weights.to_parquet(results_dir / "ppo_seed_weights.parquet", index=False)
    seed_performance.to_csv(results_dir / "ppo_seed_performance.csv", index=False)

    ensemble_cost_frames: list[pd.DataFrame] = []
    ensemble_performance_rows: list[dict[str, Any]] = []
    ensemble_weights: pd.DataFrame | None = None

    for cost_bps in config.cost_sensitivity_bps:
        monthly, weights = evaluate_policy(
            models,
            warmup_features,
            warmup_returns,
            oos_features,
            oos_returns,
            state_months,
            target_months,
            cost_bps,
        )
        monthly["cost_bps"] = cost_bps
        ensemble_cost_frames.append(monthly)
        metrics = performance_metrics(
            monthly["net_return"].to_numpy(),
            monthly["cash_return"].to_numpy(),
            monthly["turnover"].to_numpy(),
        )
        metrics.update({"strategy": "ppo_10_seed_ensemble", "cost_bps": cost_bps})
        ensemble_performance_rows.append(metrics)
        if cost_bps == config.transaction_cost_bps:
            ensemble_weights = weights

    ensemble_monthly_all = pd.concat(ensemble_cost_frames, ignore_index=True)
    ensemble_monthly_all.to_parquet(
        results_dir / "ppo_ensemble_monthly_all_costs.parquet", index=False
    )
    pd.DataFrame(ensemble_performance_rows).to_csv(
        results_dir / "ppo_ensemble_cost_sensitivity.csv", index=False
    )
    if ensemble_weights is None:
        raise RuntimeError("Primary-cost ensemble weights were not created.")
    ensemble_weights.to_parquet(
        results_dir / "ppo_ensemble_weights_10bps.parquet", index=False
    )

    primary_ensemble = ensemble_monthly_all.loc[
        ensemble_monthly_all["cost_bps"] == config.transaction_cost_bps
    ].copy()
    primary_ensemble = primary_ensemble.set_index("state_month").sort_index()
    baseline_oos = baselines.reindex(state_months)
    momentum = baseline_oos["momentum_6m_top2_invvol__net_10bps"].to_numpy(
        dtype=float
    )
    inverse_vol = baseline_oos["inverse_vol_12m__net_10bps"].to_numpy(dtype=float)
    equal_weight = baseline_oos["equal_weight__net_10bps"].to_numpy(dtype=float)
    cash = primary_ensemble["cash_return"].to_numpy(dtype=float)
    ppo_return = primary_ensemble["net_return"].to_numpy(dtype=float)

    comparison_rows = []
    for name, returns, turnover in [
        (
            "ppo_10_seed_ensemble",
            ppo_return,
            primary_ensemble["turnover"].to_numpy(dtype=float),
        ),
        (
            "momentum_6m_top2_invvol",
            momentum,
            baseline_oos["momentum_6m_top2_invvol__turnover"].to_numpy(
                dtype=float
            ),
        ),
        (
            "inverse_vol_12m",
            inverse_vol,
            baseline_oos["inverse_vol_12m__turnover"].to_numpy(dtype=float),
        ),
        (
            "equal_weight",
            equal_weight,
            baseline_oos["equal_weight__turnover"].to_numpy(dtype=float),
        ),
        (
            "GSG_buy_and_hold",
            baseline_oos["GSG_buy_and_hold"].to_numpy(dtype=float),
            None,
        ),
        (
            "DBC_buy_and_hold",
            baseline_oos["DBC_buy_and_hold"].to_numpy(dtype=float),
            None,
        ),
    ]:
        row = performance_metrics(returns, cash, turnover)
        row["strategy"] = name
        row["cost_bps"] = config.transaction_cost_bps
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "excess_sharpe", ascending=False
    )
    comparison.to_csv(results_dir / "ppo_primary_comparison.csv", index=False)

    bootstrap = paired_block_bootstrap(ppo_return, momentum, cash, config)
    bootstrap.update(
        {
            "candidate": "ppo_10_seed_ensemble",
            "benchmark": "momentum_6m_top2_invvol",
        }
    )
    pd.DataFrame([bootstrap]).to_csv(
        results_dir / "ppo_vs_momentum_block_bootstrap.csv", index=False
    )

    weight_summary = ensemble_weights[PORTFOLIO_ASSETS].agg(
        ["mean", "std", "min", "max"]
    ).T
    weight_summary.index.name = "asset"
    weight_summary.to_csv(results_dir / "ppo_ensemble_weight_summary.csv")

    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "gymnasium": gym.__version__,
        "stable_baselines3": __import__("stable_baselines3").__version__,
    }
    (results_dir / "runtime_versions.json").write_text(
        json.dumps(versions, indent=2)
    )

    final_checks = [
        {
            "check": "every_seed_reported",
            "passed": len(seed_performance) == len(config.seeds),
            "detail": f"{len(seed_performance)} of {len(config.seeds)}",
        },
        {
            "check": "ensemble_oos_rows",
            "passed": len(primary_ensemble) == 72,
            "detail": f"observed={len(primary_ensemble)} expected=72",
        },
        {
            "check": "ensemble_weights_sum_to_one",
            "passed": bool(
                np.allclose(
                    ensemble_weights[PORTFOLIO_ASSETS].sum(axis=1).to_numpy(),
                    1.0,
                    atol=1e-12,
                )
            ),
            "detail": "Tolerance 1e-12",
        },
        {
            "check": "no_negative_ensemble_weights",
            "passed": bool(
                (ensemble_weights[PORTFOLIO_ASSETS].to_numpy() >= -1e-12).all()
            ),
            "detail": "Long-only constraint",
        },
        {
            "check": "turnover_bounds",
            "passed": bool(
                (
                    (primary_ensemble["turnover"] >= -1e-12)
                    & (primary_ensemble["turnover"] <= 1.0 + 1e-12)
                ).all()
            ),
            "detail": "One-half L1 turnover in [0,1]",
        },
    ]
    pd.DataFrame(final_checks).to_csv(
        results_dir / "ppo_output_checks.csv", index=False
    )
    if not all(row["passed"] for row in final_checks):
        raise RuntimeError("One or more PPO output checks failed.")

    summary = {
        "primary_strategy": "ppo_10_seed_ensemble",
        "best_seed_reporting_prohibited": True,
        "training_rows": 132,
        "oos_rows": 72,
        "oos_target_start": "2019-01-31",
        "oos_target_end": "2024-12-31",
        "primary_transaction_cost_bps": config.transaction_cost_bps,
        "primary_benchmark": "momentum_6m_top2_invvol",
        "ppo_metrics": comparison.loc[
            comparison["strategy"] == "ppo_10_seed_ensemble"
        ].iloc[0].to_dict(),
        "momentum_metrics": comparison.loc[
            comparison["strategy"] == "momentum_6m_top2_invvol"
        ].iloc[0].to_dict(),
        "bootstrap": bootstrap,
    }
    (results_dir / "ppo_decision_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    write_manifest(results_dir)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(results_dir.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(results_dir)))

    shutil.rmtree(work_root)
    return output_zip


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    run_experiment(
        data_dir=project_root / "data",
        output_zip=project_root / "ppo_fixed_results.zip",
    )
    print(project_root / "ppo_fixed_results.zip")
