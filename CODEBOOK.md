# Codebook

## Core timing convention

Each row is indexed by a state month \(t\). State features use information available at the end of month \(t\); the target is the investable return in month \(t+1\). Chronological splits are defined by target month, not by state month.

## State features

For each of DBE, GLD, DBA, and DBB:

- `momentum_6m`: compounded total return over the six months ending at the state month;
- `momentum_12m`: compounded total return over the twelve months ending at the state month;
- `volatility_12m`: sample standard deviation of the twelve monthly total returns ending at the state month.

The primary state therefore contains 12 market-only variables. Scaling parameters are estimated only on the 2008–2018 training sample.

## Targets and returns

- `target_next__{ticker}`: next-month total return for the named sleeve;
- `cash_30day_tbill`: contemporaneous monthly CRSP 30-day Treasury return in the licensed segment and the documented public proxy in the temporal extension;
- active-strategy returns: gross portfolio return less 10 basis points per unit of drift-adjusted one-half-L1 turnover;
- passive GSG/DBC returns: buy-and-hold ETF total returns with zero modeled strategy turnover.

## Strategy identifiers

| Identifier | Definition |
|---|---|
| `equal_weight` | 25% in each rotation sleeve |
| `inverse_vol_12m` | Four-sleeve inverse-12-month-volatility allocation |
| `momentum_6m_top2_invvol` | Select the top two six-month-momentum sleeves; inverse-volatility allocate between them |
| `momentum_12m_top2_invvol` | Twelve-month version of the preceding rule |
| `absolute_momentum_12m_cash` | Hold the best 12-month-momentum sleeve if positive; otherwise cash |
| `supervised_selector` | Gradient-boosted selector among five transparent experts |
| `ppo_10_seed_ensemble` | Mean allocation of ten prespecified PPO policies |
| `GSG_buy_and_hold` | Investable broad-commodity benchmark |
| `DBC_buy_and_hold` | Investable broad-commodity benchmark |

## Performance fields

- `total_return`: terminal compounded wealth minus one;
- `cagr`: terminal wealth annualized by the observed month count;
- `arithmetic_annual_return`: 12 times the mean monthly return;
- `annual_volatility`: monthly sample standard deviation times square root of 12;
- `excess_sharpe`: annualized mean monthly return in excess of contemporaneous cash divided by monthly excess-return volatility;
- `maximum_drawdown`: minimum wealth relative to its previous running maximum;
- `annualized_turnover`: 12 times mean monthly one-half-L1 drift-adjusted turnover.

## Sample labels

- `ORIGINAL_OOS`: January 2019–December 2024 targets;
- `FROZEN_POLICY_TEMPORAL_EXTENSION`: January 2025–May 2026 targets;
- combined OOS: all 89 targets from January 2019 through May 2026.

Some immutable legacy filenames use `post_freeze`; the article and documentation use “frozen-policy temporal extension.”
