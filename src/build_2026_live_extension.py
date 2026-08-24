from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import math
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CRSP_DIR = ROOT / "publication_core_2024"
OUT = ROOT / "gsci_live_extension_2026"

TICKERS = ["DBE", "GLD", "DBA", "DBB", "GSG", "DBC"]
PRIMARY_ASSETS = ["DBE", "GLD", "DBA", "DBB"]
PERMNOS = {
    "DBE": 91709,
    "GLD": 90448,
    "DBA": 91712,
    "DBB": 91715,
    "GSG": 91381,
    "DBC": 91129,
}

YAHOO_START = "2023-12-01"
YAHOO_END_EXCLUSIVE = "2026-07-01"
NASDAQ_START = "2024-01-01"
FINAL_PRIMARY_TARGET = pd.Timestamp("2026-05-31")
PROVISIONAL_TARGET = pd.Timestamp("2026-06-30")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 Chrome/150 Safari/537.36"
)


def get_bytes(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/csv,*/*",
        },
    )
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}") from error


def utc_epoch(date_text: str) -> int:
    return int(
        datetime.strptime(date_text, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def fetch_yahoo(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    query = urllib.parse.urlencode(
        {
            "period1": utc_epoch(YAHOO_START),
            "period2": utc_epoch(YAHOO_END_EXCLUSIVE),
            "interval": "1d",
            "events": "div,splits",
        }
    )
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}?{query}"
    payload = json.loads(get_bytes(url))
    result = payload["chart"]["result"][0]
    timestamps = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None)
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    frame = pd.DataFrame(
        {
            "date": timestamps.normalize(),
            "ticker": ticker,
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "adjusted_close": adjusted,
            "volume": quote["volume"],
        }
    ).dropna(subset=["close", "adjusted_close"])

    event_rows: list[dict[str, object]] = []
    for event_type, entries in result.get("events", {}).items():
        for entry in entries.values():
            event_rows.append(
                {
                    "ticker": ticker,
                    "event_type": event_type,
                    "date": pd.to_datetime(
                        entry["date"], unit="s", utc=True
                    ).tz_convert(None).normalize(),
                    "amount": entry.get("amount"),
                    "numerator": entry.get("numerator"),
                    "denominator": entry.get("denominator"),
                    "split_ratio": entry.get("splitRatio"),
                }
            )
    events = pd.DataFrame(event_rows)
    return frame, events


def parse_money(value: object) -> float:
    if value is None:
        return np.nan
    text = str(value).replace("$", "").replace(",", "").strip()
    return float(text) if text not in {"", "N/A", "--"} else np.nan


def fetch_nasdaq(ticker: str) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {
            "assetclass": "etf",
            "fromdate": NASDAQ_START,
            "limit": 5000,
        }
    )
    url = f"https://api.nasdaq.com/api/quote/{ticker}/historical?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/",
        },
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read())
    rows = payload["data"]["tradesTable"]["rows"]
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = ticker
    for column in ["close", "open", "high", "low"]:
        if column in frame:
            frame[column] = frame[column].map(parse_money)
    if "volume" in frame:
        frame["volume"] = pd.to_numeric(
            frame["volume"].str.replace(",", "", regex=False), errors="coerce"
        )
    return frame[["date", "ticker", "open", "high", "low", "close", "volume"]]


def yahoo_monthly_total_returns(daily: pd.DataFrame) -> pd.DataFrame:
    prices = daily.pivot(index="date", columns="ticker", values="adjusted_close")
    monthly_prices = prices.resample("ME").last()
    return monthly_prices.pct_change()


def nasdaq_monthly_price_returns(daily: pd.DataFrame) -> pd.DataFrame:
    prices = daily.pivot(index="date", columns="ticker", values="close")
    monthly_prices = prices.resample("ME").last()
    return monthly_prices.pct_change()


def load_french_rf() -> tuple[pd.Series, bytes]:
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_CSV.zip"
    )
    archive_bytes = get_bytes(url)
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")

    rows: list[tuple[pd.Timestamp, float]] = []
    for line in text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 5 and len(fields[0]) == 6 and fields[0].isdigit():
            date = pd.Timestamp(
                year=int(fields[0][:4]), month=int(fields[0][4:]), day=1
            ) + pd.offsets.MonthEnd(0)
            rows.append((date, float(fields[4]) / 100.0))
    series = pd.Series(dict(rows), name="french_rf").sort_index()
    return series, archive_bytes


def load_fred_dgs1mo() -> tuple[pd.Series, bytes]:
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv?"
        "id=DGS1MO&cosd=2025-01-01&coed=2026-06-30"
    )
    csv_bytes = get_bytes(url)
    frame = pd.read_csv(io.BytesIO(csv_bytes), parse_dates=["observation_date"])
    frame["DGS1MO"] = pd.to_numeric(frame["DGS1MO"], errors="coerce")
    return frame.dropna().set_index("observation_date")["DGS1MO"], csv_bytes


def june_cash_proxy(dgs1mo: pd.Series) -> float:
    june = dgs1mo.loc["2026-06-01":"2026-06-30"]
    if june.empty:
        raise RuntimeError("No June 2026 DGS1MO observation was downloaded.")
    annual_yield = float(june.iloc[0]) / 100.0
    return (1.0 + annual_yield) ** (30.0 / 365.0) - 1.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)

    nasdaq_frames = []
    for ticker in TICKERS:
        print(f"Downloading {ticker} from Nasdaq...", flush=True)
        nasdaq_frames.append(fetch_nasdaq(ticker))

    nasdaq_daily = pd.concat(nasdaq_frames, ignore_index=True).sort_values(
        ["ticker", "date"]
    )

    nasdaq_daily.to_parquet(OUT / "nasdaq_daily_close_2024_2026.parquet", index=False)
    nasdaq_monthly = nasdaq_monthly_price_returns(nasdaq_daily)
    nasdaq_monthly.to_parquet(OUT / "nasdaq_monthly_price_returns.parquet")

    crsp = pd.read_parquet(
        CRSP_DIR / "verified_monthly_total_return_panel_NO_FILL.parquet"
    ).sort_index()
    crsp.index = pd.to_datetime(crsp.index)
    model_panel = pd.read_parquet(
        CRSP_DIR / "model_ready_primary_with_crsp_cash_2008_2024.parquet"
    ).sort_index()
    model_panel["target_month"] = pd.to_datetime(model_panel["target_month"])

    overlap = crsp[TICKERS].join(
        nasdaq_monthly[TICKERS],
        how="inner",
        lsuffix="_crsp",
        rsuffix="_nasdaq",
    ).loc["2024-01-31":"2025-12-31"]
    overlap_rows = []
    for ticker in TICKERS:
        difference = overlap[f"{ticker}_nasdaq"] - overlap[f"{ticker}_crsp"]
        non_december = difference.loc[difference.index.month != 12]
        overlap_rows.append(
            {
                "ticker": ticker,
                "overlap_months": int(difference.notna().sum()),
                "mean_absolute_difference": float(difference.abs().mean()),
                "maximum_absolute_difference": float(difference.abs().max()),
                "correlation": float(
                    overlap[f"{ticker}_nasdaq"].corr(overlap[f"{ticker}_crsp"])
                ),
                "non_december_months": int(non_december.notna().sum()),
                "non_december_mean_absolute_difference": float(
                    non_december.abs().mean()
                ),
                "non_december_maximum_absolute_difference": float(
                    non_december.abs().max()
                ),
            }
        )
    monthly_audit = pd.DataFrame(overlap_rows)
    monthly_audit.to_csv(OUT / "nasdaq_vs_crsp_monthly_audit.csv", index=False)

    french_rf, french_raw = load_french_rf()
    dgs1mo, fred_raw = load_fred_dgs1mo()
    (OUT / "source_fama_french_factors.zip").write_bytes(french_raw)
    (OUT / "source_fred_dgs1mo.csv").write_bytes(fred_raw)
    french_rf.rename("cash_return").to_frame().to_parquet(
        OUT / "fama_french_monthly_rf.parquet"
    )

    existing_cash = model_panel.set_index("target_month")["target_next__cash"]
    cash_overlap = pd.concat(
        [existing_cash.rename("crsp_mcti_t30ret"), french_rf], axis=1
    ).dropna()
    cash_overlap = cash_overlap.loc["2008-01-31":"2024-12-31"]
    cash_overlap["difference"] = (
        cash_overlap["french_rf"] - cash_overlap["crsp_mcti_t30ret"]
    )
    cash_overlap.to_csv(OUT / "french_rf_vs_crsp_cash_monthly_overlap.csv")
    pd.DataFrame(
        [
            {
                "overlap_months": len(cash_overlap),
                "mean_absolute_difference": float(
                    cash_overlap["difference"].abs().mean()
                ),
                "maximum_absolute_difference": float(
                    cash_overlap["difference"].abs().max()
                ),
                "correlation": float(
                    cash_overlap["french_rf"].corr(
                        cash_overlap["crsp_mcti_t30ret"]
                    )
                ),
            }
        ]
    ).to_csv(OUT / "french_rf_vs_crsp_cash_audit_summary.csv", index=False)

    extended_assets = crsp[TICKERS].copy()
    for month in pd.date_range("2026-01-31", "2026-06-30", freq="ME"):
        extended_assets.loc[month, TICKERS] = nasdaq_monthly.loc[month, TICKERS]
    extended_assets = extended_assets.sort_index()

    cash = existing_cash.copy()
    for month in pd.date_range("2025-01-31", "2026-05-31", freq="ME"):
        cash.loc[month] = french_rf.loc[month]
    cash.loc[PROVISIONAL_TARGET] = june_cash_proxy(dgs1mo)
    cash = cash.sort_index()

    extended = extended_assets.join(cash.rename("CASH"), how="left")
    extended = extended.loc[:PROVISIONAL_TARGET].copy()
    extended["asset_source"] = np.where(
        extended.index <= pd.Timestamp("2025-12-31"),
        "CRSP CIZ msf_v2.mthret",
        "Nasdaq daily closing-price return",
    )
    extended["cash_source"] = np.select(
        [
            extended.index <= pd.Timestamp("2024-12-31"),
            extended.index <= FINAL_PRIMARY_TARGET,
        ],
        [
            "CRSP mcti.t30ret",
            "Kenneth French monthly RF",
        ],
        default="FRED DGS1MO beginning-June yield conversion (provisional)",
    )
    extended["publication_status"] = np.where(
        extended.index <= FINAL_PRIMARY_TARGET,
        "PRIMARY_ELIGIBLE",
        "PROVISIONAL_JUNE",
    )
    extended.to_parquet(OUT / "extended_monthly_returns_through_2026_06.parquet")
    extended.to_csv(OUT / "extended_monthly_returns_through_2026_06.csv")

    feature_frame = pd.DataFrame(index=extended.index)
    for ticker in PRIMARY_ASSETS:
        returns = extended[ticker]
        feature_frame[f"state__{ticker.lower()}__momentum_6m"] = (
            (1.0 + returns).rolling(6).apply(np.prod, raw=True) - 1.0
        )
        feature_frame[f"state__{ticker.lower()}__momentum_12m"] = (
            (1.0 + returns).rolling(12).apply(np.prod, raw=True) - 1.0
        )
        feature_frame[f"state__{ticker.lower()}__volatility_12m"] = (
            returns.rolling(12).std(ddof=1)
        )

    holdout = feature_frame.loc["2024-12-31":"2026-05-31"].copy()
    holdout["target_month"] = holdout.index + pd.offsets.MonthEnd(1)
    for ticker in PRIMARY_ASSETS:
        holdout[f"target_next__{ticker.lower()}"] = extended[ticker].reindex(
            holdout["target_month"]
        ).to_numpy()
    holdout["benchmark_next__gsg"] = extended["GSG"].reindex(
        holdout["target_month"]
    ).to_numpy()
    holdout["benchmark_next__dbc"] = extended["DBC"].reindex(
        holdout["target_month"]
    ).to_numpy()
    holdout["target_next__cash"] = extended["CASH"].reindex(
        holdout["target_month"]
    ).to_numpy()
    holdout["sample_split"] = "FROZEN_POLICY_TEMPORAL_EXTENSION"
    holdout["publication_status"] = np.where(
        holdout["target_month"] <= FINAL_PRIMARY_TARGET,
        "PRIMARY_ELIGIBLE",
        "PROVISIONAL_JUNE",
    )
    holdout.index.name = "state_month"
    holdout.to_parquet(OUT / "model_ready_post_freeze_holdout_2025_2026.parquet")
    holdout.to_csv(OUT / "model_ready_post_freeze_holdout_2025_2026.csv")

    # Recompute the frozen historical features to prove identical definitions.
    feature_errors = []
    for column in [column for column in feature_frame if column.startswith("state__")]:
        comparison_index = model_panel.index.intersection(feature_frame.index)
        error = (
            model_panel.loc[comparison_index, column]
            - feature_frame.loc[comparison_index, column]
        ).abs()
        feature_errors.append(
            {
                "feature": column,
                "maximum_absolute_error": float(error.max()),
                "status": "PASS" if float(error.max()) <= 1e-15 else "FAIL",
            }
        )
    pd.DataFrame(feature_errors).to_csv(
        OUT / "historical_feature_reproduction_audit.csv", index=False
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_extension_target_start": "2025-01-31",
        "primary_extension_target_end": "2026-05-31",
        "primary_extension_months": 17,
        "provisional_additional_target": "2026-06-30",
        "primary_assets": PRIMARY_ASSETS,
        "benchmarks": ["GSG", "DBC"],
        "permnos": PERMNOS,
        "asset_source_through_2025": "CRSP CIZ msf_v2.mthret",
        "asset_source_2026": (
            "Nasdaq daily closing-price returns; treated as a frozen-policy "
            "temporal extension and overlap-audited against CRSP during 2024-2025"
        ),
        "distribution_treatment_2026_H1": (
            "The four Invesco DB funds distribute annually in December. "
            "Nasdaq price returns match CRSP total returns outside the audited "
            "December distribution months; January-June 2026 contains no "
            "scheduled annual distribution month."
        ),
        "cash_source_2025_through_2026_05": "Kenneth French monthly RF",
        "cash_source_2026_06": (
            "FRED DGS1MO first June observation converted from annual "
            "investment-basis yield to a 30-day return; provisional only"
        ),
        "do_not_tune_on_holdout": True,
    }
    (OUT / "extension_metadata.json").write_text(json.dumps(metadata, indent=2))

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

    print("\nNASDAQ VS CRSP MONTHLY AUDIT")
    print(monthly_audit.to_string(index=False))
    print("\nFRENCH RF VS CRSP CASH AUDIT")
    print(
        pd.read_csv(OUT / "french_rf_vs_crsp_cash_audit_summary.csv").to_string(
            index=False
        )
    )
    print("\nHOLDOUT")
    print(
        holdout[
            ["target_month", "publication_status"]
            + [f"target_next__{ticker.lower()}" for ticker in PRIMARY_ASSETS]
            + ["benchmark_next__gsg", "benchmark_next__dbc", "target_next__cash"]
        ].to_string()
    )


if __name__ == "__main__":
    main()
