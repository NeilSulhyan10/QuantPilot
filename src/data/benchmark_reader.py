from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.validation import validate_data, normalize_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MAX_INVALID_ROW_FRACTION = 0.01


def _remove_invalid_ohlc_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove malformed OHLC rows without modifying price values."""
    data = data.copy()

    price_columns = ["Open", "High", "Low", "Close"]

    valid = (
        data[price_columns].notna().all(axis=1)
        & (data[price_columns] > 0).all(axis=1)
        & (data["High"] >= data["Low"])
        & (data["Open"] >= data["Low"])
        & (data["Open"] <= data["High"])
        & (data["Close"] >= data["Low"])
        & (data["Close"] <= data["High"])
        & (data["Volume"] >= 0)
    )

    invalid_count = int((~valid).sum())

    return data.loc[valid].copy(), invalid_count


def _download_and_prepare(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download, clean, and strictly validate benchmark data."""
    ticker = ticker.upper().strip()

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for benchmark: {ticker}")

    data = normalize_columns(data)

    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Downloaded benchmark data for {ticker} is missing required "
            f"columns: {sorted(missing)}"
        )

    data = data[required_columns].copy()

    original_rows = len(data)

    data, invalid_count = _remove_invalid_ohlc_rows(data)

    if invalid_count:
        invalid_fraction = invalid_count / original_rows

        if invalid_fraction > MAX_INVALID_ROW_FRACTION:
            raise ValueError(
                f"Downloaded benchmark data for {ticker} contains too many "
                f"invalid OHLC rows: {invalid_count:,}/{original_rows:,} "
                f"({invalid_fraction:.2%})."
            )

    if data.empty:
        raise ValueError(
            f"All downloaded rows for benchmark {ticker} failed validation."
        )

    data = validate_data(data)

    return data[required_columns].sort_index()


def load_benchmark(
    ticker: str,
    allow_download: bool = True,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load benchmark data from the local processed cache.

    If the cache is unavailable and allow_download=True, download and validate
    the benchmark data from Yahoo Finance. This allows the Goal Planner to
    operate in deployments where the local data directory is absent.
    """
    ticker = ticker.upper().strip()

    if not ticker:
        raise ValueError("Benchmark ticker must not be empty.")

    path = PROCESSED_DATA_DIR / f"{ticker}.parquet"

    if path.exists():
        return pd.read_parquet(path)

    if not allow_download:
        raise FileNotFoundError(
            f"No processed benchmark data found for ticker: {ticker}"
        )

    return _download_and_prepare(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )
