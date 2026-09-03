from pathlib import Path

import pandas as pd
import yfinance as yf

from src.data.validation import validate_data, normalize_columns


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# Maximum fraction of downloaded rows that may be rejected as invalid.
# A small number of malformed provider rows should not make the entire
# Goal Planner fail, but a seriously corrupted dataset must still fail.
MAX_INVALID_ROW_FRACTION = 0.01


def _remove_invalid_ohlc_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Remove rows that violate basic OHLC relationships.

    Prices are never modified. Only invalid rows are removed.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned data and number of removed rows.
    """
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
    cleaned = data.loc[valid].copy()

    return cleaned, invalid_count


def _download_and_prepare(
    ticker: str,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download, normalize, conservatively clean, and validate market data.

    This is used when a processed local Parquet cache is unavailable.
    """
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
        raise ValueError(f"No data returned for ticker: {ticker}")

    # Normalize yfinance MultiIndex output before further processing.
    data = normalize_columns(data)

    # Ensure required columns exist before row-level cleaning.
    required_columns = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Downloaded data for {ticker} is missing required columns: "
            f"{sorted(missing)}"
        )

    data = data[required_columns].copy()

    original_rows = len(data)

    # Remove malformed provider rows without altering any price values.
    data, invalid_count = _remove_invalid_ohlc_rows(data)

    if invalid_count:
        invalid_fraction = invalid_count / original_rows

        if invalid_fraction > MAX_INVALID_ROW_FRACTION:
            raise ValueError(
                f"Downloaded data for {ticker} contains too many invalid "
                f"OHLC rows: {invalid_count:,}/{original_rows:,} "
                f"({invalid_fraction:.2%})."
            )

    if data.empty:
        raise ValueError(
            f"All downloaded rows for {ticker} failed OHLC validation."
        )

    # Re-validate the cleaned data using QuantPilot's strict validator.
    data = validate_data(data)

    return data[required_columns].sort_index()


def load_stock(
    ticker: str,
    allow_download: bool = True,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load historical data for one stock.

    Existing processed Parquet data is preferred. If it is unavailable and
    allow_download=True, historical data is downloaded from Yahoo Finance
    and validated in memory.

    Parameters
    ----------
    ticker : str
        Stock ticker, e.g. "AAPL".
    allow_download : bool
        Whether missing processed data may be downloaded.
    start_date : str
        Download start date.
    end_date : str | None
        Download end date.
    """
    ticker = ticker.upper().strip()

    if not ticker:
        raise ValueError("Ticker must not be empty.")

    path = PROCESSED_DATA_DIR / f"{ticker}.parquet"

    if path.exists():
        return pd.read_parquet(path)

    if not allow_download:
        raise FileNotFoundError(
            f"No processed data found for ticker: {ticker}"
        )

    return _download_and_prepare(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )


def load_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load historical data for multiple stocks.

    Existing processed files are used when available. Missing files are
    downloaded and validated automatically.
    """
    return {
        ticker.upper().strip(): load_stock(ticker)
        for ticker in tickers
    }
