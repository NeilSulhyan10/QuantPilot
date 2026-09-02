from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_stock(ticker: str) -> pd.DataFrame:
    """
    Load processed historical data for a single stock.

    Parameters
    ----------
    ticker : str
        Stock ticker, e.g. "AAPL".

    Returns
    -------
    pd.DataFrame
        Processed OHLCV data indexed by date.
    """
    ticker = ticker.upper()

    path = PROCESSED_DATA_DIR / f"{ticker}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"No processed data found for ticker: {ticker}"
        )

    return pd.read_parquet(path)


def load_universe(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load processed data for multiple stocks.

    Parameters
    ----------
    tickers : list[str]
        List of stock tickers.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping from ticker to its processed DataFrame.
    """
    return {
        ticker.upper(): load_stock(ticker)
        for ticker in tickers
    }

