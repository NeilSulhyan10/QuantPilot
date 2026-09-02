from pathlib import Path

import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_benchmark(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
) -> None:
    """Download and save benchmark OHLCV data."""

    ticker = ticker.upper()

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data downloaded for benchmark: {ticker}"
        )

    RAW_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = RAW_DATA_DIR / f"{ticker}.parquet"

    data.to_parquet(output_path)