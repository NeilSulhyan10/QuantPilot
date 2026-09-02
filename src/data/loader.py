from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "universe.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load QuantPilot data configuration from YAML."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str | None = None,
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download historical market data for one ticker."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if data.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    return data


def save_stock_data(data: pd.DataFrame, ticker: str) -> Path:
    """Save raw stock data as a Parquet file."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DATA_DIR / f"{ticker}.parquet"
    data.to_parquet(output_path)

    return output_path


def download_universe() -> None:
    """Download and save data for every ticker in the configured universe."""
    config = load_config()

    tickers = config["universe"]["tickers"]
    data_config = config["data"]

    for ticker in tickers:
        print(f"Downloading {ticker}...")

        data = download_stock_data(
            ticker=ticker,
            start_date=data_config["start_date"],
            end_date=data_config["end_date"],
            interval=data_config["interval"],
            auto_adjust=data_config["auto_adjust"],
        )

        output_path = save_stock_data(data, ticker)

        print(f"Saved {len(data):,} rows → {output_path}")


if __name__ == "__main__":
    download_universe()

