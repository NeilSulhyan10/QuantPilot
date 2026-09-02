from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_benchmark(ticker: str) -> pd.DataFrame:
    ticker = ticker.upper()

    path = PROCESSED_DATA_DIR / f"{ticker}.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"No processed benchmark data found for ticker: {ticker}"
        )

    return pd.read_parquet(path)