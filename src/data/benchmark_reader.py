import pandas as pd

from src.data.market_data import MarketDataAdapter
from src.data.markets import Market


def load_benchmark(
    ticker: str,
    allow_download: bool = True,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load benchmark data.

    Retained for backward compatibility with existing QuantPilot code.
    """

    adapter = MarketDataAdapter(Market.US)

    return adapter.load_asset(
        ticker=ticker,
        allow_download=allow_download,
        start_date=start_date,
        end_date=end_date,
    )