import pandas as pd

from src.data.market_data import MarketDataAdapter
from src.data.markets import Market


def load_stock(
    ticker: str,
    allow_download: bool = True,
    start_date: str = "2015-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load historical data for one US stock.

    This function is retained for backward compatibility.
    """

    adapter = MarketDataAdapter(Market.US)

    return adapter.load_asset(
        ticker=ticker,
        allow_download=allow_download,
        start_date=start_date,
        end_date=end_date,
    )


def load_universe(
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Load historical data for multiple US stocks.

    This function is retained for backward compatibility.
    """

    adapter = MarketDataAdapter(Market.US)

    return adapter.load_assets(
        tickers=tickers,
        allow_download=True,
    )