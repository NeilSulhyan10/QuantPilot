import numpy as np
import pandas as pd

from src.data.market_data import MarketDataAdapter
from src.data.markets import Market
from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance
from src.risk.covariance import is_positive_semidefinite


TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "GOOGL",
    "AMZN",
    "META",
    "JPM",
    "V",
    "MA",
    "JNJ",
    "UNH",
    "XOM",
    "CVX",
    "PG",
    "KO",
    "COST",
    "CAT",
    "WMT",
    "HD",
]


def make_stock_data(
    periods: int = 120,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create deterministic synthetic OHLCV data for testing.

    The data is intentionally simple but valid according to
    QuantPilot's OHLCV validation rules.
    """
    rng = np.random.default_rng(seed)

    index = pd.date_range(
        "2020-01-01",
        periods=periods,
        freq="D",
    )

    prices = 100.0 * np.exp(
        np.cumsum(
            rng.normal(
                loc=0.0005,
                scale=0.01,
                size=periods,
            )
        )
    )

    close = prices
    open_price = close * (
        1.0 + rng.normal(0.0, 0.002, periods)
    )

    high = np.maximum(
        open_price,
        close,
    ) * (
        1.0 + rng.uniform(0.001, 0.01, periods)
    )

    low = np.minimum(
        open_price,
        close,
    ) * (
        1.0 - rng.uniform(0.001, 0.01, periods)
    )

    volume = rng.integers(
        1_000_000,
        2_000_000,
        periods,
    )

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def test_covariance_matrix_is_psd(tmp_path):
    """
    Rolling covariance matrices should be positive
    semidefinite.

    The test uses temporary synthetic Parquet data so it
    does not depend on local cached market data or Yahoo
    Finance.
    """
    adapter = MarketDataAdapter(
        market=Market.US,
        data_dir=tmp_path,
    )

    for i, ticker in enumerate(TICKERS):
        data = make_stock_data(
            periods=120,
            seed=42 + i,
        )

        data.to_parquet(
            tmp_path / f"{ticker}.parquet"
        )

    stocks = adapter.load_assets(
        TICKERS,
        allow_download=False,
    )

    returns = calculate_return_matrix(stocks)

    covariances = rolling_covariance(
        returns,
        window=60,
    )

    assert len(covariances) > 0

    for date, covariance in covariances.items():
        matrix = covariance.to_numpy()

        assert is_positive_semidefinite(matrix), (
            f"Covariance matrix is not PSD on {date}"
        )


def test_known_psd_matrix():
    covariance = np.array([
        [1.0, 0.5],
        [0.5, 1.0],
    ])

    assert is_positive_semidefinite(covariance)


def test_non_psd_matrix():
    covariance = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])

    assert not is_positive_semidefinite(covariance)