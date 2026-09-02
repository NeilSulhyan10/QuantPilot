import numpy as np
import pandas as pd

from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance
from src.risk.covariance import is_positive_semidefinite


TICKERS = [
    "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL",
    "AMZN", "META", "JPM", "V", "MA",
    "JNJ", "UNH", "XOM", "CVX", "PG",
    "KO", "COST", "CAT", "WMT", "HD",
]


def test_covariance_matrix_is_psd():
    stocks = load_universe(TICKERS)
    returns = calculate_return_matrix(stocks)

    covariances = rolling_covariance(
        returns,
        window=60,
    )

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