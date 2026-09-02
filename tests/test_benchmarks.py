import numpy as np
import pandas as pd
import pytest

from src.portfolio.benchmarks import equal_weight


def test_equal_weight_sums_to_one():
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]

    weights = equal_weight(tickers)

    assert np.isclose(weights.sum(), 1.0)


def test_equal_weight_each_asset_is_equal():
    tickers = ["AAPL", "MSFT", "NVDA", "AMZN"]

    weights = equal_weight(tickers)

    expected_weight = 1.0 / len(tickers)

    assert np.allclose(weights.values, expected_weight)


def test_equal_weight_preserves_ticker_order():
    tickers = ["NVDA", "AAPL", "MSFT"]

    weights = equal_weight(tickers)

    assert list(weights.index) == tickers


def test_equal_weight_rejects_empty_tickers():
    with pytest.raises(ValueError):
        equal_weight([])


def test_equal_weight_rejects_duplicate_tickers():
    tickers = ["AAPL", "MSFT", "AAPL"]

    with pytest.raises(ValueError):
        equal_weight(tickers)