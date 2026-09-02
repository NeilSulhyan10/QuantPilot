import numpy as np
import pandas as pd
import pytest

from src.portfolio.weights import (
    calculate_weight_changes,
    calculate_turnover,
)


def test_weight_changes():
    previous = pd.Series(
        [0.10, 0.05, 0.05],
        index=["AAPL", "MSFT", "NVDA"],
    )

    target = pd.Series(
        [0.10, 0.00, 0.10],
        index=["AAPL", "MSFT", "NVDA"],
    )

    changes = calculate_weight_changes(previous, target)

    expected = pd.Series(
        [0.00, -0.05, 0.05],
        index=["AAPL", "MSFT", "NVDA"],
    )

    pd.testing.assert_series_equal(changes, expected)


def test_turnover():
    previous = pd.Series(
        [0.10, 0.05, 0.05],
        index=["AAPL", "MSFT", "NVDA"],
    )

    target = pd.Series(
        [0.10, 0.00, 0.10],
        index=["AAPL", "MSFT", "NVDA"],
    )

    turnover = calculate_turnover(previous, target)

    assert np.isclose(turnover, 0.05)


def test_mismatched_tickers_raise_error():
    previous = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "MSFT"],
    )

    target = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "NVDA"],
    )

    with pytest.raises(ValueError):
        calculate_weight_changes(previous, target)


def test_missing_weights_raise_error():
    previous = pd.Series(
        [0.50, np.nan],
        index=["AAPL", "MSFT"],
    )

    target = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "MSFT"],
    )

    with pytest.raises(ValueError):
        calculate_turnover(previous, target)