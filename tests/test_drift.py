import pandas as pd
import pytest

from src.portfolio.drift import drift_weights


def test_weights_drift_after_returns():
    weights = pd.Series(
        [0.5, 0.5],
        index=["AAPL", "MSFT"],
    )

    returns = pd.Series(
        [0.10, 0.00],
        index=["AAPL", "MSFT"],
    )

    result = drift_weights(weights, returns)

    assert result["AAPL"] == pytest.approx(0.5238095238)
    assert result["MSFT"] == pytest.approx(0.4761904762)
    assert result.sum() == pytest.approx(1.0)


def test_equal_performance_keeps_weights():
    weights = pd.Series(
        [0.4, 0.6],
        index=["AAPL", "MSFT"],
    )

    returns = pd.Series(
        [0.05, 0.05],
        index=["AAPL", "MSFT"],
    )

    result = drift_weights(weights, returns)

    pd.testing.assert_series_equal(result, weights)


def test_mismatched_assets_raise():
    weights = pd.Series([0.5, 0.5], index=["AAPL", "MSFT"])
    returns = pd.Series([0.1, 0.1], index=["AAPL", "NVDA"])

    with pytest.raises(ValueError):
        drift_weights(weights, returns)