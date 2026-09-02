import numpy as np
import pandas as pd
import pytest

from src.backtesting.returns import (
    calculate_net_portfolio_return,
    calculate_portfolio_return,
)


def test_portfolio_return():
    asset_returns = pd.Series(
        [0.01, 0.02, -0.01],
        index=["AAPL", "MSFT", "NVDA"],
    )

    weights = pd.Series(
        [0.50, 0.30, 0.20],
        index=["AAPL", "MSFT", "NVDA"],
    )

    result = calculate_portfolio_return(
        asset_returns,
        weights,
    )

    expected = (
        0.50 * 0.01
        + 0.30 * 0.02
        + 0.20 * -0.01
    )

    assert np.isclose(result, expected)


def test_net_portfolio_return():
    result = calculate_net_portfolio_return(
        gross_return=0.01,
        transaction_cost=0.0001,
        slippage=0.00005,
    )

    assert np.isclose(result, 0.00985)


def test_mismatched_indices_raise_error():
    asset_returns = pd.Series(
        [0.01, 0.02],
        index=["AAPL", "MSFT"],
    )

    weights = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "NVDA"],
    )

    with pytest.raises(ValueError):
        calculate_portfolio_return(
            asset_returns,
            weights,
        )


def test_missing_returns_raise_error():
    asset_returns = pd.Series(
        [0.01, np.nan],
        index=["AAPL", "MSFT"],
    )

    weights = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "MSFT"],
    )

    with pytest.raises(ValueError):
        calculate_portfolio_return(
            asset_returns,
            weights,
        )


def test_negative_cost_raises_error():
    with pytest.raises(ValueError):
        calculate_net_portfolio_return(
            gross_return=0.01,
            transaction_cost=-0.001,
        )
