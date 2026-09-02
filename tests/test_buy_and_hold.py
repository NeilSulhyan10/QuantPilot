import numpy as np
import pandas as pd
import pytest

from src.backtesting.buy_and_hold import BuyAndHoldBacktester
from src.backtesting.results import BacktestResult


def make_returns():
    dates = pd.bdate_range(
        "2020-01-01",
        "2020-01-10",
    )

    return pd.DataFrame(
        {
            "AAPL": 0.01,
            "MSFT": 0.02,
            "NVDA": -0.01,
            "AMZN": 0.015,
        },
        index=dates,
    )


def test_buy_and_hold_returns_result():
    returns = make_returns()

    result = BuyAndHoldBacktester().run(returns)

    assert isinstance(result, BacktestResult)


def test_buy_and_hold_initial_weights_are_equal():
    returns = make_returns()

    result = BuyAndHoldBacktester().run(returns)

    expected_weight = 1.0 / len(returns.columns)

    assert np.allclose(
        result.weights.iloc[0].values,
        expected_weight,
    )


def test_buy_and_hold_has_initial_turnover():
    returns = make_returns()

    result = BuyAndHoldBacktester().run(returns)

    assert np.isclose(
        result.turnover.iloc[0],
        1.0,
    )


def test_buy_and_hold_has_only_initial_costs():
    returns = make_returns()

    result = BuyAndHoldBacktester().run(returns)

    assert len(result.transaction_costs) == 1
    assert len(result.slippage) == 1

    assert np.isclose(
        result.transaction_costs.iloc[0],
        0.001,
    )

    assert np.isclose(
        result.slippage.iloc[0],
        0.0005,
    )


def test_buy_and_hold_has_no_missing_returns():
    returns = make_returns()

    result = BuyAndHoldBacktester().run(returns)

    assert not result.returns.isna().any()


def test_buy_and_hold_preserves_return_length():
    returns = make_returns()

    result = BuyAndHoldBacktester().run(returns)

    assert len(result.returns) == len(returns)


def test_buy_and_hold_rejects_empty_returns():
    returns = pd.DataFrame()

    with pytest.raises(ValueError):
        BuyAndHoldBacktester().run(returns)

def test_buy_and_hold_allows_weights_to_drift():
    returns = pd.DataFrame(
        {
            "AAPL": [0.10, 0.10],
            "MSFT": [0.00, 0.00],
        },
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03"]
        ),
    )

    result = BuyAndHoldBacktester().run(returns)

    # After AAPL rises 10% on day 1, its portfolio
    # weight increases from 50% to 52.38%.
    #
    # Therefore day 2's portfolio return is:
    # 52.38% * 10% = 5.238%.
    assert result.returns.iloc[1] == pytest.approx(
        0.05238095238
    )