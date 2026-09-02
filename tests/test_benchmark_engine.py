import numpy as np
import pandas as pd

from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.config import BacktestConfig
from src.backtesting.results import BacktestResult


def make_returns():
    dates = pd.bdate_range(
        "2020-01-01",
        "2020-06-30",
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


def test_equal_weight_backtester_returns_result():
    returns = make_returns()

    config = BacktestConfig()

    backtester = EqualWeightBacktester(config)

    result = backtester.run(returns)

    assert isinstance(result, BacktestResult)


def test_equal_weight_weights_sum_to_one():
    returns = make_returns()

    result = EqualWeightBacktester(
        BacktestConfig()
    ).run(returns)

    weight_sums = result.weights.sum(axis=1)

    assert np.allclose(weight_sums.values, 1.0)


def test_equal_weight_weights_are_equal():
    returns = make_returns()

    result = EqualWeightBacktester(
        BacktestConfig()
    ).run(returns)

    expected_weight = 1.0 / len(returns.columns)

    assert np.allclose(
        result.weights.values,
        expected_weight,
    )


def test_equal_weight_is_long_only():
    returns = make_returns()

    result = EqualWeightBacktester(
        BacktestConfig()
    ).run(returns)

    assert (result.weights >= 0).all().all()


def test_equal_weight_returns_have_no_missing_values():
    returns = make_returns()

    result = EqualWeightBacktester(
        BacktestConfig()
    ).run(returns)

    assert not result.returns.isna().any()


def test_equal_weight_has_initial_turnover():
    returns = make_returns()

    result = EqualWeightBacktester(
        BacktestConfig()
    ).run(returns)

    assert np.isclose(
        result.turnover.iloc[0],
        1.0,
    )


def test_equal_weight_applies_transaction_costs():
    returns = make_returns()

    result = EqualWeightBacktester(
        BacktestConfig()
    ).run(returns)

    assert result.transaction_costs.iloc[0] > 0
    assert result.slippage.iloc[0] > 0