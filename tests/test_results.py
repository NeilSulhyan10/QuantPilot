import pandas as pd

from src.backtesting.results import BacktestResult


def test_backtest_result():
    dates = pd.DatetimeIndex([
        "2024-01-02",
        "2024-01-03",
    ])

    returns = pd.Series(
        [0.01, -0.005],
        index=dates,
        name="return",
    )

    weights = pd.DataFrame(
        {
            "AAPL": [0.50, 0.50],
            "MSFT": [0.50, 0.50],
        },
        index=dates,
    )

    turnover = pd.Series(
        [0.0, 0.05],
        index=dates,
        name="turnover",
    )

    transaction_costs = pd.Series(
        [0.0, 0.00005],
        index=dates,
        name="transaction_cost",
    )

    slippage = pd.Series(
        [0.0, 0.000025],
        index=dates,
        name="slippage",
    )

    result = BacktestResult(
        returns=returns,
        weights=weights,
        turnover=turnover,
        transaction_costs=transaction_costs,
        slippage=slippage,
    )

    assert len(result.returns) == 2
    assert result.weights.shape == (2, 2)
    assert len(result.turnover) == 2
    assert len(result.transaction_costs) == 2
    assert len(result.slippage) == 2
