import pandas as pd
import numpy as np

from src.backtesting.config import BacktestConfig
from src.backtesting.engine import WalkForwardBacktester


def test_daily_returns_use_drifting_weights():

    dates = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="D",
    )

    returns = pd.DataFrame(
        {
            "A": [0.10, 0.10, 0.00, 0.00],
            "B": [0.00, 0.00, 0.10, 0.10],
        },
        index=dates,
    )

    rebalance_dates = pd.DatetimeIndex(
        [dates[0], dates[3]]
    )

    target_weights = {
        dates[0]: pd.Series(
            {"A": 0.5, "B": 0.5}
        ),
        dates[3]: pd.Series(
            {"A": 0.5, "B": 0.5}
        ),
    }

    backtester = WalkForwardBacktester(
        config=BacktestConfig()
    )

    result = backtester._calculate_daily_returns(
        returns=returns,
        rebalance_dates=rebalance_dates,
        target_weights=target_weights,
    )

    # First day after rebalance:
    # 0.5 * 10% + 0.5 * 0% = 5%
    assert np.isclose(
        result.loc[dates[1]],
        0.05,
    )

    # After the first day's return,
    # A's weight increases relative to B.
    expected_second_day = (
        (0.50 * 1.00) / 1.05
    ) * 0.10

    assert np.isclose(
        result.loc[dates[2]],
        expected_second_day,
    )


def test_daily_returns_exclude_rebalance_dates():

    dates = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="D",
    )

    returns = pd.DataFrame(
        {
            "A": [0.01] * 5,
            "B": [0.01] * 5,
        },
        index=dates,
    )

    rebalance_dates = pd.DatetimeIndex(
        [dates[0], dates[2], dates[4]]
    )

    target_weights = {
        date: pd.Series(
            {"A": 0.5, "B": 0.5}
        )
        for date in rebalance_dates
    }

    backtester = WalkForwardBacktester(
        config=BacktestConfig()
    )

    result = backtester._calculate_daily_returns(
        returns=returns,
        rebalance_dates=rebalance_dates,
        target_weights=target_weights,
    )

    assert dates[0] not in result.index
    assert dates[2] not in result.index
    assert dates[4] not in result.index