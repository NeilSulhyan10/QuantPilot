import pandas as pd
import pytest

from src.portfolio.rebalance import (
    calculate_rebalance_turnover,
    simulate_rebalance_period,
)


def test_rebalance_turnover():
    current = pd.Series(
        [0.60, 0.40],
        index=["AAPL", "MSFT"],
    )

    target = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "MSFT"],
    )

    turnover = calculate_rebalance_turnover(
        current,
        target,
    )

    assert turnover == pytest.approx(0.10)


def test_rebalance_turnover_zero_when_already_at_target():
    weights = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "MSFT"],
    )

    assert calculate_rebalance_turnover(
        weights,
        weights,
    ) == pytest.approx(0.0)


def test_period_simulation_accounts_for_drift():
    starting_weights = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "MSFT"],
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.10],
            "MSFT": [0.00],
        },
        index=pd.to_datetime(["2025-01-02"]),
    )

    target_weights = pd.Series(
        [0.50, 0.50],
        index=["AAPL", "MSFT"],
    )

    ending_weights, turnover = simulate_rebalance_period(
        starting_weights,
        returns,
        target_weights,
    )

    assert ending_weights["AAPL"] == pytest.approx(
        0.5238095238
    )

    assert ending_weights["MSFT"] == pytest.approx(
        0.4761904762
    )

    assert turnover == pytest.approx(
        0.0238095238
    )