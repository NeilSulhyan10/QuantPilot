import numpy as np
import pytest

from src.backtesting.costs import calculate_transaction_cost


def test_transaction_cost():
    cost = calculate_transaction_cost(
        turnover=0.05,
        cost_rate=0.001,
    )

    assert np.isclose(cost, 0.00005)


def test_zero_turnover():
    cost = calculate_transaction_cost(
        turnover=0.0,
        cost_rate=0.001,
    )

    assert cost == 0.0


def test_negative_turnover_raises_error():
    with pytest.raises(ValueError):
        calculate_transaction_cost(
            turnover=-0.05,
            cost_rate=0.001,
        )


def test_negative_cost_rate_raises_error():
    with pytest.raises(ValueError):
        calculate_transaction_cost(
            turnover=0.05,
            cost_rate=-0.001,
        )