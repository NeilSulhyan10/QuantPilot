import numpy as np
import pytest

from src.backtesting.slippage import calculate_slippage


def test_slippage():
    slippage = calculate_slippage(
        turnover=0.05,
        slippage_rate=0.0005,
    )

    assert np.isclose(slippage, 0.000025)


def test_zero_turnover():
    slippage = calculate_slippage(
        turnover=0.0,
        slippage_rate=0.0005,
    )

    assert slippage == 0.0


def test_negative_turnover_raises_error():
    with pytest.raises(ValueError):
        calculate_slippage(
            turnover=-0.05,
            slippage_rate=0.0005,
        )


def test_negative_slippage_rate_raises_error():
    with pytest.raises(ValueError):
        calculate_slippage(
            turnover=0.05,
            slippage_rate=-0.0005,
        )