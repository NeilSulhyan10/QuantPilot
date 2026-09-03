import numpy as np
import pandas as pd
import pytest

from src.goals.portfolio import build_goal_portfolio


@pytest.fixture
def asset_data():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        "2024-01-01",
        periods=400,
    )

    data = {}

    for ticker, drift, volatility in [
        ("AAPL", 0.00035, 0.010),
        ("MSFT", 0.00045, 0.009),
        ("NVDA", 0.00070, 0.018),
        ("JPM", 0.00040, 0.011),
    ]:
        returns = (
            drift
            + volatility * rng.standard_normal(len(dates))
        )

        prices = 100 * np.cumprod(1 + returns)

        data[ticker] = pd.DataFrame(
            {
                "Close": prices,
            },
            index=dates,
        )

    return data


def test_goal_portfolio_is_created(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.05,
        risk_tolerance="aggressive",
        max_weight=0.50,
        minimum_observations=252,
        estimation_window=60,
    )

    assert result.feasible is True
    assert not result.weights.empty
    assert result.weights.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_goal_portfolio_respects_long_only(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.05,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.weights.min() >= -1e-6


def test_goal_portfolio_respects_position_cap(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.05,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.weights.max() <= 0.50 + 1e-6


def test_goal_portfolio_hits_target(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.05,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.expected_return >= (
        result.target_return - 1e-6
    )


def test_goal_portfolio_respects_risk_limit(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.05,
        risk_tolerance="aggressive",
        max_weight=0.50,
    )

    assert result.expected_volatility <= (
        result.maximum_allowed_volatility + 1e-6
    )


def test_insufficient_history_is_rejected(asset_data):
    with pytest.raises(ValueError, match="Insufficient"):
        build_goal_portfolio(
            asset_data=asset_data,
            target_return=0.05,
            risk_tolerance="aggressive",
            max_weight=0.50,
            minimum_observations=500,
        )


def test_insufficient_estimation_window_is_rejected(
    asset_data,
):
    with pytest.raises(
        ValueError,
        match="estimation window",
    ):
        build_goal_portfolio(
            asset_data=asset_data,
            target_return=0.05,
            risk_tolerance="aggressive",
            max_weight=0.50,
            estimation_window=500,
        )