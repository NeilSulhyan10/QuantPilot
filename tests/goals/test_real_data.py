import pandas as pd
import pytest

from src.config import load_config
from src.goals.assets import (
    build_selected_return_matrix,
    load_selected_assets,
    validate_minimum_history,
)
from src.goals.portfolio import build_goal_portfolio


SELECTED_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "GOOGL",
    "AMZN",
    "META",
    "JPM",
    "V",
    "MA",
)


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def asset_data(config):
    return load_selected_assets(
        SELECTED_TICKERS,
        config.universe.tickers,
    )


@pytest.fixture(scope="module")
def returns(asset_data):
    return build_selected_return_matrix(
        asset_data,
        price_column="Close",
        method="simple",
    )


def test_real_assets_are_loaded(asset_data):
    assert set(asset_data) == set(SELECTED_TICKERS)

    for ticker, data in asset_data.items():
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert "Close" in data.columns
        assert data.index.is_monotonic_increasing


def test_real_return_matrix_is_valid(returns):
    assert list(returns.columns) == list(SELECTED_TICKERS)
    assert not returns.empty
    assert returns.index.is_monotonic_increasing
    assert not returns.isna().any().any()


def test_real_data_has_sufficient_history(returns):
    validate_minimum_history(
        returns,
        minimum_observations=252,
    )


def test_real_goal_portfolio_can_be_constructed(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.08,
        risk_tolerance="aggressive",
        max_weight=0.10,
        minimum_observations=252,
        estimation_window=60,
    )

    assert result.feasible is True
    assert not result.weights.empty

    assert result.weights.index.tolist() == list(
        SELECTED_TICKERS
    )

    assert result.weights.sum() == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_real_goal_portfolio_is_long_only(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.08,
        risk_tolerance="aggressive",
        max_weight=0.10,
    )

    assert result.weights.min() >= -1e-6


def test_real_goal_portfolio_respects_position_cap(
    asset_data,
):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.08,
        risk_tolerance="aggressive",
        max_weight=0.10,
    )

    assert result.weights.max() <= 0.10 + 1e-6


def test_real_goal_portfolio_hits_target(asset_data):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.08,
        risk_tolerance="aggressive",
        max_weight=0.10,
    )

    assert result.expected_return >= (
        result.target_return - 1e-6
    )


def test_real_goal_portfolio_respects_risk_limit(
    asset_data,
):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.08,
        risk_tolerance="aggressive",
        max_weight=0.10,
    )

    assert result.expected_volatility <= (
        result.maximum_allowed_volatility + 1e-6
    )


def test_real_goal_portfolio_contains_no_nan_weights(
    asset_data,
):
    result = build_goal_portfolio(
        asset_data=asset_data,
        target_return=0.08,
        risk_tolerance="aggressive",
        max_weight=0.10,
    )

    assert not result.weights.isna().any()