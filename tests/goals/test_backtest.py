import numpy as np
import pandas as pd
import pytest

from src.backtesting.config import BacktestConfig
from src.goals.backtest import backtest_goal_portfolio
from src.goals.portfolio import GoalPortfolioResult
from src.goals.risk_profile import RiskTolerance


@pytest.fixture
def returns():
    dates = pd.bdate_range("2024-01-01", periods=80)

    return pd.DataFrame(
        {
            "AAPL": np.full(80, 0.0005),
            "MSFT": np.full(80, 0.0004),
            "NVDA": np.full(80, 0.0006),
            "JPM": np.full(80, 0.0003),
            "GOOGL": np.full(80, 0.0004),
            "AMZN": np.full(80, 0.0005),
            "META": np.full(80, 0.0005),
            "V": np.full(80, 0.0003),
            "MA": np.full(80, 0.0003),
            "AVGO": np.full(80, 0.0006),
        },
        index=dates,
    )


@pytest.fixture
def portfolio():
    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "JPM",
        "GOOGL",
        "AMZN",
        "META",
        "V",
        "MA",
        "AVGO",
    ]

    weights = pd.Series(
        0.10,
        index=tickers,
        dtype=float,
    )

    return GoalPortfolioResult(
        weights=weights,
        expected_return=0.15,
        expected_volatility=0.18,
        target_return=0.12,
        risk_tolerance=RiskTolerance.MODERATE,
        maximum_allowed_volatility=0.18,
        feasible=True,
    )


def test_goal_portfolio_backtest_runs(returns, portfolio):
    result = backtest_goal_portfolio(
        returns,
        portfolio,
        BacktestConfig(
            estimation_window=20,
        ),
    )

    assert not result.returns.empty
    assert not result.returns.isna().any()
    assert len(result.returns) > 0


def test_goal_portfolio_backtest_has_weights(returns, portfolio):
    result = backtest_goal_portfolio(
        returns,
        portfolio,
        BacktestConfig(
            estimation_window=20,
        ),
    )

    assert not result.weights.empty
    assert set(result.weights.columns) == set(portfolio.weights.index)


def test_goal_portfolio_backtest_has_costs(returns, portfolio):
    result = backtest_goal_portfolio(
        returns,
        portfolio,
        BacktestConfig(
            estimation_window=20,
        ),
    )

    assert not result.transaction_costs.empty
    assert not result.slippage.empty


def test_missing_asset_is_rejected(returns, portfolio):
    bad_weights = portfolio.weights.copy()
    bad_weights["FAKE"] = 0.01

    bad_portfolio = GoalPortfolioResult(
        weights=bad_weights,
        expected_return=portfolio.expected_return,
        expected_volatility=portfolio.expected_volatility,
        target_return=portfolio.target_return,
        risk_tolerance=portfolio.risk_tolerance,
        maximum_allowed_volatility=portfolio.maximum_allowed_volatility,
        feasible=True,
    )

    with pytest.raises(ValueError, match="missing from returns"):
        backtest_goal_portfolio(returns, bad_portfolio)


def test_nan_returns_are_rejected(returns, portfolio):
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        backtest_goal_portfolio(broken, portfolio)