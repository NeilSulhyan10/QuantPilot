import pytest
import numpy as np
import pandas as pd

from src.goals.planner import build_goal_plan
from src.data.markets import Market
from src.data.market_data import MarketDataAdapter
from src.goals.planner import (
    build_goal_plan,
    build_goal_plan_from_selection,
)

def _build_asset_data():
    dates = pd.bdate_range(
        "2020-01-01",
        periods=300,
    )

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "JPM",
    ]

    asset_data = {}

    for i, ticker in enumerate(tickers):
        daily_return = 0.0002 + i * 0.00002

        returns = np.full(
            len(dates) - 1,
            daily_return,
        )

        prices = [100.0]

        for ret in returns:
            prices.append(
                prices[-1] * (1.0 + ret)
            )

        asset_data[ticker] = pd.DataFrame(
            {
                "Open": prices,
                "High": np.array(prices) * 1.01,
                "Close": prices,
                "Volume": np.full(
                    len(prices),
                    1_000_000,
                ),
            },
            index=dates,
        )

    return asset_data


def test_build_goal_plan():
    asset_data = _build_asset_data()

    plan = build_goal_plan(
        asset_data=asset_data,
        target_amount=1_000_000,
        years=10,
        risk_tolerance="aggressive",
        max_weight=0.50,
        minimum_observations=252,
        estimation_window=60,
    )

    assert plan.target_amount == 1_000_000
    assert plan.years == 10
    assert plan.feasibility.feasible

    assert (
        plan.recommended_return
        <= plan.maximum_feasible_return
    )

    assert not plan.portfolio.weights.isna().any()

    assert (
        plan.portfolio.weights.sum()
        == pytest.approx(1.0, abs=1e-5)
    )

    assert (
        plan.scenarios.conservative.annual_return
        <= plan.scenarios.expected.annual_return
        <= plan.scenarios.optimistic.annual_return
    )

def test_build_goal_plan_from_selection_uses_market_adapter():
    asset_data = _build_asset_data()

    class FakeAdapter:
        def load_assets(
            self,
            tickers,
            allow_download=True,
            start_date=None,
            end_date=None,
        ):
            assert tuple(tickers) == (
                "AAPL",
                "MSFT",
                "NVDA",
                "JPM",
            )

            return asset_data

    plan = build_goal_plan_from_selection(
        tickers=[
            "AAPL",
            "MSFT",
            "NVDA",
            "JPM",
        ],
        market=Market.US,
        target_amount=1_000_000,
        years=10,
        risk_tolerance="aggressive",
        max_weight=0.50,
        minimum_observations=252,
        estimation_window=60,
        allow_download=False,
        data_adapter=FakeAdapter(),
    )

    assert plan.target_amount == 1_000_000
    assert plan.years == 10
    assert plan.feasibility.feasible
    assert not plan.portfolio.weights.isna().any()
    assert plan.portfolio.weights.sum() == pytest.approx(
        1.0,
        abs=1e-5,
    )