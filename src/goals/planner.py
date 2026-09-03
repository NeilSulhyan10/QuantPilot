from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.goals.feasibility import (
    FeasibilityResult,
    assess_goal_feasibility,
    calculate_maximum_feasible_return,
    calculate_recommended_target_return,
)
from src.goals.portfolio import GoalPortfolioResult, build_goal_portfolio
from src.goals.scenarios import ScenarioSet, build_scenario_set


@dataclass(frozen=True)
class GoalPlan:
    """Complete quantitative plan for an investor goal."""

    target_amount: float
    years: float
    risk_tolerance: str

    maximum_feasible_return: float
    recommended_return: float

    feasibility: FeasibilityResult
    portfolio: GoalPortfolioResult
    scenarios: ScenarioSet


def build_goal_plan(
    asset_data: dict[str, pd.DataFrame],
    target_amount: float,
    years: float,
    risk_tolerance: str,
    max_weight: float = 0.10,
    minimum_observations: int = 252,
    estimation_window: int = 60,
) -> GoalPlan:
    """
    Build a complete quantitative goal plan.

    The planner:
    1. estimates the maximum feasible return,
    2. derives a risk-profile-specific recommended return,
    3. constructs the recommended portfolio,
    4. builds conservative/expected/optimistic scenarios.

    Investment amounts are calculated independently for:
    - an initial lump-sum investment
    - a monthly contribution strategy
    """

    if target_amount <= 0:
        raise ValueError(
            "target_amount must be greater than 0."
        )

    if years <= 0:
        raise ValueError(
            "years must be greater than 0."
        )

    # Build a temporary return matrix to estimate the inputs
    # required by the feasibility and portfolio engines.
    from src.goals.assets import (
        build_selected_return_matrix,
        validate_minimum_history,
    )

    returns = build_selected_return_matrix(asset_data)

    validate_minimum_history(
        returns,
        minimum_observations=minimum_observations,
    )

    if len(returns) < estimation_window:
        raise ValueError(
            "Insufficient observations for the estimation window."
        )

    estimation_returns = returns.tail(estimation_window)

    daily_expected_returns = estimation_returns.mean()

    expected_returns = (
        (1.0 + daily_expected_returns) ** 252 - 1.0
    )

    covariance = estimation_returns.cov()

    maximum_feasible_return = calculate_maximum_feasible_return(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_tolerance=risk_tolerance,
        max_weight=max_weight,
    )

    recommended_return = calculate_recommended_target_return(
        maximum_feasible_return=maximum_feasible_return,
        risk_tolerance=risk_tolerance,
    )

    feasibility = assess_goal_feasibility(
        target_return=recommended_return,
        expected_returns=expected_returns,
        covariance=covariance,
        risk_tolerance=risk_tolerance,
        max_weight=max_weight,
    )

    portfolio = build_goal_portfolio(
        asset_data=asset_data,
        target_return=recommended_return,
        risk_tolerance=risk_tolerance,
        max_weight=max_weight,
        minimum_observations=minimum_observations,
        estimation_window=estimation_window,
    )

    # Scenario spread is based on the portfolio's estimated
    # annual volatility. This is a planning convention, not
    # a confidence interval or guaranteed outcome.
    conservative_return = max(
        -0.99,
        portfolio.expected_return - portfolio.expected_volatility,
    )

    expected_return = portfolio.expected_return

    optimistic_return = (
        portfolio.expected_return
        + portfolio.expected_volatility
    )

    scenarios = build_scenario_set(
        target_amount=target_amount,
        years=years,
        conservative_return=conservative_return,
        expected_return=expected_return,
        optimistic_return=optimistic_return,
    )

    return GoalPlan(
        target_amount=float(target_amount),
        years=float(years),
        risk_tolerance=portfolio.risk_tolerance,
        maximum_feasible_return=float(
            maximum_feasible_return
        ),
        recommended_return=float(
            recommended_return
        ),
        feasibility=feasibility,
        portfolio=portfolio,
        scenarios=scenarios,
    )
