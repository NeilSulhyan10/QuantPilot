"""Goal portfolio construction for QuantPilot V2."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.goals.assets import (
    build_selected_return_matrix,
    validate_minimum_history,
)
from src.goals.feasibility import assess_goal_feasibility
from src.goals.optimizer import optimize_for_goal


@dataclass(frozen=True)
class GoalPortfolioResult:
    """Portfolio generated from an investor goal."""

    weights: pd.Series
    expected_return: float
    expected_volatility: float
    target_return: float
    risk_tolerance: str
    maximum_allowed_volatility: float
    feasible: bool


def build_goal_portfolio(
    asset_data: dict[str, pd.DataFrame],
    target_return: float,
    risk_tolerance: str,
    max_weight: float = 0.10,
    minimum_observations: int = 252,
    estimation_window: int = 60,
) -> GoalPortfolioResult:
    """Construct a portfolio satisfying a return/risk goal.

    Expected returns are estimated using the historical mean over
    the most recent estimation window. Covariance is estimated from
    the same window.

    The historical data is assumed to contain only information
    available before the portfolio decision date.
    """

    if estimation_window <= 0:
        raise ValueError(
            "estimation_window must be greater than 0."
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

    # Convert expected daily returns to effective annual returns.
    expected_returns = (
        (1.0 + daily_expected_returns) ** 252 - 1.0
    )

    covariance = estimation_returns.cov()

    feasibility = assess_goal_feasibility(
        target_return=target_return,
        expected_returns=expected_returns,
        covariance=covariance,
        risk_tolerance=risk_tolerance,
        max_weight=max_weight,
    )

    if not feasibility.feasible:
        raise ValueError(
            "Goal is infeasible. "
            f"Target return: {target_return:.2%}; "
            f"maximum feasible return: "
            f"{feasibility.maximum_feasible_return:.2%}."
        )

    optimization = optimize_for_goal(
        expected_returns=expected_returns,
        covariance=covariance,
        target_return=target_return,
        risk_tolerance=risk_tolerance,
        max_weight=max_weight,
    )

    return GoalPortfolioResult(
        weights=optimization.weights,
        expected_return=optimization.expected_return,
        expected_volatility=optimization.expected_volatility,
        target_return=optimization.target_return,
        risk_tolerance=optimization.risk_tolerance,
        maximum_allowed_volatility=(
            optimization.maximum_allowed_volatility
        ),
        feasible=True,
    )