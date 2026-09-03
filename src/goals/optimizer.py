"""Risk-constrained goal-aware portfolio optimization for QuantPilot V2."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from src.goals.risk_profile import (
    RiskProfile,
    get_risk_profile,
)


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class GoalOptimizationResult:
    """Result produced by the V2 goal-aware optimizer."""

    weights: pd.Series
    expected_return: float
    expected_volatility: float
    target_return: float
    maximum_allowed_volatility: float
    risk_tolerance: str
    status: str


def _validate_inputs(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    max_weight: float,
) -> None:
    """Validate optimization inputs."""

    if expected_returns.empty:
        raise ValueError("expected_returns cannot be empty.")

    if expected_returns.isna().any():
        raise ValueError("expected_returns contains NaN values.")

    if covariance.empty:
        raise ValueError("covariance cannot be empty.")

    if covariance.isna().any().any():
        raise ValueError("covariance contains NaN values.")

    if list(expected_returns.index) != list(covariance.index):
        raise ValueError(
            "expected_returns and covariance indices must match."
        )

    if list(covariance.index) != list(covariance.columns):
        raise ValueError(
            "covariance must have matching row and column labels."
        )

    if not np.isfinite(expected_returns.to_numpy()).all():
        raise ValueError("expected_returns contains non-finite values.")

    if not np.isfinite(covariance.to_numpy()).all():
        raise ValueError("covariance contains non-finite values.")

    if target_return <= -1:
        raise ValueError(
            "target_return must be greater than -100%."
        )

    if not 0 < max_weight <= 1:
        raise ValueError(
            "max_weight must be greater than 0 and at most 1."
        )

    n_assets = len(expected_returns)

    if max_weight * n_assets < 1.0 - 1e-12:
        raise ValueError(
            "max_weight is too restrictive for the number of assets."
        )


def optimize_for_goal(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    risk_tolerance: RiskProfile | str,
    max_weight: float = 0.10,
) -> GoalOptimizationResult:
    """Find the minimum-variance portfolio satisfying the goal.

    Constraints:
    - fully invested
    - long-only
    - maximum position weight
    - expected return >= target return
    - annualized volatility <= risk-profile limit

    Covariance is expected to be based on daily returns.
    """

    _validate_inputs(
        expected_returns,
        covariance,
        target_return,
        max_weight,
    )

    profile = (
        risk_tolerance
        if isinstance(risk_tolerance, RiskProfile)
        else get_risk_profile(risk_tolerance)
    )

    tickers = list(expected_returns.index)

    mu = expected_returns.to_numpy(dtype=float)
    daily_sigma = covariance.to_numpy(dtype=float)

    # Numerical symmetry safeguard.
    daily_sigma = (daily_sigma + daily_sigma.T) / 2.0

    # Convert daily covariance to annual covariance.
    annual_sigma = daily_sigma * TRADING_DAYS_PER_YEAR

    weights = cp.Variable(len(tickers))

    portfolio_variance = cp.quad_form(
        weights,
        annual_sigma,
    )

    maximum_variance = profile.max_annual_volatility ** 2

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
        mu @ weights >= target_return,
        portfolio_variance <= maximum_variance,
    ]

    problem = cp.Problem(
        cp.Minimize(portfolio_variance),
        constraints,
    )

    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-9,
        tol_gap_rel=1e-9,
        tol_feas=1e-9,
        max_iter=500,
    )

    if problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:
        raise ValueError(
            "Goal is infeasible under the supplied "
            f"return, risk, and portfolio constraints. "
            f"Solver status: {problem.status}"
        )

    if weights.value is None:
        raise ValueError(
            "Optimizer did not return portfolio weights."
        )

    result_weights = pd.Series(
        np.asarray(weights.value).reshape(-1),
        index=tickers,
        dtype=float,
    )

    realized_expected_return = float(
        mu @ result_weights.to_numpy()
    )

    variance = float(
        result_weights.to_numpy().T
        @ annual_sigma
        @ result_weights.to_numpy()
    )

    expected_volatility = float(
        np.sqrt(max(variance, 0.0))
    )

    return GoalOptimizationResult(
        weights=result_weights,
        expected_return=realized_expected_return,
        expected_volatility=expected_volatility,
        target_return=float(target_return),
        maximum_allowed_volatility=(
            profile.max_annual_volatility
        ),
        risk_tolerance=profile.tolerance.value,
        status=problem.status,
    )