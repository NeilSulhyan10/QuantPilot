"""Goal feasibility analysis for QuantPilot V2."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from src.goals.risk_profile import RiskProfile, get_risk_profile


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class FeasibilityResult:
    """Result of a QuantPilot goal feasibility analysis."""

    feasible: bool
    target_return: float
    maximum_feasible_return: float
    risk_tolerance: str
    maximum_allowed_volatility: float


def calculate_maximum_feasible_return(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_tolerance: RiskProfile | str,
    max_weight: float = 0.10,
) -> float:
    """Calculate the maximum expected return under risk constraints.

    The optimizer maximizes expected return subject to:
    - fully invested portfolio
    - long-only positions
    - maximum position weight
    - annualized volatility limit
    """

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

    if not 0 < max_weight <= 1:
        raise ValueError(
            "max_weight must be greater than 0 and at most 1."
        )

    n_assets = len(expected_returns)

    if max_weight * n_assets < 1.0 - 1e-12:
        raise ValueError(
            "max_weight is too restrictive for the number of assets."
        )

    profile = (
        risk_tolerance
        if isinstance(risk_tolerance, RiskProfile)
        else get_risk_profile(risk_tolerance)
    )

    mu = expected_returns.to_numpy(dtype=float)

    daily_sigma = covariance.to_numpy(dtype=float)
    daily_sigma = (daily_sigma + daily_sigma.T) / 2.0

    annual_sigma = daily_sigma * TRADING_DAYS_PER_YEAR

    weights = cp.Variable(n_assets)

    portfolio_variance = cp.quad_form(
        weights,
        annual_sigma,
    )

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
        portfolio_variance
        <= profile.max_annual_volatility ** 2,
    ]

    problem = cp.Problem(
        cp.Maximize(mu @ weights),
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
            "Unable to determine maximum feasible return. "
            f"Solver status: {problem.status}"
        )

    if weights.value is None:
        raise ValueError(
            "Feasibility optimizer did not return weights."
        )

    return float(mu @ np.asarray(weights.value).reshape(-1))


def assess_goal_feasibility(
    target_return: float,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_tolerance: RiskProfile | str,
    max_weight: float = 0.10,
) -> FeasibilityResult:
    """Determine whether a target return is feasible."""

    if target_return <= -1:
        raise ValueError(
            "target_return must be greater than -100%."
        )

    profile = (
        risk_tolerance
        if isinstance(risk_tolerance, RiskProfile)
        else get_risk_profile(risk_tolerance)
    )

    maximum_return = calculate_maximum_feasible_return(
        expected_returns=expected_returns,
        covariance=covariance,
        risk_tolerance=profile,
        max_weight=max_weight,
    )

    return FeasibilityResult(
        feasible=(
            target_return
            <= maximum_return + 1e-8
        ),
        target_return=float(target_return),
        maximum_feasible_return=maximum_return,
        risk_tolerance=profile.tolerance.value,
        maximum_allowed_volatility=(
            profile.max_annual_volatility
        ),
    )