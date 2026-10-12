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
    warning: str | None = None


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

    if not np.isfinite(expected_returns.to_numpy(dtype=float)).all():
        raise ValueError("expected_returns contains non-finite values.")
    covariance_values = covariance.to_numpy(dtype=float)
    if not np.isfinite(covariance_values).all():
        raise ValueError("covariance contains non-finite values.")
    if not np.allclose(covariance_values, covariance_values.T, atol=1e-10, rtol=1e-8):
        raise ValueError("covariance must be symmetric.")
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_values).min())
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance eigendecomposition failed.") from exc
    if minimum_eigenvalue < -1e-10:
        raise ValueError("covariance must be positive semidefinite.")

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

    optimized_weights = np.asarray(weights.value).reshape(-1)
    if not np.isfinite(optimized_weights).all():
        raise ValueError("Feasibility optimizer returned non-finite weights.")
    result = float(mu @ optimized_weights)
    if not np.isfinite(result):
        raise ValueError("Maximum feasible return is non-finite.")
    return result


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

    warning = None
    if expected_returns.max() < 0:
        warning = (
            "All expected returns are negative. The selected universe and "
            "estimation window imply negative expected returns. Consider "
            "using a longer estimation window or a different asset universe."
        )
    elif expected_returns.mean() < 0:
        warning = (
            "Average expected return is negative. The selected universe and "
            "estimation window imply negative expected returns. Consider "
            "using a longer estimation window or a different asset universe."
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
        warning=warning,
    )

def calculate_recommended_target_return(
    maximum_feasible_return: float,
    risk_tolerance: RiskProfile | str,
) -> float:
    """Calculate a risk-profile-specific recommended target return.

    The maximum feasible return is treated as the upper boundary.
    The recommendation uses a fraction of that feasible return:

    Conservative: 50%
    Moderate:     70%
    Aggressive:   85%

    When maximum_feasible_return is negative, the recommended return
    is set equal to the maximum feasible return (acknowledging that
    the universe cannot achieve positive returns under the risk constraint).

    This is a planning convention, not a forecast or guarantee.
    """

    profile = (
        risk_tolerance
        if isinstance(risk_tolerance, RiskProfile)
        else get_risk_profile(risk_tolerance)
    )

    multipliers = {
        "conservative": 0.50,
        "moderate": 0.70,
        "aggressive": 0.85,
    }

    multiplier = multipliers[profile.tolerance.value]

    if maximum_feasible_return < 0:
        return float(maximum_feasible_return)

    return float(maximum_feasible_return * multiplier)


def calculate_maximum_feasible_return_with_cash(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_tolerance: RiskProfile | str,
    max_weight: float = 0.10,
    risk_free_rate: float = 0.045,
    max_cash_weight: float = 1.0,
) -> float:
    """Calculate the maximum expected return under risk constraints with optional cash asset.

    Cash asset properties:
    - Expected return = risk_free_rate (annual)
    - Volatility = 0
    - Covariance with all assets = 0

    The optimizer maximizes expected return subject to:
    - fully invested portfolio (stocks + cash = 1)
    - long-only positions
    - maximum position weight for stocks
    - cash weight between 0 and max_cash_weight
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

    if not np.isfinite(expected_returns.to_numpy(dtype=float)).all():
        raise ValueError("expected_returns contains non-finite values.")
    covariance_values = covariance.to_numpy(dtype=float)
    if not np.isfinite(covariance_values).all():
        raise ValueError("covariance contains non-finite values.")
    if not np.allclose(covariance_values, covariance_values.T, atol=1e-10, rtol=1e-8):
        raise ValueError("covariance must be symmetric.")
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_values).min())
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance eigendecomposition failed.") from exc
    if minimum_eigenvalue < -1e-10:
        raise ValueError("covariance must be positive semidefinite.")

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

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite.")
    if not 0 <= max_cash_weight <= 1:
        raise ValueError("max_cash_weight must be between 0 and 1.")

    profile = (
        risk_tolerance
        if isinstance(risk_tolerance, RiskProfile)
        else get_risk_profile(risk_tolerance)
    )

    n_stocks = len(expected_returns)

    # Build combined expected returns: [cash, stocks]
    mu = np.concatenate([[risk_free_rate], expected_returns.to_numpy(dtype=float)])

    # Build combined covariance: cash has 0 variance and 0 covariance
    daily_sigma = covariance.to_numpy(dtype=float)
    daily_sigma = (daily_sigma + daily_sigma.T) / 2.0
    annual_sigma = daily_sigma * TRADING_DAYS_PER_YEAR

    n_stocks = len(expected_returns)
    n_total = n_stocks + 1

    annual_sigma_combined = np.zeros((n_total, n_total))
    annual_sigma_combined[1:, 1:] = covariance.to_numpy(dtype=float) * TRADING_DAYS_PER_YEAR

    mu_full = np.concatenate([[risk_free_rate], expected_returns.to_numpy(dtype=float)])

    weights = cp.Variable(n_total)

    portfolio_variance = cp.quad_form(
        weights,
        annual_sigma_combined,
    )

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights[1:] <= max_weight,
        weights[0] <= max_cash_weight,
        cp.quad_form(weights, annual_sigma_combined) <= profile.max_annual_volatility ** 2,
    ]

    # Linear objective: maximize sum(weights * mu)
    objective = cp.Maximize(mu_full @ weights)

    problem = cp.Problem(objective, constraints)

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
            "Unable to determine maximum feasible return with cash. "
            f"Solver status: {problem.status}"
        )

    if weights.value is None:
        raise ValueError(
            "Feasibility optimizer did not return weights."
        )

    optimized_weights = np.asarray(weights.value).reshape(-1)
    if not np.isfinite(optimized_weights).all():
        raise ValueError("Feasibility optimizer returned non-finite weights.")
    if not np.isclose(optimized_weights.sum(), 1.0, atol=1e-5):
        raise ValueError("Feasibility optimizer weights do not sum to 1.")
    result = float(mu_full @ optimized_weights)
    if not np.isfinite(result):
        raise ValueError("Maximum feasible return is non-finite.")
    return result
