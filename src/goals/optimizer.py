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
# Annualized volatility below 0.01% is treated as effectively zero. This
# covers solver residuals around a zero-volatility cash allocation and avoids
# unstable Sharpe ratios at the precision shown by the Research Lab.
MIN_SHARPE_VOLATILITY = 1e-4


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
    cash_weight: float = 0.0


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

    covariance_values = covariance.to_numpy(dtype=float)
    if not np.allclose(covariance_values, covariance_values.T, atol=1e-10, rtol=1e-8):
        raise ValueError("covariance must be symmetric.")
    try:
        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_values).min())
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance eigendecomposition failed.") from exc
    if minimum_eigenvalue < -1e-10:
        raise ValueError("covariance must be positive semidefinite.")

    if target_return <= -1:
        raise ValueError(
            "target_return must be greater than -100%."
        )
    if not np.isfinite(target_return):
        raise ValueError("target_return must be finite.")

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

    if not np.isfinite(np.asarray(weights.value)).all():
        raise ValueError("Optimizer returned non-finite portfolio weights.")

    result_weights = pd.Series(
        np.asarray(weights.value).reshape(-1),
        index=tickers,
        dtype=float,
    )
    if not np.isclose(result_weights.sum(), 1.0, atol=1e-5):
        raise ValueError("Optimizer stock weights do not sum to 1.")

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
    if not np.isfinite(realized_expected_return) or not np.isfinite(expected_volatility):
        raise ValueError("Portfolio metrics are non-finite.")

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
        cash_weight=0.0,
    )


def optimize_for_goal_with_cash(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target_return: float,
    risk_tolerance: RiskProfile | str,
    max_weight: float = 0.10,
    risk_free_rate: float = 0.045,
    max_cash_weight: float = 1.0,
) -> GoalOptimizationResult:
    """Find the minimum-variance portfolio satisfying the goal with optional cash asset.

    Cash asset properties:
    - Expected return = risk_free_rate (annual)
    - Volatility = 0
    - Covariance with all assets = 0

    Constraints:
    - fully invested (stocks + cash = 1)
    - long-only (including cash)
    - maximum position weight for stocks
    - cash weight between 0 and max_cash_weight
    - expected return >= target return
    - annualized volatility <= risk-profile limit
    """
    _validate_inputs(
        expected_returns,
        covariance,
        target_return,
        max_weight,
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

    stock_tickers = list(expected_returns.index)
    n_stocks = len(stock_tickers)

    # Stock expected returns and covariance
    mu_stocks = expected_returns.to_numpy(dtype=float)
    daily_sigma = covariance.to_numpy(dtype=float)
    daily_sigma = (daily_sigma + daily_sigma.T) / 2.0
    annual_sigma = daily_sigma * TRADING_DAYS_PER_YEAR

    # Cash asset properties
    rf_annual = risk_free_rate
    rf_daily = (1.0 + rf_annual) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    # Cash alone is the exact minimum-variance solution whenever it meets the
    # return target and the full cash allocation is allowed. Solving this
    # corner case numerically can leave tiny residual stock positions.
    if rf_annual >= target_return and max_cash_weight >= 1.0:
        return GoalOptimizationResult(
            weights=pd.Series(0.0, index=stock_tickers, dtype=float),
            expected_return=float(rf_annual),
            expected_volatility=0.0,
            target_return=float(target_return),
            maximum_allowed_volatility=profile.max_annual_volatility,
            risk_tolerance=profile.tolerance.value,
            status=cp.OPTIMAL,
            cash_weight=1.0,
        )

    # Build combined expected returns: [cash, stocks]
    mu = np.concatenate([[rf_annual], mu_stocks])

    # Build combined covariance: cash has 0 variance and 0 covariance
    n_total = n_stocks + 1
    annual_sigma_combined = np.zeros((n_total, n_total))
    annual_sigma_combined[1:, 1:] = annual_sigma
    # Cash row/col remains zeros (0 variance, 0 covariance)

    weights = cp.Variable(n_total)

    portfolio_variance = cp.quad_form(weights, annual_sigma_combined)

    maximum_variance = profile.max_annual_volatility ** 2

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights[1:] <= max_weight,  # max weight only for stocks
        weights[0] <= max_cash_weight,  # cash weight limit
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

    result_weights = np.asarray(weights.value).flatten()
    if not np.isfinite(result_weights).all():
        raise ValueError("Optimizer returned non-finite portfolio weights.")

    cash_weight = float(result_weights[0])
    stock_weights = result_weights[1:]

    # Verify constraints
    if not np.isclose(cash_weight + stock_weights.sum(), 1.0, atol=1e-5):
        raise RuntimeError("Weight sum constraint violated.")

    result_weights_series = pd.Series(
        stock_weights,
        index=stock_tickers,
        dtype=float,
    )

    realized_expected_return = float(
        rf_annual * cash_weight + mu_stocks @ stock_weights
    )

    variance = float(
        stock_weights @ annual_sigma @ stock_weights
    )

    expected_volatility = float(
        np.sqrt(max(variance, 0.0))
    )
    if not np.isfinite(realized_expected_return) or not np.isfinite(expected_volatility):
        raise ValueError("Portfolio metrics are non-finite.")

    return GoalOptimizationResult(
        weights=result_weights_series,
        expected_return=realized_expected_return,
        expected_volatility=expected_volatility,
        target_return=float(target_return),
        maximum_allowed_volatility=(
            profile.max_annual_volatility
        ),
        risk_tolerance=profile.tolerance.value,
        status=problem.status,
        cash_weight=cash_weight,
    )
