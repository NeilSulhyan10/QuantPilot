import cvxpy as cp
import numpy as np
import pandas as pd


def optimize_mean_variance(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    tickers: list[str],
    risk_aversion: float = 10.0,
    max_weight: float = 0.10,
) -> pd.Series:
    """
    Solve a long-only, fully-invested mean-variance optimization problem.

    Objective:
        maximize expected return - risk_aversion * portfolio variance

    Constraints:
        - Fully invested: sum(weights) == 1
        - Long-only: weights >= 0
        - Maximum position size: weights <= max_weight
    """

    if expected_returns.ndim != 1:
        raise ValueError("expected_returns must be a 1D array.")

    n_assets = len(expected_returns)

    if len(tickers) != n_assets:
        raise ValueError(
            "Number of tickers does not match number of assets."
        )

    if covariance.shape != (n_assets, n_assets):
        raise ValueError(
            "Covariance matrix shape does not match expected returns."
        )

    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")

    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be between 0 and 1.")

    if max_weight * n_assets < 1:
        raise ValueError(
            "max_weight is too small to construct a fully invested portfolio."
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains invalid values.")

    if not np.isfinite(covariance).all():
        raise ValueError("covariance contains invalid values.")

    weights = cp.Variable(n_assets)

    expected_return = expected_returns @ weights
    portfolio_variance = cp.quad_form(weights, covariance)

    objective = cp.Maximize(
        expected_return - risk_aversion * portfolio_variance
    )

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        weights <= max_weight,
    ]

    problem = cp.Problem(objective, constraints)

    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-9,
        tol_gap_rel=1e-9,
        tol_feas=1e-9,
        max_iter=500,
    )

    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(
            f"Optimization failed. Solver status: {problem.status}"
        )

    if weights.value is None:
        raise RuntimeError("Optimizer returned no weights.")

    solution = np.asarray(weights.value).flatten()

    result = pd.Series(
        solution,
        index=tickers,
        name="weight",
    )

    # CVXPY solvers operate with finite numerical tolerances.
    # Therefore, allow very small deviations from the hard constraints.
    tolerance = 1e-5

    if not np.isclose(result.sum(), 1.0, atol=tolerance):
        raise RuntimeError(
            f"Optimizer returned weights summing to "
            f"{result.sum():.10f}."
        )

    if result.min() < -tolerance:
        raise RuntimeError(
            "Optimizer returned materially negative portfolio weights."
        )

    if result.max() > max_weight + tolerance:
        raise RuntimeError(
            "Optimizer returned a materially oversized position."
        )

    return result