from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations

import numpy as np
import pandas as pd

import cvxpy as cp

from src.data.markets import Market, normalize_tickers

from src.goals.feasibility import (
    FeasibilityResult,
    assess_goal_feasibility,
    calculate_maximum_feasible_return,
    calculate_maximum_feasible_return_with_cash,
    calculate_recommended_target_return,
)
from src.goals.goal_math import (
    calculate_future_value,
    calculate_required_monthly_contribution,
)
from src.goals.portfolio import GoalPortfolioResult, build_goal_portfolio
from src.goals.optimizer import MIN_SHARPE_VOLATILITY, optimize_for_goal
from src.goals.scenarios import ScenarioSet, build_scenario_set
from src.data.market_data import MarketDataAdapter
from src.data.markets import Market
from src.goals.assets import load_selected_assets


class ExpectedReturnModel(str, Enum):
    """Expected return estimation models."""
    SHRINKAGE = "shrinkage"      # Cross-sectional shrinkage (current default)
    CAPM = "capm"                # CAPM: rf + beta * (market_rf)
    HISTORICAL_MEAN = "historical_mean"  # Raw historical mean (no shrinkage)


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


@dataclass(frozen=True)
class SubsetResult:
    """Result of evaluating a single company subset."""

    tickers: tuple[str, ...]
    subset_size: int
    max_weight: float

    feasible: bool
    goal_plan: GoalPlan | None = None
    error: str | None = None

    # Ranking metrics (populated if feasible)
    expected_return: float | None = None
    expected_volatility: float | None = None
    sharpe_ratio: float | None = None
    max_feasible_return: float | None = None
    recommended_return: float | None = None


def evaluate_company_subsets(
    all_tickers: list[str] | tuple[str, ...],
    market: Market | str,
    target_amount: float,
    years: float,
    risk_tolerance: str,
    minimum_observations: int = 252,
    estimation_window: int = 252,
    shrinkage_alpha: float = 0.5,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_download: bool = True,
    data_adapter: MarketDataAdapter | None = None,
    min_subset_size: int = 3,
) -> list[SubsetResult]:
    """
    Evaluate all company subsets from min_subset_size to len(all_tickers).

    For each subset:
    - Position cap = 1 / subset_size (auto-relaxed)
    - Target return = max_feasible_return × risk_profile_multiplier
    - Optimize portfolio

    Returns list of SubsetResult sorted by feasibility first, then by
    recommended return (can be re-ranked by caller).
    """
    if len(all_tickers) < min_subset_size:
        raise ValueError(
            f"Need at least {min_subset_size} tickers, got {len(all_tickers)}."
        )

    if len(all_tickers) > 15:
        raise ValueError(
            f"Maximum 15 tickers supported for subset evaluation, got {len(all_tickers)}."
        )

    # Load all asset data once
    asset_data = load_selected_assets(
        tickers=all_tickers,
        market=market,
        data_adapter=data_adapter,
        start_date=start_date,
        end_date=end_date,
        allow_download=allow_download,
    )

    results = []

    # Test all subset sizes from min_subset_size to N
    for subset_size in range(min_subset_size, len(all_tickers) + 1):
        max_weight = 1.0 / subset_size  # Auto-relaxed cap: 1/N

        for subset_tickers in combinations(all_tickers, subset_size):
            subset_list = list(subset_tickers)
            # Normalize subset tickers for asset_data lookup
            normalized_subset = normalize_tickers(subset_list, market)

            try:
                # Filter asset data for this subset (use normalized keys)
                subset_asset_data = {
                    t: asset_data[t] for t in normalized_subset
                }

                # Build goal plan with auto-relaxed cap
                plan = build_goal_plan(
                    asset_data=subset_asset_data,
                    target_amount=target_amount,
                    years=years,
                    risk_tolerance=risk_tolerance,
                    max_weight=max_weight,
                    minimum_observations=minimum_observations,
                    estimation_window=estimation_window,
                    shrinkage_alpha=shrinkage_alpha,
                )

                # Extract ranking metrics
                portfolio = plan.portfolio
                expected_return = portfolio.expected_return
                expected_volatility = portfolio.expected_volatility
                sharpe_ratio = _sharpe_or_none(
                    expected_return, expected_volatility
                )

                result = SubsetResult(
                    tickers=subset_tickers,
                    subset_size=subset_size,
                    max_weight=max_weight,
                    feasible=plan.feasibility.feasible,
                    goal_plan=plan,
                    expected_return=expected_return,
                    expected_volatility=expected_volatility,
                    sharpe_ratio=sharpe_ratio,
                    max_feasible_return=plan.maximum_feasible_return,
                    recommended_return=plan.recommended_return,
                )

            except Exception as exc:
                result = SubsetResult(
                    tickers=subset_tickers,
                    subset_size=subset_size,
                    max_weight=max_weight,
                    feasible=False,
                    error=str(exc),
                )

            results.append(result)

    # Sort: feasible first, then by recommended_return descending
    results.sort(
        key=lambda r: (
            not r.feasible,
            -(r.recommended_return if r.recommended_return is not None else -1.0),
        )
    )

    return results


@dataclass(frozen=True)
class RiskProfileResult:
    """Result for a single risk profile (Conservative/Moderate/Aggressive)."""

    risk_profile: str  # "conservative", "moderate", "aggressive"
    max_allowed_volatility: float
    max_feasible_return: float
    recommended_return: float
    expected_return: float | None
    expected_volatility: float | None
    sharpe_ratio: float | None
    future_value_lump_sum: float | None
    future_value_monthly: float | None
    feasible: bool
    warning: str | None = None
    weights: pd.Series | None = None
    cash_weight: float = 0.0


def _sharpe_or_none(expected_return: float, volatility: float) -> float | None:
    """Return a finite Sharpe estimate only when volatility is meaningful.

    Annualized volatility <= 1e-8 is treated as zero to avoid unstable ratios.
    """
    if not np.isfinite(expected_return) or not np.isfinite(volatility):
        return None
    if volatility <= MIN_SHARPE_VOLATILITY:
        return None
    value = expected_return / volatility
    return float(value) if np.isfinite(value) else None


def build_allocation_weights(
    tickers: tuple[str, ...] | list[str],
    weights: pd.Series,
    cash_weight: float = 0.0,
) -> pd.Series:
    """Map every selected ticker to its weight and append the cash allocation."""
    if len(set(tickers)) != len(tickers):
        raise ValueError("Selected tickers contain duplicates.")
    if not weights.index.is_unique:
        raise ValueError("Portfolio weights contain duplicate tickers.")
    if set(weights.index) != set(tickers):
        raise ValueError("Portfolio weights do not match selected tickers.")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio weights contain non-finite values.")
    if not np.isfinite(cash_weight) or cash_weight < -1e-8:
        raise ValueError("Cash weight must be finite and non-negative.")
    allocation = weights.reindex(list(tickers)).astype(float)
    allocation.loc["Cash"] = float(cash_weight)
    if not np.isclose(allocation.sum(), 1.0, atol=1e-5):
        raise ValueError("Stock and cash weights must sum to 1.")
    return allocation


@dataclass(frozen=True)
class SubsetInvestmentResult:
    """Result of evaluating a company subset with a given investment."""

    tickers: tuple[str, ...]
    subset_size: int
    max_weight: float
    feasible: bool
    error: str | None = None

    # Results for each risk profile
    conservative: RiskProfileResult | None = None
    moderate: RiskProfileResult | None = None
    aggressive: RiskProfileResult | None = None

    # Ranking metric (best of the three, e.g., max future value)
    best_future_value: float | None = None

    # Data quality warnings
    extreme_return_warnings: list[str] | None = None


def _sort_subset_results(
    results: list[SubsetInvestmentResult],
    ranking_metric: str,
) -> None:
    """Sort feasible subsets by the selected user-facing ranking objective."""
    profiles = ("conservative", "moderate", "aggressive")

    def profile_values(result: SubsetInvestmentResult, attribute: str, feasible_only: bool) -> list[float]:
        values = []
        for name in profiles:
            profile = getattr(result, name)
            if profile is None or (feasible_only and not profile.feasible):
                continue
            value = getattr(profile, attribute)
            if value is not None and np.isfinite(value):
                values.append(float(value))
        return values

    def best_expected_return(result: SubsetInvestmentResult) -> float | None:
        values = profile_values(result, "expected_return", feasible_only=True)
        if not values:
            values = profile_values(result, "expected_return", feasible_only=False)
        return max(values) if values else None

    def lowest_volatility(result: SubsetInvestmentResult) -> float | None:
        values = profile_values(result, "expected_volatility", feasible_only=True)
        if not values:
            values = profile_values(result, "expected_volatility", feasible_only=False)
        return min(values) if values else None

    if ranking_metric == "low_risk":
        results.sort(
            key=lambda result: (
                not result.feasible,
                lowest_volatility(result) if lowest_volatility(result) is not None else float("inf"),
                -(result.best_future_value if result.best_future_value is not None else float("-inf")),
                -(best_expected_return(result) if best_expected_return(result) is not None else float("-inf")),
            )
        )
    elif ranking_metric == "max_future_value":
        results.sort(
            key=lambda result: (
                not result.feasible,
                -(result.best_future_value if result.best_future_value is not None else float("-inf")),
                -(best_expected_return(result) if best_expected_return(result) is not None else float("-inf")),
            )
        )
    else:
        raise ValueError("ranking_metric must be 'low_risk' or 'max_future_value'.")


def _evaluate_subset_for_risk_profile(
    asset_data: dict[str, pd.DataFrame],
    years: float,
    risk_tolerance: str,
    max_weight: float,
    minimum_observations: int,
    estimation_window: int,
    shrinkage_alpha: float,
    lump_sum: float,
    monthly_sip: float,
    expected_return_model: ExpectedReturnModel = ExpectedReturnModel.SHRINKAGE,
    market: Market | str = Market.US,
    use_cash: bool = False,
    risk_free_rate: float = 0.045,
    max_cash_weight: float = 1.0,
) -> RiskProfileResult:
    """Evaluate a single subset for one risk profile."""
    # Initialize variables to prevent UnboundLocalError
    min_var_vol = None
    min_var_return = None
    min_var_weights = None

    from src.goals.assets import (
        build_selected_return_matrix,
        validate_minimum_history,
    )
    from src.goals.risk_profile import get_risk_profile
    from src.data.markets import Market

    returns = build_selected_return_matrix(asset_data)
    validate_minimum_history(returns, minimum_observations=minimum_observations)

    if len(returns) < estimation_window:
        raise ValueError("Insufficient observations for the estimation window.")

    estimation_returns = returns.tail(estimation_window)

    # ---- Expected Return Models ----
    if expected_return_model == ExpectedReturnModel.SHRINKAGE:
        daily_expected_returns = estimation_returns.mean()
        cross_sectional_mean = daily_expected_returns.mean()
        daily_expected_returns = (
            cross_sectional_mean
            + shrinkage_alpha * (daily_expected_returns - cross_sectional_mean)
        )

    elif expected_return_model == ExpectedReturnModel.HISTORICAL_MEAN:
        daily_expected_returns = estimation_returns.mean()

    elif expected_return_model == ExpectedReturnModel.CAPM:
        # CAPM: E[r_i] = rf + beta_i * (rm - rf)
        # Use US 10Y Treasury as risk-free rate (~4.5% annual)
        rf_annual = 0.045
        rf_daily = (1 + rf_annual) ** (1/252) - 1

        # Market proxy: SPY (or first asset if only subset available)
        market_returns = estimation_returns.mean(axis=1)  # Equal-weight market proxy
        market_excess = market_returns - rf_daily

        daily_expected_returns = {}
        for ticker in estimation_returns.columns:
            asset_returns = estimation_returns[ticker]
            # Covariance with market / variance of market = beta
            cov_asset_market = np.cov(asset_returns, market_returns)[0, 1]
            var_market = np.var(market_returns)
            if var_market > 1e-12:
                beta = cov_asset_market / var_market
            else:
                beta = 1.0
            # CAPM expected excess return
            expected_excess = beta * market_excess.mean()
            daily_expected_returns[ticker] = rf_daily + expected_excess
        daily_expected_returns = pd.Series(daily_expected_returns, index=estimation_returns.columns)

    else:
        raise ValueError(f"Unknown expected_return_model: {expected_return_model}")

    expected_returns = (1.0 + daily_expected_returns) ** 252 - 1.0
    covariance = estimation_returns.cov()

    profile = get_risk_profile(risk_tolerance)
    max_allowed_vol = profile.max_annual_volatility

    # Calculate maximum feasible return and build portfolio
    if use_cash:
        # Cash-aware feasibility and optimization
        try:
            maximum_feasible_return = calculate_maximum_feasible_return_with_cash(
                expected_returns=expected_returns,
                covariance=covariance,
                risk_tolerance=risk_tolerance,
                max_weight=max_weight,
                risk_free_rate=risk_free_rate,
                max_cash_weight=max_cash_weight,
            )
            used_vol_limit = max_allowed_vol
            vol_warning = None
            cash_feasible = True
        except ValueError as e:
            cash_feasible = False
            # Even with cash, no feasible portfolio exists
            raise ValueError(f"Cannot find feasible portfolio with cash for {risk_tolerance} profile.") from e

        recommended_return = calculate_recommended_target_return(
            maximum_feasible_return=maximum_feasible_return,
            risk_tolerance=risk_tolerance,
        )

        # With cash, we know the portfolio is feasible if max feasible return succeeded
        # (the cash optimizer will handle volatility constraint)
        # Build portfolio using cash optimizer
        from src.goals.portfolio import GoalPortfolioResult
        from src.goals.optimizer import optimize_for_goal_with_cash

        try:
            optimization = optimize_for_goal_with_cash(
                expected_returns=expected_returns,
                covariance=covariance,
                target_return=recommended_return,
                risk_tolerance=risk_tolerance,
                max_weight=max_weight,
                risk_free_rate=risk_free_rate,
                max_cash_weight=max_cash_weight,
            )
            portfolio = GoalPortfolioResult(
                weights=optimization.weights,
                expected_return=optimization.expected_return,
                expected_volatility=optimization.expected_volatility,
                target_return=optimization.target_return,
                risk_tolerance=optimization.risk_tolerance,
                maximum_allowed_volatility=optimization.maximum_allowed_volatility,
                feasible=True,
                cash_weight=optimization.cash_weight,
            )
        except ValueError as e:
            # Cash optimizer failed - target return too high for risk limit
            cash_feasible = False
            return RiskProfileResult(
                risk_profile=risk_tolerance,
                max_allowed_volatility=max_allowed_vol,
                max_feasible_return=maximum_feasible_return,
                recommended_return=recommended_return,
                expected_return=None,
                expected_volatility=None,
                sharpe_ratio=None,
                future_value_lump_sum=None,
                future_value_monthly=None,
                feasible=False,
                warning=f"Target return {recommended_return:.2%} exceeds what's achievable with cash at {max_allowed_vol:.0%} vol limit. " + str(e),
                weights=None,
                cash_weight=0.0,
            )

    else:
        # Standard feasibility and optimization (no cash)
        min_variance_fallback = False
        try:
            maximum_feasible_return = calculate_maximum_feasible_return(
                expected_returns=expected_returns,
                covariance=covariance,
                risk_tolerance=risk_tolerance,
                max_weight=max_weight,
            )
            used_vol_limit = max_allowed_vol
            vol_warning = None
        except ValueError as e:
            min_variance_fallback = True
            # Risk profile volatility limit too tight for this subset
            # Find minimum variance portfolio return instead
            n_assets = len(expected_returns)
            mu = expected_returns.to_numpy(dtype=float)
            daily_sigma = covariance.to_numpy(dtype=float)
            daily_sigma = (daily_sigma + daily_sigma.T) / 2.0
            annual_sigma = daily_sigma * 252

            weights = cp.Variable(len(expected_returns))
            portfolio_variance = cp.quad_form(weights, annual_sigma)
            constraints = [cp.sum(weights) == 1, weights >= 0, weights <= max_weight]
            problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
            problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=500)

            if problem.status in ["optimal", "optimal_inaccurate"] and weights.value is not None:
                min_var_weights = np.asarray(weights.value).flatten()
                min_var_variance = float(min_var_weights @ annual_sigma @ min_var_weights)
                min_var_vol = np.sqrt(max(min_var_variance, 0.0))
                min_var_return = float(expected_returns.to_numpy(dtype=float) @ min_var_weights)
                maximum_feasible_return = min_var_return
                used_vol_limit = min_var_vol
                vol_warning = (
                    f"Risk profile limit ({max_allowed_vol:.0%}) too tight for {len(expected_returns)} stocks. "
                    f"Using minimum variance portfolio ({min_var_vol:.0%} vol)."
                )
            else:
                raise ValueError(f"Cannot find feasible portfolio for {risk_tolerance} profile.")

        recommended_return = calculate_recommended_target_return(
            maximum_feasible_return=maximum_feasible_return,
            risk_tolerance=risk_tolerance,
        )

        from src.goals.portfolio import GoalPortfolioResult
        if min_variance_fallback:
            # The risk limit excluded all portfolios in the return-maximization
            # problem. Report the minimum-variance portfolio as infeasible when
            # its volatility still exceeds the profile limit.
            vol_feasible = min_var_vol <= max_allowed_vol + 1e-8
            ret_feasible = recommended_return <= maximum_feasible_return + 1e-8
            fully_feasible = vol_feasible and ret_feasible

            if not fully_feasible:
                warning_msg = vol_warning
                if not vol_feasible:
                    warning_msg = (warning_msg + " " if warning_msg else "") + \
                        f"Min-variance portfolio volatility ({min_var_vol:.1%}) exceeds risk limit ({max_allowed_vol:.1%})."

                return RiskProfileResult(
                    risk_profile=risk_tolerance,
                    max_allowed_volatility=min_var_vol,
                    max_feasible_return=maximum_feasible_return,
                    recommended_return=recommended_return,
                    expected_return=min_var_return,
                    expected_volatility=min_var_vol,
                    sharpe_ratio=_sharpe_or_none(min_var_return, min_var_vol),
                    future_value_lump_sum=None,
                    future_value_monthly=None,
                    feasible=False,
                    warning=warning_msg,
                    weights=pd.Series(min_var_weights, index=expected_returns.index),
                    cash_weight=0.0,
                )

            portfolio = GoalPortfolioResult(
                weights=pd.Series(min_var_weights, index=expected_returns.index),
                expected_return=min_var_return,
                expected_volatility=min_var_vol,
                target_return=recommended_return,
                risk_tolerance=risk_tolerance,
                maximum_allowed_volatility=used_vol_limit,
                feasible=True,
            )
        else:
            optimization = optimize_for_goal(
                expected_returns=expected_returns,
                covariance=covariance,
                target_return=recommended_return,
                risk_tolerance=risk_tolerance,
                max_weight=max_weight,
            )
            portfolio = GoalPortfolioResult(
                weights=optimization.weights,
                expected_return=optimization.expected_return,
                expected_volatility=optimization.expected_volatility,
                target_return=optimization.target_return,
                risk_tolerance=optimization.risk_tolerance,
                maximum_allowed_volatility=optimization.maximum_allowed_volatility,
                feasible=True,
            )

    exp_ret = portfolio.expected_return
    exp_vol = portfolio.expected_volatility
    sharpe = _sharpe_or_none(exp_ret, exp_vol)

    # Calculate future values
    fv_lump = 0.0
    fv_monthly = 0.0

    if lump_sum > 0:
        fv_lump = calculate_future_value(lump_sum, recommended_return, years)

    if monthly_sip > 0:
        fv_monthly = calculate_future_value(
            monthly_sip * 12,  # annual equivalent for monthly SIP formula
            recommended_return,
            years,
        )  # Note: this is approximate; proper SIP formula in goal_math

    # Explicit feasibility check: BOTH return AND volatility constraints
    vol_feasible = exp_vol <= max_allowed_vol + 1e-8
    ret_feasible = recommended_return <= maximum_feasible_return + 1e-8
    fully_feasible = vol_feasible and ret_feasible

    return RiskProfileResult(
        risk_profile=risk_tolerance,
        max_allowed_volatility=exp_vol,  # Actual portfolio volatility, not limit
        max_feasible_return=maximum_feasible_return,
        recommended_return=recommended_return,
        expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe_ratio=sharpe,
        future_value_lump_sum=fv_lump,
        future_value_monthly=fv_monthly,
        feasible=fully_feasible,
        warning=vol_warning,
        weights=portfolio.weights,
        cash_weight=getattr(portfolio, 'cash_weight', 0.0),
    )


def check_extreme_returns(
    returns: pd.DataFrame,
    estimation_window: int,
    max_daily_return: float = 0.50,
    max_abs_return: float = 0.50,
) -> tuple[list[str], list[str]]:
    """
    Check for extreme daily returns in the estimation window.

    Args:
        returns: Full return matrix
        estimation_window: Number of trading days to check
        max_daily_return: Threshold for max daily return (default 50%)
        max_abs_return: Threshold for absolute daily return (default 50%)

    Returns:
        Tuple of (warnings list, flagged_tickers list)
    """
    est = returns.tail(estimation_window)
    warnings = []
    flagged = []

    for ticker in est.columns:
        max_ret = est[ticker].max()
        min_ret = est[ticker].min()
        if max_ret > max_daily_return or abs(min_ret) > max_abs_return:
            flagged.append(ticker)
            warnings.append(
                f"{ticker}: extreme daily return detected "
                f"(max={max_ret:.2%}, min={min_ret:.2%}). "
                f"Estimation may be unreliable."
            )

    return warnings, flagged


def evaluate_subsets_with_investment(
    all_tickers: list[str] | tuple[str, ...],
    market: Market | str,
    lump_sum: float,
    monthly_sip: float,
    years: float,
    minimum_observations: int = 252,
    estimation_window: int = 252,
    shrinkage_alpha: float = 0.5,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_download: bool = True,
    data_adapter: MarketDataAdapter | None = None,
    min_subset_size: int = 3,
    ranking_metric: str = "max_future_value",
    max_subset_size: int | None = None,
    max_position_weight: float = 0.25,
) -> list[SubsetInvestmentResult]:
    """
    Evaluate all company subsets with a given investment amount.

    For each subset:
    - Position cap defaults to max_position_weight and is raised only as needed
      to make the selected subset fully investable
    - Tests all 3 risk profiles (Conservative/Moderate/Aggressive)
    - Calculates future value for lump sum and/or monthly SIP

    Results are sorted by the requested ranking metric, with feasible subsets first.
    """
    if ranking_metric not in {"low_risk", "max_future_value"}:
        raise ValueError("ranking_metric must be 'low_risk' or 'max_future_value'.")

    if lump_sum <= 0 and monthly_sip <= 0:
        raise ValueError("At least one of lump_sum or monthly_sip must be positive.")

    if years <= 0:
        raise ValueError("years must be greater than 0.")

    if len(all_tickers) < min_subset_size:
        raise ValueError(
            f"Need at least {min_subset_size} tickers, got {len(all_tickers)}."
        )

    if len(all_tickers) > 15:
        raise ValueError(
            f"Maximum 15 tickers supported for subset evaluation, got {len(all_tickers)}."
        )

    if max_subset_size is None:
        max_subset_size = len(all_tickers)
    if max_subset_size < min_subset_size or max_subset_size > len(all_tickers):
        raise ValueError(
            f"max_subset_size must be between {min_subset_size} and {len(all_tickers)}."
        )
    if not 0 < max_position_weight <= 1:
        raise ValueError("max_position_weight must be greater than 0 and at most 1.")

    # Load all asset data once
    asset_data = load_selected_assets(
        tickers=all_tickers,
        market=market,
        data_adapter=data_adapter,
        start_date=start_date,
        end_date=end_date,
        allow_download=allow_download,
    )

    # Build full return matrix for extreme-return check
    from src.goals.assets import build_selected_return_matrix
    full_returns = build_selected_return_matrix(asset_data)

    # Check for extreme returns in estimation window
    extreme_warnings, extreme_tickers = check_extreme_returns(
        full_returns, estimation_window, max_daily_return=0.50, max_abs_return=0.50
    )

    risk_profiles = ["conservative", "moderate", "aggressive"]
    results = []

    for subset_size in range(min_subset_size, max_subset_size + 1):
        # Keep a useful concentration limit while relaxing it only as much as
        # necessary to make a fully invested portfolio feasible.
        max_weight = max(max_position_weight, 1.0 / subset_size)

        for subset_tickers in combinations(all_tickers, subset_size):
            subset_list = list(subset_tickers)
            normalized_subset = normalize_tickers(subset_list, market)

            try:
                subset_asset_data = {
                    t: asset_data[t] for t in normalized_subset
                }

                # Filter extreme warnings for this subset
                subset_warnings = [
                    w for w in extreme_warnings
                    if any(t in w for t in normalized_subset)
                ]

                risk_results = {}
                any_feasible = False

                for rp in risk_profiles:
                    rpr = _evaluate_subset_for_risk_profile(
                        asset_data=subset_asset_data,
                        years=years,
                        risk_tolerance=rp,
                        max_weight=max_weight,
                        minimum_observations=minimum_observations,
                        estimation_window=estimation_window,
                        shrinkage_alpha=shrinkage_alpha,
                        lump_sum=lump_sum,
                        monthly_sip=monthly_sip,
                        expected_return_model=ExpectedReturnModel.SHRINKAGE,
                        market=market,
                        use_cash=False,
                        risk_free_rate=0.045,
                        max_cash_weight=1.0,
                    )
                    risk_results[rp] = rpr
                    if rpr.feasible:
                        any_feasible = True

                # Best future value across risk profiles
                best_fv = 0.0
                for rpr in risk_results.values():
                    if rpr.feasible:
                        if lump_sum > 0:
                            if rpr.future_value_lump_sum is not None:
                                best_fv = max(best_fv, rpr.future_value_lump_sum)
                        if monthly_sip > 0:
                            if rpr.future_value_monthly is not None:
                                best_fv = max(best_fv, rpr.future_value_monthly)

                result = SubsetInvestmentResult(
                    tickers=subset_tickers,
                    subset_size=subset_size,
                    max_weight=max_weight,
                    feasible=any_feasible,
                    conservative=risk_results["conservative"],
                    moderate=risk_results["moderate"],
                    aggressive=risk_results["aggressive"],
                    best_future_value=best_fv if best_fv > 0 else None,
                    extreme_return_warnings=subset_warnings if subset_warnings else None,
                )

            except Exception as exc:
                result = SubsetInvestmentResult(
                    tickers=subset_tickers,
                    subset_size=subset_size,
                    max_weight=max_weight,
                    feasible=False,
                    error=str(exc),
                    extreme_return_warnings=subset_warnings if subset_warnings else None,
                )

            results.append(result)

    _sort_subset_results(results, ranking_metric)

    return results


def build_goal_plan(
    asset_data: dict[str, pd.DataFrame],
    target_amount: float,
    years: float,
    risk_tolerance: str,
    max_weight: float = 0.10,
    minimum_observations: int = 252,
    estimation_window: int = 252,
    shrinkage_alpha: float = 0.5,
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

    if not 0.0 <= shrinkage_alpha <= 1.0:
        raise ValueError("shrinkage_alpha must be between 0 and 1.")

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

    # Apply cross-sectional shrinkage (QuantPilot V2 style)
    cross_sectional_mean = daily_expected_returns.mean()
    daily_expected_returns = (
        cross_sectional_mean
        + shrinkage_alpha * (daily_expected_returns - cross_sectional_mean)
    )

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
        shrinkage_alpha=shrinkage_alpha,
        use_cash=True,
        risk_free_rate=0.045,
        max_cash_weight=1.0,
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

def build_goal_plan_from_selection(
    tickers: list[str] | tuple[str, ...],
    market: Market | str,
    target_amount: float,
    years: float,
    risk_tolerance: str,
    max_weight: float = 0.10,
    minimum_observations: int = 252,
    estimation_window: int = 252,
    shrinkage_alpha: float = 0.5,
    start_date: str | None = None,
    end_date: str | None = None,
    allow_download: bool = True,
    data_adapter: MarketDataAdapter | None = None,
) -> GoalPlan:
    """
    Build a complete goal plan from a user-selected market
    and asset universe.

    This is the V3 market-aware entry point. The underlying
    goal mathematics and portfolio optimizer remain market-
    agnostic.
    """

    asset_data = load_selected_assets(
        tickers=tickers,
        market=market,
        data_adapter=data_adapter,
        start_date=start_date,
        end_date=end_date,
        allow_download=allow_download,
    )

    return build_goal_plan(
        asset_data=asset_data,
        target_amount=target_amount,
        years=years,
        risk_tolerance=risk_tolerance,
        max_weight=max_weight,
        minimum_observations=minimum_observations,
        estimation_window=estimation_window,
        shrinkage_alpha=shrinkage_alpha,
    )
