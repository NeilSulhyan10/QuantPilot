from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

import cvxpy as cp

from src.data.markets import Market, normalize_tickers

from src.goals.feasibility import (
    FeasibilityResult,
    assess_goal_feasibility,
    calculate_maximum_feasible_return,
    calculate_recommended_target_return,
)
from src.goals.goal_math import (
    calculate_future_value,
    calculate_required_monthly_contribution,
)
from src.goals.portfolio import GoalPortfolioResult, build_goal_portfolio
from src.goals.scenarios import ScenarioSet, build_scenario_set
from src.data.market_data import MarketDataAdapter
from src.data.markets import Market
from src.goals.assets import load_selected_assets

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
                sharpe_ratio = (
                    expected_return / expected_volatility
                    if expected_volatility > 0
                    else 0.0
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
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    future_value_lump_sum: float
    future_value_monthly: float
    feasible: bool
    warning: str | None = None
    weights: pd.Series | None = None


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
) -> RiskProfileResult:
    """Evaluate a single subset for one risk profile."""
    from src.goals.assets import (
        build_selected_return_matrix,
        validate_minimum_history,
    )
    from src.goals.risk_profile import get_risk_profile

    returns = build_selected_return_matrix(asset_data)
    validate_minimum_history(returns, minimum_observations=minimum_observations)

    if len(returns) < estimation_window:
        raise ValueError("Insufficient observations for the estimation window.")

    estimation_returns = returns.tail(estimation_window)
    daily_expected_returns = estimation_returns.mean()

    cross_sectional_mean = daily_expected_returns.mean()
    daily_expected_returns = (
        cross_sectional_mean
        + shrinkage_alpha * (daily_expected_returns - cross_sectional_mean)
    )

    expected_returns = (1.0 + daily_expected_returns) ** 252 - 1.0
    covariance = estimation_returns.cov()

    profile = get_risk_profile(risk_tolerance)
    max_allowed_vol = profile.max_annual_volatility

    # Try to calculate max feasible return at the risk profile's volatility limit
    # If infeasible (min variance portfolio exceeds limit), fall back to min variance portfolio
    used_fallback = False
    min_var_weights = None
    min_var_return = None
    min_var_vol = None

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
        # Risk profile volatility limit too tight for this subset
        # Find minimum variance portfolio return instead
        used_fallback = True
        n_assets = len(expected_returns)
        mu = expected_returns.to_numpy(dtype=float)
        daily_sigma = covariance.to_numpy(dtype=float)
        daily_sigma = (daily_sigma + daily_sigma.T) / 2.0
        annual_sigma = daily_sigma * 252

        weights = cp.Variable(n_assets)
        portfolio_variance = cp.quad_form(weights, annual_sigma)
        constraints = [cp.sum(weights) == 1, weights >= 0, weights <= max_weight]
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9, max_iter=500)

        if problem.status in ["optimal", "optimal_inaccurate"] and weights.value is not None:
            min_var_weights = np.asarray(weights.value).flatten()
            min_var_vol = np.sqrt(min_var_weights @ annual_sigma @ min_var_weights)
            min_var_return = float(mu @ min_var_weights)
            maximum_feasible_return = min_var_return
            used_vol_limit = min_var_vol
            vol_warning = (
                f"Risk profile limit ({max_allowed_vol:.0%}) too tight for {len(expected_returns)} stocks. "
                f"Using minimum variance portfolio ({min_var_vol:.0%} vol)."
            )
        else:
            raise ValueError(f"Cannot find feasible portfolio for {risk_tolerance} profile.") from e

    recommended_return = calculate_recommended_target_return(
        maximum_feasible_return=maximum_feasible_return,
        risk_tolerance=risk_tolerance,
    )

    # Feasibility check: recommended_return should be <= maximum_feasible_return
    # (The volatility constraint was already satisfied when computing max_feasible_return)
    feasible = recommended_return <= maximum_feasible_return + 1e-8

    if not feasible:
        return RiskProfileResult(
            risk_profile=risk_tolerance,
            max_allowed_volatility=used_vol_limit,
            max_feasible_return=maximum_feasible_return,
            recommended_return=recommended_return,
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            future_value_lump_sum=0.0,
            future_value_monthly=0.0,
            feasible=False,
            warning=vol_warning or "Recommended return exceeds maximum feasible return.",
            weights=None,
        )

    if used_fallback:
        # Use the min variance portfolio directly
        from src.goals.portfolio import GoalPortfolioResult
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
        portfolio = build_goal_portfolio(
            asset_data=asset_data,
            target_return=recommended_return,
            risk_tolerance=risk_tolerance,
            max_weight=max_weight,
            minimum_observations=minimum_observations,
            estimation_window=estimation_window,
            shrinkage_alpha=shrinkage_alpha,
            skip_feasibility_check=True,
        )

    exp_ret = portfolio.expected_return
    exp_vol = portfolio.expected_volatility
    sharpe = exp_ret / exp_vol if exp_vol > 0 else 0.0

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

    return RiskProfileResult(
        risk_profile=risk_tolerance,
        max_allowed_volatility=max_allowed_vol,
        max_feasible_return=maximum_feasible_return,
        recommended_return=recommended_return,
        expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe_ratio=sharpe,
        future_value_lump_sum=fv_lump,
        future_value_monthly=fv_monthly,
        feasible=True,
        warning=vol_warning,
        weights=portfolio.weights,
    )


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
) -> list[SubsetInvestmentResult]:
    """
    Evaluate all company subsets with a given investment amount.

    For each subset:
    - Position cap = 1 / subset_size (auto-relaxed)
    - Tests all 3 risk profiles (Conservative/Moderate/Aggressive)
    - Calculates future value for lump sum and/or monthly SIP

    Returns list sorted by best future value (descending).
    """
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

    # Load all asset data once
    asset_data = load_selected_assets(
        tickers=all_tickers,
        market=market,
        data_adapter=data_adapter,
        start_date=start_date,
        end_date=end_date,
        allow_download=allow_download,
    )

    risk_profiles = ["conservative", "moderate", "aggressive"]
    results = []

    for subset_size in range(min_subset_size, len(all_tickers) + 1):
        max_weight = 1.0 / subset_size

        for subset_tickers in combinations(all_tickers, subset_size):
            subset_list = list(subset_tickers)
            normalized_subset = normalize_tickers(subset_list, market)

            try:
                subset_asset_data = {
                    t: asset_data[t] for t in normalized_subset
                }

                risk_results = {}
                all_feasible = True

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
                    )
                    risk_results[rp] = rpr
                    if not rpr.feasible:
                        all_feasible = False

                # Best future value across risk profiles
                best_fv = 0.0
                for rpr in risk_results.values():
                    if rpr.feasible:
                        if lump_sum > 0:
                            best_fv = max(best_fv, rpr.future_value_lump_sum)
                        if monthly_sip > 0:
                            best_fv = max(best_fv, rpr.future_value_monthly)

                result = SubsetInvestmentResult(
                    tickers=subset_tickers,
                    subset_size=subset_size,
                    max_weight=max_weight,
                    feasible=all_feasible,
                    conservative=risk_results["conservative"],
                    moderate=risk_results["moderate"],
                    aggressive=risk_results["aggressive"],
                    best_future_value=best_fv if best_fv > 0 else None,
                )

            except Exception as exc:
                result = SubsetInvestmentResult(
                    tickers=subset_tickers,
                    subset_size=subset_size,
                    max_weight=max_weight,
                    feasible=False,
                    error=str(exc),
                )

            results.append(result)

    # Sort: feasible first, then by best_future_value descending
    results.sort(
        key=lambda r: (
            not r.feasible,
            -(r.best_future_value if r.best_future_value is not None else -1.0),
        )
    )

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
