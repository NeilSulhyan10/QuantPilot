"""QuantPilot Goals Module."""

from .returns import (
    Trend,
    MultiHorizonReturns,
    TechnicalIndicators,
    StockAnalysis,
    calculate_multi_horizon_returns,
    calculate_moving_averages,
    calculate_trend_score,
    classify_trend,
    analyze_stock,
    screen_bearish_stocks,
    calculate_annual_volatility,
    calculate_sharpe_ratio,
)

from .planner import (
    ExpectedReturnModel,
    GoalPlan,
    SubsetResult,
    SubsetInvestmentResult,
    RiskProfileResult,
    build_goal_plan,
    build_goal_plan_from_selection,
    evaluate_company_subsets,
    evaluate_subsets_with_investment,
)

__all__ = [
    "Trend",
    "MultiHorizonReturns",
    "TechnicalIndicators",
    "StockAnalysis",
    "calculate_multi_horizon_returns",
    "calculate_moving_averages",
    "calculate_trend_score",
    "classify_trend",
    "analyze_stock",
    "screen_bearish_stocks",
    "calculate_annual_volatility",
    "calculate_sharpe_ratio",
    "ExpectedReturnModel",
    "GoalPlan",
    "SubsetResult",
    "SubsetInvestmentResult",
    "RiskProfileResult",
    "build_goal_plan",
    "build_goal_plan_from_selection",
    "evaluate_company_subsets",
    "evaluate_subsets_with_investment",
]