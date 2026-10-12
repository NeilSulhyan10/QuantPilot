from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.data.markets import Market, get_market_config, normalize_tickers
from src.goals.assets import (
    build_selected_return_matrix,
    load_selected_assets,
    validate_minimum_history,
)
from src.goals.backtest import backtest_goal_portfolio
from src.goals.feasibility import (
    calculate_maximum_feasible_return,
    calculate_recommended_target_return,
)
from src.goals.goal_math import parse_horizon, format_horizon
from src.goals.planner import (
    evaluate_subsets_with_investment,
    SubsetInvestmentResult,
)
from src.goals.portfolio import GoalPortfolioResult
from src.goals.returns import analyze_stock, StockAnalysis, Trend
from src.goals.scenarios import build_scenario_set
from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.config import BacktestConfig
from src.data.market_data import MarketDataAdapter
from src.evaluation.metrics import evaluate_returns

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
BACKTESTS_DIR = RESULTS_DIR / "backtests"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="QuantPilot API",
    description="Quantitative Portfolio Management API",
    version="1.0.0",
    lifespan=lifespan,
)

local_frontend_origins = [
    "http://localhost:3000",
    "http://localhost:3002",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3002",
]
deployed_frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[*local_frontend_origins, *deployed_frontend_origins],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MarketConfigResponse(BaseModel):
    market: str
    name: str
    currency: str
    benchmark: str
    ticker_suffix: str


class EvaluateRequest(BaseModel):
    tickers: list[str]
    market: str = "us"
    lump_sum: float = 0.0
    monthly_sip: float = 0.0
    years: float
    minimum_observations: int = 252
    estimation_window: int = 252
    shrinkage_alpha: float = 0.5
    min_subset_size: int = 3
    max_subset_size: int | None = None
    max_position_weight: float = Field(default=0.25, gt=0, le=1)
    ranking_metric: Literal["low_risk", "max_future_value"] = "max_future_value"


class BacktestRequest(BaseModel):
    tickers: list[str]
    market: str = "us"
    weights: dict[str, float]
    cash_weight: float = 0.0
    expected_return: float
    expected_volatility: float
    target_return: float
    risk_tolerance: str
    maximum_allowed_volatility: float
    estimation_window: int = 252
    shrinkage_alpha: float = 0.5


class TrendRequest(BaseModel):
    tickers: list[str]
    market: str = "us"


def _convert_subset_result(r: SubsetInvestmentResult) -> dict[str, Any]:
    def _rpr_to_dict(rpr, rp_name: str) -> dict[str, Any] | None:
        if rpr is None:
            return None
        return {
            "risk_profile": rpr.risk_profile,
            "max_allowed_volatility": rpr.max_allowed_volatility,
            "max_feasible_return": rpr.max_feasible_return,
            "recommended_return": rpr.recommended_return,
            "expected_return": rpr.expected_return,
            "expected_volatility": rpr.expected_volatility,
            "sharpe_ratio": rpr.sharpe_ratio,
            "future_value_lump_sum": rpr.future_value_lump_sum,
            "future_value_monthly": rpr.future_value_monthly,
            "feasible": rpr.feasible,
            "warning": rpr.warning,
            "weights": rpr.weights.to_dict() if rpr.weights is not None else None,
            "cash_weight": rpr.cash_weight,
        }

    return {
        "tickers": list(r.tickers),
        "subset_size": r.subset_size,
        "max_weight": r.max_weight,
        "feasible": r.feasible,
        "error": r.error,
        "conservative": _rpr_to_dict(r.conservative, "conservative"),
        "moderate": _rpr_to_dict(r.moderate, "moderate"),
        "aggressive": _rpr_to_dict(r.aggressive, "aggressive"),
        "best_future_value": r.best_future_value,
        "extreme_return_warnings": r.extreme_return_warnings,
    }


def _convert_stock_analysis(a: StockAnalysis) -> dict[str, Any]:
    return {
        "ticker": a.ticker,
        "current_price": a.current_price,
        "multi_horizon": a.multi_horizon.to_dict(),
        "trend": a.trend.value,
        "trend_score": a.trend_score,
        "technical": a.technical.to_dict(),
    }


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "QuantPilot API"}


@app.get("/api/markets", response_model=list[MarketConfigResponse])
async def get_markets():
    configs = []
    for m in Market:
        cfg = get_market_config(m)
        configs.append(MarketConfigResponse(
            market=cfg.market.value,
            name=cfg.name,
            currency=cfg.currency,
            benchmark=cfg.benchmark,
            ticker_suffix=cfg.ticker_suffix,
        ))
    return configs


@app.post("/api/evaluate")
async def evaluate_subsets(req: EvaluateRequest):
    try:
        market = Market(req.market.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid market: {req.market}. Use 'us' or 'india'.")

    try:
        normalized = normalize_tickers(req.tickers, market)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if len(normalized) < req.min_subset_size:
        raise HTTPException(400, f"Need at least {req.min_subset_size} tickers, got {len(normalized)}")
    if len(normalized) > 15:
        raise HTTPException(400, "Maximum 15 tickers supported")
    if req.max_subset_size is not None and not req.min_subset_size <= req.max_subset_size <= len(normalized):
        raise HTTPException(
            400,
            f"max_subset_size must be between {req.min_subset_size} and {len(normalized)}",
        )

    if req.lump_sum <= 0 and req.monthly_sip <= 0:
        raise HTTPException(400, "At least one of lump_sum or monthly_sip must be positive")
    if req.years <= 0:
        raise HTTPException(400, "years must be positive")

    try:
        results = evaluate_subsets_with_investment(
            all_tickers=normalized,
            market=market,
            lump_sum=req.lump_sum,
            monthly_sip=req.monthly_sip,
            years=req.years,
            minimum_observations=req.minimum_observations,
            estimation_window=req.estimation_window,
            shrinkage_alpha=req.shrinkage_alpha,
            allow_download=True,
            min_subset_size=req.min_subset_size,
            ranking_metric=req.ranking_metric,
            max_subset_size=req.max_subset_size,
            max_position_weight=req.max_position_weight,
        )
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {str(e)}")

    return [_convert_subset_result(r) for r in results]


@app.post("/api/backtest")
async def run_backtest(req: BacktestRequest):
    try:
        market = Market(req.market.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid market: {req.market}")

    try:
        normalized = normalize_tickers(req.tickers, market)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        asset_data = load_selected_assets(
            tickers=normalized,
            market=market,
            allow_download=True,
        )
        returns = build_selected_return_matrix(asset_data)
        validate_minimum_history(returns, minimum_observations=252)

        portfolio = GoalPortfolioResult(
            weights=pd.Series(req.weights),
            expected_return=req.expected_return,
            expected_volatility=req.expected_volatility,
            target_return=req.target_return,
            risk_tolerance=req.risk_tolerance,
            maximum_allowed_volatility=req.maximum_allowed_volatility,
            feasible=True,
            cash_weight=req.cash_weight,
        )

        goal_backtest = backtest_goal_portfolio(
            returns=returns,
            portfolio=portfolio,
            rebalance_monthly=True,
        )

        comparison_tickers = [str(ticker).upper() for ticker in portfolio.weights.index]
        equal_weight_backtest = EqualWeightBacktester(config=BacktestConfig()).run(
            returns=returns.loc[:, comparison_tickers]
        )

        adapter = MarketDataAdapter(market)
        benchmark_data = adapter.load_benchmark(allow_download=True)
        benchmark_returns = benchmark_data["Close"].pct_change().dropna()

        comparison_returns = pd.concat(
            {
                "QuantPilot Portfolio": goal_backtest.returns,
                "Equal Weight": equal_weight_backtest.returns,
                "Benchmark": benchmark_returns,
            },
            axis=1,
        ).dropna()

        metrics = {}
        if len(comparison_returns) >= 30:
            metrics["QuantPilot Portfolio"] = evaluate_returns(comparison_returns["QuantPilot Portfolio"])
            metrics["Equal Weight"] = evaluate_returns(comparison_returns["Equal Weight"])
            metrics["Benchmark"] = evaluate_returns(comparison_returns["Benchmark"])

        equity = (1.0 + comparison_returns).cumprod()
        equity_curve = equity.copy()
        equity_curve.insert(
            0,
            "date",
            [pd.Timestamp(date).isoformat() for date in equity_curve.index],
        )

        return {
            "goal_returns": goal_backtest.returns.to_dict(),
            "goal_weights": goal_backtest.weights.to_dict(orient="records"),
            "equal_weight_returns": equal_weight_backtest.returns.to_dict(),
            "benchmark_returns": benchmark_returns.to_dict(),
            "comparison_returns": comparison_returns.to_dict(orient="records"),
            "equity_curve": equity_curve.to_dict(orient="records"),
            "metrics": metrics,
        }
    except Exception as e:
        raise HTTPException(500, f"Backtest failed: {str(e)}")


@app.post("/api/trend")
async def analyze_trends(req: TrendRequest):
    try:
        market = Market(req.market.lower())
    except ValueError:
        raise HTTPException(400, f"Invalid market: {req.market}")

    try:
        normalized = normalize_tickers(req.tickers, market)
    except ValueError as e:
        raise HTTPException(400, str(e))

    adapter = MarketDataAdapter(market)
    results = []

    for ticker in normalized:
        try:
            data = adapter.load_asset(ticker, allow_download=True)
            prices = data["Close"].dropna()
            if len(prices) >= 2:
                analysis = analyze_stock(prices, ticker)
                results.append(_convert_stock_analysis(analysis))
            else:
                results.append({"ticker": ticker, "error": "Insufficient price data"})
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})

    return results


def _get_portfolio_builder_path(data_dir: Path, filename: str, market: str) -> Path:
    """Get the path for portfolio builder data files, with market-specific naming."""
    # Try market-specific file first
    if market.lower() != "us":
        market_path = data_dir / f"quantpilot_v2_{filename}_{market}.csv"
        if market_path.exists():
            return market_path
    # Fallback to generic (US) file - try with and without quantpilot_v2_ prefix
    path_with_prefix = data_dir / f"quantpilot_v2_{filename}.csv"
    if path_with_prefix.exists():
        return path_with_prefix
    # Fallback to just the filename
    return data_dir / f"{filename}.csv"


@app.get("/api/portfolio-builder/comparison")
async def get_comparison(market: str = Query("us")):
    path = _get_portfolio_builder_path(COMPARISONS_DIR, "strategy_comparison", market)
    if not path.exists():
        raise HTTPException(404, f"Comparison data not found for market: {market}")
    df = pd.read_csv(path, index_col="strategy")
    return df.to_dict(orient="index")


@app.get("/api/portfolio-builder/returns")
async def get_returns(market: str = Query("us")):
    path = _get_portfolio_builder_path(BACKTESTS_DIR, "returns", market)
    if not path.exists():
        raise HTTPException(404, f"Returns data not found for market: {market}")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df.reset_index().to_dict(orient="records")


@app.get("/api/portfolio-builder/weights")
async def get_weights(market: str = Query("us")):
    path = _get_portfolio_builder_path(BACKTESTS_DIR, "weights", market)
    if not path.exists():
        raise HTTPException(404, f"Weights data not found for market: {market}")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df.reset_index().to_dict(orient="records")


@app.get("/api/portfolio-builder/turnover")
async def get_turnover(market: str = Query("us")):
    path = _get_portfolio_builder_path(BACKTESTS_DIR, "turnover", market)
    if not path.exists():
        raise HTTPException(404, f"Turnover data not found for market: {market}")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df.reset_index().to_dict(orient="records")


@app.get("/api/portfolio-builder/average-weights")
async def get_average_weights(market: str = Query("us")):
    path = _get_portfolio_builder_path(BACKTESTS_DIR, "average_weights", market)
    if not path.exists():
        raise HTTPException(404, f"Average weights data not found for market: {market}")
    df = pd.read_csv(path, index_col="ticker")
    return df.to_dict(orient="index")


@app.get("/api/portfolio-builder/effective-stocks")
async def get_effective_stocks(market: str = Query("us")):
    path = _get_portfolio_builder_path(BACKTESTS_DIR, "effective_stocks", market)
    if not path.exists():
        raise HTTPException(404, f"Effective stocks data not found for market: {market}")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df.reset_index().to_dict(orient="records")


@app.get("/api/parse-horizon")
async def parse_horizon_endpoint(horizon: str = Query(...)):
    try:
        years = parse_horizon(horizon)
        formatted = format_horizon(years)
        return {"years": years, "formatted": formatted}
    except Exception as e:
        raise HTTPException(400, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
