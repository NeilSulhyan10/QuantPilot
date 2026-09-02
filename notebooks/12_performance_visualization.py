"""
QuantPilot — Performance Visualization

Generates:
1. Cumulative wealth curves
2. Drawdown curves
3. 252-day rolling Sharpe ratios
4. Annual return comparison

All strategies are evaluated over the same common period.

Primary ML specification:
    Training window = 252 trading days
    Forecast horizon = 21 trading days
    Ridge alpha = 1.0

V2 specification:
    Historical mean window = 60 trading days
    Shrinkage alpha = 0.5
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
import pandas as pd


# ============================================================
# Project path
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# QuantPilot imports
# ============================================================

from src.data.reader import load_universe
from src.data.benchmark_reader import load_benchmark

from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance

from src.models.expected_returns import (
    calculate_expected_returns,
)

from src.models.ml_expected_returns import (
    calculate_ml_expected_returns,
)

from src.backtesting.engine import (
    BacktestConfig,
    WalkForwardBacktester,
)

from src.backtesting.ml_engine import (
    MLWalkForwardBacktester,
)

from src.backtesting.benchmark import (
    EqualWeightBacktester,
)

from src.backtesting.buy_and_hold import (
    BuyAndHoldBacktester,
)

def calculate_performance_summary(returns: pd.Series) -> dict:
    """
    Calculate the main performance metrics used by QuantPilot.
    """

    returns = returns.dropna()

    if returns.empty:
        raise ValueError(
            "Cannot evaluate an empty return series."
        )

    periods_per_year = 252

    cumulative_return = (
        (1.0 + returns).prod() - 1.0
    )

    years = len(returns) / periods_per_year

    cagr = (
        (1.0 + cumulative_return)
        ** (1.0 / years)
        - 1.0
    )

    annualized_volatility = (
        returns.std()
        * (periods_per_year ** 0.5)
    )

    sharpe_ratio = (
        returns.mean()
        / returns.std()
        * (periods_per_year ** 0.5)
    )

    downside_returns = returns[
        returns < 0
    ]

    if len(downside_returns) > 0:
        downside_deviation = (
            downside_returns.std()
            * (periods_per_year ** 0.5)
        )

        sortino_ratio = (
            returns.mean()
            * periods_per_year
            / downside_deviation
        )
    else:
        sortino_ratio = float("nan")

    wealth = (
        1.0 + returns
    ).cumprod()

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max - 1.0
    )

    maximum_drawdown = drawdown.min()

    return {
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "maximum_drawdown": maximum_drawdown,
    }


# ============================================================
# Configuration
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AVGO",
    "GOOGL",
    "AMZN",
    "META",
    "JPM",
    "V",
    "MA",
    "JNJ",
    "UNH",
    "XOM",
    "CVX",
    "PG",
    "KO",
    "COST",
    "CAT",
    "WMT",
    "HD",
]

COVARIANCE_WINDOW = 60

# V2
V2_RETURN_WINDOW = 60
V2_SHRINKAGE_ALPHA = 0.5

# ML
ML_TRAIN_WINDOW = 252
ML_HORIZON = 21
ML_RIDGE_ALPHA = 1.0

# Portfolio
RISK_AVERSION = 10.0
MAX_WEIGHT = 0.10

# Trading costs
TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.0005


# ============================================================
# Load data
# ============================================================

print("\nLoading market data...")

price_data = load_universe(TICKERS)

spy_data = load_benchmark("SPY")

print("Market data loaded.")


# ============================================================
# Calculate asset returns
# ============================================================

print("\nCalculating returns...")

returns = calculate_return_matrix(
    price_data,
    price_column="Close",
    method="simple",
)

spy_returns = spy_data["Close"].pct_change().dropna()

print(f"Asset return matrix: {returns.shape}")
print(
    f"Asset return period: "
    f"{returns.index[0].date()} → "
    f"{returns.index[-1].date()}"
)


# ============================================================
# Calculate rolling covariance matrices
# ============================================================

print("\nCalculating rolling covariance...")

covariance_matrices = rolling_covariance(
    returns,
    window=COVARIANCE_WINDOW,
)

print(
    f"Covariance matrices: "
    f"{len(covariance_matrices)}"
)


# ============================================================
# V2 Expected Returns
# ============================================================

print("\nCalculating V2 expected returns...")

expected_returns_v2 = calculate_expected_returns(
    returns,
    window=V2_RETURN_WINDOW,
    alpha=V2_SHRINKAGE_ALPHA,
)

print("V2 expected returns calculated.")


# ============================================================
# ML Expected Returns
# ============================================================

print("\nCalculating ML expected returns...")

ml_expected_returns = calculate_ml_expected_returns(
    returns,
    min_train_size=ML_TRAIN_WINDOW,
    horizon=ML_HORIZON,
    alpha=ML_RIDGE_ALPHA,
)

print("ML expected returns calculated.")


# ============================================================
# Backtest configuration
# ============================================================

config = BacktestConfig(
    estimation_window=COVARIANCE_WINDOW,
    risk_aversion=RISK_AVERSION,
    max_weight=MAX_WEIGHT,
    transaction_cost_rate=TRANSACTION_COST_RATE,
    slippage_rate=SLIPPAGE_RATE,
)


# ============================================================
# Run V2 QuantPilot
# ============================================================

print("\nRunning V2 QuantPilot backtest...")

v2_backtester = WalkForwardBacktester(
    config=config,
)

v2_result = v2_backtester.run(
    returns=returns,
    expected_returns=expected_returns_v2,
    covariance_matrices=covariance_matrices,
)

print("V2 backtest complete.")


# ============================================================
# Run ML QuantPilot
# ============================================================

print("\nRunning ML QuantPilot backtest...")

ml_backtester = MLWalkForwardBacktester(
    config=config,
)

ml_result = ml_backtester.run_ml(
    returns=returns,
    ml_expected_returns=ml_expected_returns,
    covariance_matrices=covariance_matrices,
)

print("ML backtest complete.")


# ============================================================
# Equal Weight benchmark
# ============================================================

print("\nRunning Equal Weight benchmark...")

equal_weight_backtester = EqualWeightBacktester(
    config=config,
)

equal_weight_result = equal_weight_backtester.run(
    returns
)

print("Equal Weight backtest complete.")


# ============================================================
# Buy & Hold benchmark
# ============================================================

print("\nRunning Buy & Hold benchmark...")

buy_hold_backtester = BuyAndHoldBacktester()

buy_hold_result = buy_hold_backtester.run(
    returns
)

print("Buy & Hold backtest complete.")


# ============================================================
# Build strategy return dataframe
# ============================================================

strategy_returns = pd.DataFrame(
    {
        "ML QuantPilot": ml_result.returns,
        "V2 QuantPilot": v2_result.returns,
        "Equal Weight": equal_weight_result.returns,
        "Buy & Hold": buy_hold_result.returns,
        "SPY": spy_returns,
    }
)


# ============================================================
# Common evaluation period
# ============================================================

strategy_returns = strategy_returns.dropna(
    how="any"
)

if strategy_returns.empty:
    raise ValueError(
        "No common evaluation period exists."
    )


evaluation_start = strategy_returns.index[0]
evaluation_end = strategy_returns.index[-1]

print("\n" + "=" * 70)
print("COMMON EVALUATION PERIOD")
print("=" * 70)

print(
    f"Start: {evaluation_start.date()}"
)

print(
    f"End:   {evaluation_end.date()}"
)

print(
    f"Observations: {len(strategy_returns)}"
)


# ============================================================
# Performance evaluation
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

performance = {}

for strategy in strategy_returns.columns:

    performance[strategy] = (
        calculate_performance_summary(
            strategy_returns[strategy]
        )
    )


performance_df = pd.DataFrame(
    performance
).T


print(
    performance_df.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# Trading statistics
# ============================================================

print("\n" + "=" * 70)
print("TRADING STATISTICS")
print("=" * 70)

print(
    f"ML average turnover: "
    f"{ml_result.turnover.mean():.4f}"
)

print(
    f"ML transaction costs: "
    f"{ml_result.transaction_costs.sum():.6f}"
)

print(
    f"ML slippage: "
    f"{ml_result.slippage.sum():.6f}"
)

print(
    f"ML rebalances: "
    f"{len(ml_result.turnover)}"
)


# ============================================================
# 1. Cumulative Wealth
# ============================================================

wealth = (
    1.0 + strategy_returns
).cumprod()


plt.figure(figsize=(13, 7))

for column in wealth.columns:

    plt.plot(
        wealth.index,
        wealth[column],
        label=column,
        linewidth=1.5,
    )


plt.title(
    "QuantPilot — Cumulative Wealth"
)

plt.xlabel("Date")

plt.ylabel(
    "Growth of $1"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "cumulative_wealth.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 2. Drawdown
# ============================================================

drawdown = (
    wealth
    / wealth.cummax()
    - 1.0
)


plt.figure(figsize=(13, 7))

for column in drawdown.columns:

    plt.plot(
        drawdown.index,
        drawdown[column],
        label=column,
        linewidth=1.5,
    )


plt.title(
    "QuantPilot — Drawdown"
)

plt.xlabel("Date")

plt.ylabel(
    "Drawdown"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "drawdown.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 3. 252-Day Rolling Sharpe
# ============================================================

rolling_mean = (
    strategy_returns
    .rolling(window=252)
    .mean()
)

rolling_std = (
    strategy_returns
    .rolling(window=252)
    .std()
)

rolling_sharpe = (
    rolling_mean
    / rolling_std
) * (252 ** 0.5)


plt.figure(figsize=(13, 7))

for column in rolling_sharpe.columns:

    plt.plot(
        rolling_sharpe.index,
        rolling_sharpe[column],
        label=column,
        linewidth=1.5,
    )


plt.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.title(
    "QuantPilot — 252-Day Rolling Sharpe Ratio"
)

plt.xlabel("Date")

plt.ylabel(
    "Rolling Sharpe"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "rolling_sharpe_252d.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 4. Annual Returns
# ============================================================

annual_returns = (
    (1.0 + strategy_returns)
    .groupby(
        strategy_returns.index.year
    )
    .prod()
    - 1.0
)


print("\n" + "=" * 70)
print("ANNUAL RETURNS")
print("=" * 70)

print(
    annual_returns.to_string(
        float_format=lambda x: f"{x:.2%}"
    )
)


plt.figure(figsize=(14, 7))

annual_returns.plot(
    kind="bar",
    ax=plt.gca(),
)

plt.title(
    "QuantPilot — Annual Returns"
)

plt.xlabel("Year")

plt.ylabel(
    "Annual Return"
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

plt.legend(
    title="Strategy"
)

plt.tight_layout()

plt.savefig(OUTPUT_DIR / "annual_returns.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print(
    f"Evaluation period: "
    f"{evaluation_start.date()} → "
    f"{evaluation_end.date()}"
)

print(
    f"Observations: "
    f"{len(strategy_returns)}"
)

print("\nStrategies:")
for strategy in strategy_returns.columns:
    print(f"  - {strategy}")

print("\nML specification:")
print(
    f"  Training window: {ML_TRAIN_WINDOW}"
)
print(
    f"  Forecast horizon: {ML_HORIZON}"
)
print(
    f"  Ridge alpha: {ML_RIDGE_ALPHA}"
)

print("\nVisualization phase complete.")