import pandas as pd
import yaml

from src.data.reader import load_universe
from src.data.benchmark_reader import load_benchmark

from src.features.returns import calculate_return_matrix
from src.features.statistics import rolling_mean
from src.features.covariance import rolling_covariance

from src.backtesting.config import BacktestConfig
from src.backtesting.engine import WalkForwardBacktester
from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.buy_and_hold import BuyAndHoldBacktester

from src.evaluation.alignment import align_return_series


# ============================================================
# Load Universe Configuration
# ============================================================

with open("configs/universe.yaml", "r") as file:
    config = yaml.safe_load(file)

tickers = config["universe"]["tickers"]


# ============================================================
# Load Data
# ============================================================

prices = load_universe(tickers)
spy_prices = load_benchmark("SPY")

returns = calculate_return_matrix(
    prices,
    price_column="Close",
    method="simple",
)

spy_returns = calculate_return_matrix(
    {"SPY": spy_prices},
    price_column="Close",
    method="simple",
)["SPY"]


# ============================================================
# Rolling Features
# ============================================================

expected_returns = rolling_mean(
    returns,
    window=60,
)

covariance_matrices = rolling_covariance(
    returns,
    window=60,
)


# ============================================================
# Run QuantPilot
# ============================================================

backtest_config = BacktestConfig()

quantpilot = WalkForwardBacktester(
    config=backtest_config,
)

quantpilot_result = quantpilot.run(
    returns=returns,
    expected_returns=expected_returns,
    covariance_matrices=covariance_matrices,
)


# ============================================================
# Run Equal-Weight
# ============================================================

equal_weight = EqualWeightBacktester(
    config=backtest_config,
)

equal_weight_result = equal_weight.run(
    returns=returns,
    rebalance_dates=quantpilot_result.weights.index,
)


# ============================================================
# Run Buy-and-Hold
# ============================================================

buy_hold = BuyAndHoldBacktester()

buy_hold_result = buy_hold.run(
    returns
)


# ============================================================
# Align All Strategies
# ============================================================

quantpilot_returns, equal_weight_returns = align_return_series(
    quantpilot_result.returns,
    equal_weight_result.returns,
)

quantpilot_returns, buy_hold_returns = align_return_series(
    quantpilot_returns,
    buy_hold_result.returns,
)

quantpilot_returns, spy_returns = align_return_series(
    quantpilot_returns,
    spy_returns,
)


# ============================================================
# Explicit Common Index
# ============================================================

common_index = (
    quantpilot_returns.index
    .intersection(equal_weight_returns.index)
    .intersection(buy_hold_returns.index)
    .intersection(spy_returns.index)
)

quantpilot_returns = quantpilot_returns.loc[common_index]
equal_weight_returns = equal_weight_returns.loc[common_index]
buy_hold_returns = buy_hold_returns.loc[common_index]
spy_returns = spy_returns.loc[common_index]


# ============================================================
# Build Return DataFrame
# ============================================================

daily_returns = pd.DataFrame(
    {
        "QuantPilot": quantpilot_returns,
        "Equal-Weight": equal_weight_returns,
        "Buy-and-Hold": buy_hold_returns,
        "SPY": spy_returns,
    }
)


# ============================================================
# Annual Performance
# ============================================================

annual_returns = (
    (1 + daily_returns)
    .groupby(daily_returns.index.year)
    .prod()
    - 1
)

annual_returns.index.name = "Year"


# ============================================================
# Strategy Differences
# ============================================================

annual_returns["QP_vs_EQ"] = (
    annual_returns["QuantPilot"]
    - annual_returns["Equal-Weight"]
)

annual_returns["QP_vs_SPY"] = (
    annual_returns["QuantPilot"]
    - annual_returns["SPY"]
)


# ============================================================
# Print Annual Performance
# ============================================================

print("\nAnnual Performance")
print("==================")

print(
    annual_returns.round(4).to_string()
)


# ============================================================
# Win/Loss Analysis
# ============================================================

qp_vs_eq = annual_returns["QP_vs_EQ"]
qp_vs_spy = annual_returns["QP_vs_SPY"]


print("\nQuantPilot vs Equal-Weight")
print("==========================")

print(
    f"Years QuantPilot outperformed: "
    f"{(qp_vs_eq > 0).sum()}"
)

print(
    f"Years QuantPilot underperformed: "
    f"{(qp_vs_eq < 0).sum()}"
)

print(
    f"Average annual advantage: "
    f"{qp_vs_eq.mean():.4f}"
)

print(
    f"Best year vs Equal-Weight: "
    f"{qp_vs_eq.max():.4f}"
)

print(
    f"Worst year vs Equal-Weight: "
    f"{qp_vs_eq.min():.4f}"
)


print("\nQuantPilot vs SPY")
print("=================")

print(
    f"Years QuantPilot outperformed: "
    f"{(qp_vs_spy > 0).sum()}"
)

print(
    f"Years QuantPilot underperformed: "
    f"{(qp_vs_spy < 0).sum()}"
)

print(
    f"Average annual advantage: "
    f"{qp_vs_spy.mean():.4f}"
)

print(
    f"Best year vs SPY: "
    f"{qp_vs_spy.max():.4f}"
)

print(
    f"Worst year vs SPY: "
    f"{qp_vs_spy.min():.4f}"
)


# ============================================================
# Best and Worst Years
# ============================================================

print("\nBest Years for QuantPilot vs Equal-Weight")
print("==========================================")

print(
    qp_vs_eq
    .sort_values(ascending=False)
    .head(5)
    .round(4)
    .to_string()
)


print("\nWorst Years for QuantPilot vs Equal-Weight")
print("==========================================")

print(
    qp_vs_eq
    .sort_values()
    .head(5)
    .round(4)
    .to_string()
)


# ============================================================
# Evaluation Period
# ============================================================

print("\nEvaluation Period")
print("=================")

print(
    f"Observations: {len(common_index)}"
)

print(
    f"Start: {common_index[0].date()}"
)

print(
    f"End: {common_index[-1].date()}"
)