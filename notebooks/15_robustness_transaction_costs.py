"""
QuantPilot — Phase 8C
Robustness: Transaction Cost & Slippage Sensitivity

Research question:
    Does QuantPilot remain attractive when trading friction
    becomes more expensive?

Scenarios:
    - Frictionless: 0 bps transaction cost, 0 bps slippage
    - Low:          5 bps transaction cost, 2.5 bps slippage
    - Baseline:    10 bps transaction cost, 5 bps slippage
    - High:        20 bps transaction cost, 10 bps slippage
    - Very High:   50 bps transaction cost, 25 bps slippage

Fixed portfolio configuration:
    - Expected-return window: 60 days
    - Expected-return shrinkage alpha: 0.5
    - Covariance window: 60 days
    - Risk aversion: 10
    - Maximum position weight: 10%
    - Monthly rebalancing

Important:
    This experiment evaluates sensitivity to implementation costs.
    It does not select the final strategy based solely on historical
    performance.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# QuantPilot imports
# ---------------------------------------------------------------------

from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance
from src.models.expected_returns import calculate_expected_returns
from src.backtesting.engine import WalkForwardBacktester, BacktestConfig
from src.evaluation.metrics import (
    cumulative_return,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    maximum_drawdown,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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

EXPECTED_RETURN_WINDOW = 60
EXPECTED_RETURN_ALPHA = 0.5
COVARIANCE_WINDOW = 60

RISK_AVERSION = 10.0
MAX_WEIGHT = 0.10

# Costs are represented as decimal rates.
# 10 bps = 0.001
COST_SCENARIOS = [
    {
        "scenario": "frictionless",
        "transaction_cost_bps": 0.0,
        "slippage_bps": 0.0,
    },
    {
        "scenario": "low",
        "transaction_cost_bps": 5.0,
        "slippage_bps": 2.5,
    },
    {
        "scenario": "baseline",
        "transaction_cost_bps": 10.0,
        "slippage_bps": 5.0,
    },
    {
        "scenario": "high",
        "transaction_cost_bps": 20.0,
        "slippage_bps": 10.0,
    },
    {
        "scenario": "very_high",
        "transaction_cost_bps": 50.0,
        "slippage_bps": 25.0,
    },
]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "robustness_transaction_costs.csv"
)


# ---------------------------------------------------------------------
# Performance helper
# ---------------------------------------------------------------------

def calculate_performance_summary(
    returns: pd.Series,
) -> dict:
    """
    Calculate standard portfolio performance metrics.
    """

    returns = returns.dropna()

    if returns.empty:
        return {
            "cumulative_return": np.nan,
            "cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "sortino_ratio": np.nan,
            "maximum_drawdown": np.nan,
        }

    return {
        "cumulative_return": cumulative_return(returns),
        "cagr": cagr(returns),
        "annualized_volatility": annualized_volatility(
            returns
        ),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "maximum_drawdown": maximum_drawdown(returns),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("QUANTPILOT — PHASE 8C")
    print("ROBUSTNESS: TRANSACTION COST & SLIPPAGE SENSITIVITY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load market data
    # ---------------------------------------------------------------

    print("\nLoading market data...")

    price_data = load_universe(TICKERS)

    print("Market data loaded.")

    # ---------------------------------------------------------------
    # Calculate returns
    # ---------------------------------------------------------------

    print("\nCalculating returns...")

    returns = calculate_return_matrix(
        price_data,
        price_column="Close",
        method="simple",
    )

    print(f"Asset return matrix: {returns.shape}")

    print(
        f"Asset return period: "
        f"{returns.index.min().date()} → "
        f"{returns.index.max().date()}"
    )

    # ---------------------------------------------------------------
    # Calculate covariance matrices
    # ---------------------------------------------------------------

    print("\nCalculating rolling covariance...")

    covariance_matrices = rolling_covariance(
        returns,
        window=COVARIANCE_WINDOW,
    )

    print(
        f"Covariance matrices: "
        f"{len(covariance_matrices)}"
    )

    # ---------------------------------------------------------------
    # Calculate V2 expected returns
    # ---------------------------------------------------------------

    print("\nCalculating V2 expected returns...")

    expected_returns = calculate_expected_returns(
        returns,
        window=EXPECTED_RETURN_WINDOW,
        alpha=EXPECTED_RETURN_ALPHA,
    )

    # Remove incomplete rolling estimates.
    expected_returns = expected_returns.dropna(
        how="any"
    )

    print("V2 expected returns calculated.")

    # ---------------------------------------------------------------
    # Run cost scenarios
    # ---------------------------------------------------------------

    results = []

    print("\n" + "=" * 70)
    print("RUNNING TRANSACTION COST SCENARIOS")
    print("=" * 70)

    for scenario in COST_SCENARIOS:

        scenario_name = scenario["scenario"]

        transaction_cost_bps = (
            scenario["transaction_cost_bps"]
        )

        slippage_bps = scenario["slippage_bps"]

        transaction_cost_rate = (
            transaction_cost_bps / 10000.0
        )

        slippage_rate = (
            slippage_bps / 10000.0
        )

        print(
            f"\nRunning {scenario_name.upper()} scenario..."
        )

        print(
            f"  Transaction cost: "
            f"{transaction_cost_bps:.1f} bps"
        )

        print(
            f"  Slippage: "
            f"{slippage_bps:.1f} bps"
        )

        # -----------------------------------------------------------
        # Backtest configuration
        # -----------------------------------------------------------

        config = BacktestConfig(
            estimation_window=EXPECTED_RETURN_WINDOW,
            risk_aversion=RISK_AVERSION,
            max_weight=MAX_WEIGHT,
            transaction_cost_rate=transaction_cost_rate,
            slippage_rate=slippage_rate,
        )

        # -----------------------------------------------------------
        # Backtester
        # -----------------------------------------------------------

        backtester = WalkForwardBacktester(
            config=config
        )

        # -----------------------------------------------------------
        # Run
        # -----------------------------------------------------------

        result = backtester.run(
            returns=returns,
            expected_returns=expected_returns,
            covariance_matrices=covariance_matrices,
        )

        # -----------------------------------------------------------
        # Performance
        # -----------------------------------------------------------

        performance = calculate_performance_summary(
            result.returns
        )

        # -----------------------------------------------------------
        # Trading statistics
        # -----------------------------------------------------------

        turnover = result.turnover.dropna()

        if not turnover.empty:
            average_turnover = turnover.mean()
            total_turnover = turnover.sum()
        else:
            average_turnover = np.nan
            total_turnover = np.nan

        if not result.transaction_costs.empty:
            total_transaction_costs = (
                result.transaction_costs.sum()
            )
        else:
            total_transaction_costs = 0.0

        if not result.slippage.empty:
            total_slippage = result.slippage.sum()
        else:
            total_slippage = 0.0

        total_trading_friction = (
            total_transaction_costs
            + total_slippage
        )

        results.append(
            {
                "scenario": scenario_name,
                "transaction_cost_bps": (
                    transaction_cost_bps
                ),
                "slippage_bps": slippage_bps,
                "cumulative_return": performance[
                    "cumulative_return"
                ],
                "cagr": performance["cagr"],
                "annualized_volatility": performance[
                    "annualized_volatility"
                ],
                "sharpe_ratio": performance[
                    "sharpe_ratio"
                ],
                "sortino_ratio": performance[
                    "sortino_ratio"
                ],
                "maximum_drawdown": performance[
                    "maximum_drawdown"
                ],
                "average_turnover": average_turnover,
                "total_turnover": total_turnover,
                "transaction_costs": (
                    total_transaction_costs
                ),
                "slippage": total_slippage,
                "total_trading_friction": (
                    total_trading_friction
                ),
                "observations": len(result.returns),
                "rebalances": len(result.turnover),
            }
        )

        print(
            f"  CAGR: "
            f"{performance['cagr']:.2%}"
        )

        print(
            f"  Sharpe: "
            f"{performance['sharpe_ratio']:.3f}"
        )

        print(
            f"  Max Drawdown: "
            f"{performance['maximum_drawdown']:.2%}"
        )

        print(
            f"  Total trading friction: "
            f"{total_trading_friction:.4f}"
        )

    # ---------------------------------------------------------------
    # Results DataFrame
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRANSACTION COST ROBUSTNESS RESULTS")
    print("=" * 70)

    display_columns = [
        "scenario",
        "transaction_cost_bps",
        "slippage_bps",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "average_turnover",
        "transaction_costs",
        "slippage",
        "total_trading_friction",
    ]

    print(
        results_df[display_columns].to_string(
            index=False,
            formatters={
                "cagr": "{:.2%}".format,
                "annualized_volatility": "{:.2%}".format,
                "sharpe_ratio": "{:.3f}".format,
                "sortino_ratio": "{:.3f}".format,
                "maximum_drawdown": "{:.2%}".format,
                "average_turnover": "{:.2%}".format,
                "transaction_costs": "{:.4f}".format,
                "slippage": "{:.4f}".format,
                "total_trading_friction": "{:.4f}".format,
            },
        )
    )

    # ---------------------------------------------------------------
    # Performance degradation
    # ---------------------------------------------------------------

    baseline = results_df.loc[
        results_df["scenario"] == "baseline"
    ].iloc[0]

    frictionless = results_df.loc[
        results_df["scenario"] == "frictionless"
    ].iloc[0]

    print("\n" + "=" * 70)
    print("PERFORMANCE DEGRADATION FROM TRADING FRICTION")
    print("=" * 70)

    frictionless_cagr = frictionless["cagr"]

    for _, row in results_df.iterrows():

        cagr_loss = (
            frictionless_cagr
            - row["cagr"]
        )

        print(
            f"{row['scenario']:12s} | "
            f"CAGR: {row['cagr']:.2%} | "
            f"Loss vs frictionless: {cagr_loss:.2%} | "
            f"Friction: "
            f"{row['total_trading_friction']:.4f}"
        )

    # ---------------------------------------------------------------
    # Baseline comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("BASELINE COST IMPACT")
    print("=" * 70)

    print(
        f"\nFrictionless CAGR: "
        f"{frictionless['cagr']:.2%}"
    )

    print(
        f"Baseline CAGR:    "
        f"{baseline['cagr']:.2%}"
    )

    print(
        f"CAGR reduction:   "
        f"{frictionless['cagr'] - baseline['cagr']:.2%}"
    )

    print(
        f"\nFrictionless Sharpe: "
        f"{frictionless['sharpe_ratio']:.3f}"
    )

    print(
        f"Baseline Sharpe:    "
        f"{baseline['sharpe_ratio']:.3f}"
    )

    print(
        f"Sharpe reduction:   "
        f"{frictionless['sharpe_ratio'] - baseline['sharpe_ratio']:.3f}"
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nFixed portfolio parameters:")
    print(
        f"  Expected-return window: "
        f"{EXPECTED_RETURN_WINDOW}"
    )

    print(
        f"  Expected-return alpha: "
        f"{EXPECTED_RETURN_ALPHA}"
    )

    print(
        f"  Covariance window: "
        f"{COVARIANCE_WINDOW}"
    )

    print(
        f"  Risk aversion: "
        f"{RISK_AVERSION}"
    )

    print(
        f"  Maximum weight: "
        f"{MAX_WEIGHT:.0%}"
    )

    print("\nCost scenarios:")
    for scenario in COST_SCENARIOS:
        print(
            f"  {scenario['scenario']:12s}: "
            f"{scenario['transaction_cost_bps']:.1f} bps "
            f"transaction cost + "
            f"{scenario['slippage_bps']:.1f} bps slippage"
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "\nIMPORTANT: Transaction-cost sensitivity is a "
        "robustness analysis. Do not assume that the "
        "lowest-cost scenario represents realistic execution."
    )


if __name__ == "__main__":
    main()