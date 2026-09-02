"""
QuantPilot — Phase 8A
Robustness: Parameter Sensitivity

Research question:
    How sensitive is V2 QuantPilot's performance to its core
    portfolio parameters?

Parameters tested:
    - Expected-return estimation window: 60, 120, 252 days
    - Risk aversion: 5, 10, 20
    - Maximum position weight: 5%, 10%, 15%

Fixed:
    - Monthly rebalancing
    - Transaction cost: 10 bps
    - Slippage: 5 bps
    - Same 20-stock universe
    - Same rolling covariance methodology

Important:
    This experiment is for robustness analysis, not parameter
    selection. We do not select the best configuration as the
    final strategy based on these results.
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

ESTIMATION_WINDOWS = [60, 120, 252]
RISK_AVERSIONS = [5.0, 10.0, 20.0]
MAX_WEIGHTS = [0.05, 0.10, 0.15]

TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.0005

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# Helper: performance summary
# ---------------------------------------------------------------------

def calculate_performance_summary(returns: pd.Series) -> dict:
    """
    Calculate the main portfolio performance metrics.

    Returns
    -------
    dict
        cumulative return,
        CAGR,
        annualized volatility,
        Sharpe ratio,
        Sortino ratio,
        maximum drawdown.
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
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "maximum_drawdown": maximum_drawdown(returns),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("QUANTPILOT — PHASE 8A")
    print("ROBUSTNESS: PARAMETER SENSITIVITY")
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
        window=60,
    )

    print(f"Covariance matrices: {len(covariance_matrices)}")

    # ---------------------------------------------------------------
    # Run parameter grid
    # ---------------------------------------------------------------

    results = []

    total_experiments = (
        len(ESTIMATION_WINDOWS)
        * len(RISK_AVERSIONS)
        * len(MAX_WEIGHTS)
    )

    experiment_number = 0

    print("\n" + "=" * 70)
    print(f"RUNNING {total_experiments} CONFIGURATIONS")
    print("=" * 70)

    for estimation_window in ESTIMATION_WINDOWS:

        print(
            f"\nExpected-return window: "
            f"{estimation_window} days"
        )

        # -----------------------------------------------------------
        # Expected returns
        # -----------------------------------------------------------

        expected_returns = calculate_expected_returns(
            returns,
            window=estimation_window,
            alpha=0.5,
        )

        # Remove dates where rolling expected returns are incomplete.
        expected_returns = expected_returns.dropna(how="any")

        # Keep only dates available in the return matrix.
        common_index = returns.index.intersection(expected_returns.index)

        expected_returns = expected_returns.loc[common_index]

        for risk_aversion in RISK_AVERSIONS:

            for max_weight in MAX_WEIGHTS:

                experiment_number += 1

                print(
                    f"\n[{experiment_number}/{total_experiments}] "
                    f"window={estimation_window}, "
                    f"risk_aversion={risk_aversion}, "
                    f"max_weight={max_weight:.0%}"
                )

                # ---------------------------------------------------
                # Backtest configuration
                # ---------------------------------------------------

                config = BacktestConfig(
                    estimation_window=estimation_window,
                    risk_aversion=risk_aversion,
                    max_weight=max_weight,
                    transaction_cost_rate=TRANSACTION_COST_RATE,
                    slippage_rate=SLIPPAGE_RATE,
                )

                # ---------------------------------------------------
                # Backtest
                # ---------------------------------------------------

                backtester = WalkForwardBacktester(
                    config=config
                )

                result = backtester.run(
                    returns=returns,
                    expected_returns=expected_returns,
                    covariance_matrices=covariance_matrices,
                )

                # ---------------------------------------------------
                # Store performance
                # ---------------------------------------------------

                performance = calculate_performance_summary(
                    result.returns
                )

                # ---------------------------------------------------
                # Trading statistics
                # ---------------------------------------------------

                turnover = result.turnover.dropna()

                if not turnover.empty:
                    average_turnover = turnover.mean()
                    total_turnover = turnover.sum()
                else:
                    average_turnover = np.nan
                    total_turnover = np.nan

                transaction_costs = (
                    result.transaction_costs.sum()
                    if not result.transaction_costs.empty
                    else np.nan
                )

                slippage = (
                    result.slippage.sum()
                    if not result.slippage.empty
                    else np.nan
                )

                results.append(
                    {
                        "estimation_window": estimation_window,
                        "risk_aversion": risk_aversion,
                        "max_weight": max_weight,
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
                        "transaction_costs": transaction_costs,
                        "slippage": slippage,
                        "observations": len(result.returns),
                        "rebalances": len(result.turnover),
                    }
                )

    # ---------------------------------------------------------------
    # Results DataFrame
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------------
    # Print full results
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ROBUSTNESS RESULTS")
    print("=" * 70)

    display_columns = [
        "estimation_window",
        "risk_aversion",
        "max_weight",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "average_turnover",
        "transaction_costs",
        "slippage",
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
            },
        )
    )

    # ---------------------------------------------------------------
    # Best/worst configurations
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXTREME RESULTS")
    print("=" * 70)

    best_cagr = results_df.loc[
        results_df["cagr"].idxmax()
    ]

    worst_cagr = results_df.loc[
        results_df["cagr"].idxmin()
    ]

    best_sharpe = results_df.loc[
        results_df["sharpe_ratio"].idxmax()
    ]

    worst_drawdown = results_df.loc[
        results_df["maximum_drawdown"].idxmin()
    ]

    print("\nHighest CAGR:")
    print(
        f"  Window: {int(best_cagr['estimation_window'])}"
        f" | Risk aversion: {best_cagr['risk_aversion']}"
        f" | Max weight: {best_cagr['max_weight']:.0%}"
    )
    print(f"  CAGR: {best_cagr['cagr']:.2%}")

    print("\nLowest CAGR:")
    print(
        f"  Window: {int(worst_cagr['estimation_window'])}"
        f" | Risk aversion: {worst_cagr['risk_aversion']}"
        f" | Max weight: {worst_cagr['max_weight']:.0%}"
    )
    print(f"  CAGR: {worst_cagr['cagr']:.2%}")

    print("\nHighest Sharpe:")
    print(
        f"  Window: {int(best_sharpe['estimation_window'])}"
        f" | Risk aversion: {best_sharpe['risk_aversion']}"
        f" | Max weight: {best_sharpe['max_weight']:.0%}"
    )
    print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.3f}")

    print("\nWorst maximum drawdown:")
    print(
        f"  Window: {int(worst_drawdown['estimation_window'])}"
        f" | Risk aversion: {worst_drawdown['risk_aversion']}"
        f" | Max weight: {worst_drawdown['max_weight']:.0%}"
    )
    print(
        f"  Maximum drawdown: "
        f"{worst_drawdown['maximum_drawdown']:.2%}"
    )

    # ---------------------------------------------------------------
    # Aggregate by individual parameter
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE BY ESTIMATION WINDOW")
    print("=" * 70)

    window_summary = (
        results_df
        .groupby("estimation_window")
        .agg(
            cagr=("cagr", "mean"),
            volatility=("annualized_volatility", "mean"),
            sharpe=("sharpe_ratio", "mean"),
            sortino=("sortino_ratio", "mean"),
            max_drawdown=("maximum_drawdown", "mean"),
            turnover=("average_turnover", "mean"),
        )
        .reset_index()
    )

    print(
        window_summary.to_string(
            index=False,
            formatters={
                "cagr": "{:.2%}".format,
                "volatility": "{:.2%}".format,
                "sharpe": "{:.3f}".format,
                "sortino": "{:.3f}".format,
                "max_drawdown": "{:.2%}".format,
                "turnover": "{:.2%}".format,
            },
        )
    )

    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE BY RISK AVERSION")
    print("=" * 70)

    risk_summary = (
        results_df
        .groupby("risk_aversion")
        .agg(
            cagr=("cagr", "mean"),
            volatility=("annualized_volatility", "mean"),
            sharpe=("sharpe_ratio", "mean"),
            sortino=("sortino_ratio", "mean"),
            max_drawdown=("maximum_drawdown", "mean"),
            turnover=("average_turnover", "mean"),
        )
        .reset_index()
    )

    print(
        risk_summary.to_string(
            index=False,
            formatters={
                "cagr": "{:.2%}".format,
                "volatility": "{:.2%}".format,
                "sharpe": "{:.3f}".format,
                "sortino": "{:.3f}".format,
                "max_drawdown": "{:.2%}".format,
                "turnover": "{:.2%}".format,
            },
        )
    )

    print("\n" + "=" * 70)
    print("AVERAGE PERFORMANCE BY MAXIMUM WEIGHT")
    print("=" * 70)

    weight_summary = (
        results_df
        .groupby("max_weight")
        .agg(
            cagr=("cagr", "mean"),
            volatility=("annualized_volatility", "mean"),
            sharpe=("sharpe_ratio", "mean"),
            sortino=("sortino_ratio", "mean"),
            max_drawdown=("maximum_drawdown", "mean"),
            turnover=("average_turnover", "mean"),
        )
        .reset_index()
    )

    print(
        weight_summary.to_string(
            index=False,
            formatters={
                "cagr": "{:.2%}".format,
                "volatility": "{:.2%}".format,
                "sharpe": "{:.3f}".format,
                "sortino": "{:.3f}".format,
                "max_drawdown": "{:.2%}".format,
                "turnover": "{:.2%}".format,
            },
        )
    )

    # ---------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "robustness_parameter_sensitivity.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS EXPERIMENT COMPLETE")
    print("=" * 70)

    print(f"\nResults saved to:")
    print(output_path)

    print("\nConfigurations tested:")
    print(f"  Estimation windows: {ESTIMATION_WINDOWS}")
    print(f"  Risk aversions:     {RISK_AVERSIONS}")
    print(f"  Maximum weights:    {MAX_WEIGHTS}")

    print("\nFixed:")
    print("  Rebalancing:        Monthly")
    print("  Transaction costs:  10 bps")
    print("  Slippage:            5 bps")

    print(
        "\nIMPORTANT: These results are for robustness analysis. "
        "Do not select the best configuration as the final strategy "
        "solely because it has the highest historical performance."
    )


if __name__ == "__main__":
    main()