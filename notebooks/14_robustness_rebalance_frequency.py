"""
QuantPilot — Phase 8B
Robustness: Rebalance Frequency

Research question:
    Does QuantPilot's performance remain robust when the portfolio
    is rebalanced less frequently?

Frequencies tested:
    - Monthly
    - Quarterly
    - Semiannual
    - Annual

Fixed:
    - Expected-return window: 60 days
    - Expected-return shrinkage alpha: 0.5
    - Covariance window: 60 days
    - Risk aversion: 10
    - Maximum position weight: 10%
    - Transaction cost: 10 bps
    - Slippage: 5 bps

Important:
    This experiment studies the effect of trading frequency.
    It does not select a frequency solely because it produced
    the best historical return.
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

TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.0005

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------
# Performance helper
# ---------------------------------------------------------------------

def calculate_performance_summary(returns: pd.Series) -> dict:
    """
    Calculate portfolio performance metrics.
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
# Rebalance schedule
# ---------------------------------------------------------------------

def get_rebalance_dates(
    returns: pd.DataFrame,
    frequency: str,
) -> pd.DatetimeIndex:
    """
    Generate rebalance dates from the available trading dates.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns.

    frequency : str
        One of:
            monthly
            quarterly
            semiannual
            annual

    Returns
    -------
    pd.DatetimeIndex
    """

    frequency = frequency.lower()

    if frequency == "monthly":
        period = returns.index.to_period("M")

    elif frequency == "quarterly":
        period = returns.index.to_period("Q")

    elif frequency == "semiannual":
        # Convert dates into half-year periods.
        period = (
            returns.index.year.astype(str)
            + "-"
            + np.where(returns.index.month <= 6, "H1", "H2")
        )

        period = pd.Index(period)

    elif frequency == "annual":
        period = returns.index.to_period("Y")

    else:
        raise ValueError(
            "frequency must be one of: "
            "monthly, quarterly, semiannual, annual"
        )

    # Pick the final available trading day of each period.
    rebalance_dates = (
        pd.Series(returns.index, index=returns.index)
        .groupby(period)
        .last()
    )

    return pd.DatetimeIndex(rebalance_dates.values)


# ---------------------------------------------------------------------
# Frequency-aware backtester
# ---------------------------------------------------------------------

class FrequencyBacktester(WalkForwardBacktester):
    """
    Walk-forward backtester with configurable rebalance frequency.

    The parent backtester normally uses its built-in monthly schedule.
    This class overrides the schedule so that the robustness experiment
    can compare different rebalance frequencies.
    """

    def __init__(
        self,
        config: BacktestConfig,
        frequency: str,
    ):
        super().__init__(config=config)

        self.frequency = frequency

    def _get_rebalance_dates(
        self,
        returns,
        expected_returns,
        covariance_matrices,
    ):
        """
        Return valid rebalance dates for the requested frequency.
        """

        schedule = get_rebalance_dates(
            returns=returns,
            frequency=self.frequency,
        )

        expected_index = expected_returns.index

        covariance_index = pd.DatetimeIndex(
            covariance_matrices.keys()
        )

        valid_dates = (
            schedule
            .intersection(expected_index)
            .intersection(covariance_index)
        )

        return valid_dates.sort_values()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("QUANTPILOT — PHASE 8B")
    print("ROBUSTNESS: REBALANCE FREQUENCY")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print("\nLoading market data...")

    price_data = load_universe(TICKERS)

    print("Market data loaded.")

    # ---------------------------------------------------------------
    # Returns
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
    # Covariance
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
    # Expected returns
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
    # Frequencies
    # ---------------------------------------------------------------

    frequencies = [
        "monthly",
        "quarterly",
        "semiannual",
        "annual",
    ]

    results = []

    print("\n" + "=" * 70)
    print("RUNNING REBALANCE FREQUENCY EXPERIMENT")
    print("=" * 70)

    for frequency in frequencies:

        print(
            f"\nRunning {frequency.upper()} "
            f"rebalance strategy..."
        )

        # -----------------------------------------------------------
        # Backtest configuration
        # -----------------------------------------------------------

        config = BacktestConfig(
            estimation_window=EXPECTED_RETURN_WINDOW,
            risk_aversion=RISK_AVERSION,
            max_weight=MAX_WEIGHT,
            transaction_cost_rate=TRANSACTION_COST_RATE,
            slippage_rate=SLIPPAGE_RATE,
        )

        backtester = FrequencyBacktester(
            config=config,
            frequency=frequency,
        )

        # -----------------------------------------------------------
        # Backtest
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
                "frequency": frequency,
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
            f"  Average Turnover: "
            f"{average_turnover:.2%}"
        )

    # ---------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("REBALANCE FREQUENCY RESULTS")
    print("=" * 70)

    display_columns = [
        "frequency",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "average_turnover",
        "transaction_costs",
        "slippage",
        "rebalances",
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
    # Frequency comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRADING COST COMPARISON")
    print("=" * 70)

    for _, row in results_df.iterrows():

        total_cost = (
            row["transaction_costs"]
            + row["slippage"]
        )

        print(
            f"{row['frequency']:12s} | "
            f"Rebalances: {int(row['rebalances']):3d} | "
            f"Turnover: {row['average_turnover']:.2%} | "
            f"Trading friction: {total_cost:.4f}"
        )

    # ---------------------------------------------------------------
    # Best metrics
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXTREME RESULTS")
    print("=" * 70)

    best_cagr = results_df.loc[
        results_df["cagr"].idxmax()
    ]

    best_sharpe = results_df.loc[
        results_df["sharpe_ratio"].idxmax()
    ]

    lowest_turnover = results_df.loc[
        results_df["average_turnover"].idxmin()
    ]

    best_drawdown = results_df.loc[
        results_df["maximum_drawdown"].idxmax()
    ]

    print("\nHighest CAGR:")
    print(
        f"  {best_cagr['frequency']}: "
        f"{best_cagr['cagr']:.2%}"
    )

    print("\nHighest Sharpe:")
    print(
        f"  {best_sharpe['frequency']}: "
        f"{best_sharpe['sharpe_ratio']:.3f}"
    )

    print("\nLowest turnover:")
    print(
        f"  {lowest_turnover['frequency']}: "
        f"{lowest_turnover['average_turnover']:.2%}"
    )

    print("\nBest maximum drawdown:")
    print(
        f"  {best_drawdown['frequency']}: "
        f"{best_drawdown['maximum_drawdown']:.2%}"
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "robustness_rebalance_frequency.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("ROBUSTNESS EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nFixed parameters:")
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

    print(
        f"  Transaction cost: "
        f"{TRANSACTION_COST_RATE:.2%}"
    )

    print(
        f"  Slippage: "
        f"{SLIPPAGE_RATE:.2%}"
    )

    print("\nResults saved to:")
    print(output_path)

    print(
        "\nIMPORTANT: Do not select a rebalance frequency "
        "solely because it produced the highest historical "
        "performance. Evaluate the trade-off between return, "
        "risk, turnover, and transaction costs."
    )


if __name__ == "__main__":
    main()