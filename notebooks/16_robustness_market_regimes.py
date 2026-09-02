"""
QuantPilot — Phase 8D
Robustness: Market Regime Analysis

Research question:
    How does QuantPilot behave across different market environments?

Regimes:
    1. 2018 Correction / Bear Market
    2. 2020 COVID Crash & Recovery
    3. 2022 Bear Market
    4. 2023-2024 Strong Equity Market
    5. 2025-2026 Recent Period

Strategies:
    - V2 QuantPilot
    - Equal Weight
    - Buy & Hold
    - SPY

Fixed V2 configuration:
    - Expected-return window: 60 days
    - Expected-return shrinkage alpha: 0.5
    - Covariance window: 60 days
    - Risk aversion: 10
    - Maximum position weight: 10%
    - Monthly rebalancing
    - Transaction cost: 10 bps
    - Slippage: 5 bps

Important:
    Regime dates are fixed in advance and are not selected based
    on strategy performance.

    This experiment is descriptive robustness analysis, not a
    claim that the strategy predicts market regimes.
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
from src.data.benchmark_reader import load_benchmark
from src.data.benchmark_loader import download_benchmark
from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance
from src.models.expected_returns import calculate_expected_returns
from src.backtesting.engine import (
    WalkForwardBacktester,
    BacktestConfig,
)
from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.buy_and_hold import BuyAndHoldBacktester
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

# ---------------------------------------------------------------------
# Fixed regime definitions
#
# These dates are chosen as economically meaningful historical
# periods, not because they produce favorable strategy results.
# ---------------------------------------------------------------------

REGIMES = {
    "2018_correction": (
        "2018-01-01",
        "2018-12-31",
    ),
    "2020_covid": (
        "2020-01-01",
        "2020-12-31",
    ),
    "2022_bear_market": (
        "2022-01-01",
        "2022-12-31",
    ),
    "2023_2024_bull": (
        "2023-01-01",
        "2024-12-31",
    ),
    "2025_2026_recent": (
        "2025-01-01",
        "2026-08-27",
    ),
}

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "robustness_market_regimes.csv"
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
            "observations": 0,
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
        "observations": len(returns),
    }


# ---------------------------------------------------------------------
# Slice returns to a fixed regime
# ---------------------------------------------------------------------

def slice_regime(
    returns: pd.Series,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Extract returns belonging to a fixed regime.
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    return returns.loc[
        (returns.index >= start)
        & (returns.index <= end)
    ].dropna()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("QUANTPILOT — PHASE 8D")
    print("ROBUSTNESS: MARKET REGIME ANALYSIS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load market data
    # ---------------------------------------------------------------

    print("\nLoading market data...")

    price_data = load_universe(TICKERS)

    print("Market data loaded.")

    # ---------------------------------------------------------------
    # Calculate asset returns
    # ---------------------------------------------------------------

    print("\nCalculating asset returns...")

    returns = calculate_return_matrix(
        price_data,
        price_column="Close",
        method="simple",
    )

    print(
        f"Asset return matrix: "
        f"{returns.shape}"
    )

    print(
        f"Asset return period: "
        f"{returns.index.min().date()} → "
        f"{returns.index.max().date()}"
    )

    # ---------------------------------------------------------------
    # Rolling covariance
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

    expected_returns = expected_returns.dropna(
        how="any"
    )

    print("V2 expected returns calculated.")

    # ---------------------------------------------------------------
    # V2 backtest
    # ---------------------------------------------------------------

    print("\nRunning V2 QuantPilot backtest...")

    config = BacktestConfig(
        estimation_window=EXPECTED_RETURN_WINDOW,
        risk_aversion=RISK_AVERSION,
        max_weight=MAX_WEIGHT,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )

    quantpilot_backtester = WalkForwardBacktester(
        config=config
    )

    quantpilot_result = quantpilot_backtester.run(
        returns=returns,
        expected_returns=expected_returns,
        covariance_matrices=covariance_matrices,
    )

    print("V2 QuantPilot backtest complete.")

    # ---------------------------------------------------------------
    # Equal Weight
    # ---------------------------------------------------------------

    print("\nRunning Equal Weight benchmark...")

    equal_weight_backtester = EqualWeightBacktester(
        config=config
    )

    equal_weight_result = equal_weight_backtester.run(
        returns
    )

    print("Equal Weight backtest complete.")

    # ---------------------------------------------------------------
    # Buy & Hold
    # ---------------------------------------------------------------

    print("\nRunning Buy & Hold benchmark...")

    buy_hold_backtester = BuyAndHoldBacktester()

    buy_hold_result = buy_hold_backtester.run(
        returns
    )

    print("Buy & Hold backtest complete.")

    # ---------------------------------------------------------------
    # SPY
    # ---------------------------------------------------------------

    print("\nLoading SPY benchmark...")

    try:
        spy_data = load_benchmark("SPY")

    except FileNotFoundError:
        print(
            "SPY processed data not found. "
            "Downloading benchmark..."
        )

        download_benchmark(
            ticker="SPY",
            start_date="2015-01-01",
        )

        spy_data = load_benchmark("SPY")

    spy_returns = spy_data["Close"].pct_change().dropna()

    print(
        f"SPY return period: "
        f"{spy_returns.index.min().date()} → "
        f"{spy_returns.index.max().date()}"
    )

    # ---------------------------------------------------------------
    # Align strategy return series
    # ---------------------------------------------------------------

    strategy_returns = {
        "V2 QuantPilot": quantpilot_result.returns,
        "Equal Weight": equal_weight_result.returns,
        "Buy & Hold": buy_hold_result.returns,
        "SPY": spy_returns,
    }

    # ---------------------------------------------------------------
    # Determine common evaluation period
    # ---------------------------------------------------------------

    common_start = max(
        series.dropna().index.min()
        for series in strategy_returns.values()
    )

    common_end = min(
        series.dropna().index.max()
        for series in strategy_returns.values()
    )

    print("\n" + "=" * 70)
    print("COMMON EVALUATION PERIOD")
    print("=" * 70)

    print(
        f"Start: {common_start.date()}"
    )

    print(
        f"End:   {common_end.date()}"
    )

    # ---------------------------------------------------------------
    # Align all returns
    # ---------------------------------------------------------------

    aligned_returns = {}

    for name, series in strategy_returns.items():

        aligned_returns[name] = series.loc[
            (series.index >= common_start)
            & (series.index <= common_end)
        ].dropna()

    # ---------------------------------------------------------------
    # Regime analysis
    # ---------------------------------------------------------------

    results = []

    print("\n" + "=" * 70)
    print("MARKET REGIME ANALYSIS")
    print("=" * 70)

    for regime_name, (
        regime_start,
        regime_end,
    ) in REGIMES.items():

        print("\n" + "-" * 70)

        print(
            f"{regime_name.upper()}"
        )

        print(
            f"Period: "
            f"{regime_start} → {regime_end}"
        )

        # -----------------------------------------------------------
        # Analyze each strategy
        # -----------------------------------------------------------

        regime_results = {}

        for strategy_name, series in aligned_returns.items():

            regime_returns = slice_regime(
                series,
                regime_start,
                regime_end,
            )

            if regime_returns.empty:
                print(
                    f"  {strategy_name}: "
                    "No observations"
                )
                continue

            performance = (
                calculate_performance_summary(
                    regime_returns
                )
            )

            regime_results[strategy_name] = performance

            results.append(
                {
                    "regime": regime_name,
                    "start_date": regime_start,
                    "end_date": regime_end,
                    "strategy": strategy_name,
                    "observations": performance[
                        "observations"
                    ],
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
                }
            )

            print(
                f"\n  {strategy_name}"
            )

            print(
                f"    Cumulative return: "
                f"{performance['cumulative_return']:.2%}"
            )

            print(
                f"    CAGR: "
                f"{performance['cagr']:.2%}"
            )

            print(
                f"    Volatility: "
                f"{performance['annualized_volatility']:.2%}"
            )

            print(
                f"    Sharpe: "
                f"{performance['sharpe_ratio']:.3f}"
            )

            print(
                f"    Sortino: "
                f"{performance['sortino_ratio']:.3f}"
            )

            print(
                f"    Max Drawdown: "
                f"{performance['maximum_drawdown']:.2%}"
            )

        # -----------------------------------------------------------
        # Best/worst relative performance
        # -----------------------------------------------------------

        if (
            "V2 QuantPilot" in regime_results
            and "SPY" in regime_results
        ):

            qp_return = regime_results[
                "V2 QuantPilot"
            ]["cumulative_return"]

            spy_return = regime_results[
                "SPY"
            ]["cumulative_return"]

            relative_return = (
                qp_return - spy_return
            )

            print(
                f"\n  V2 QuantPilot vs SPY: "
                f"{relative_return:+.2%}"
            )

    # ---------------------------------------------------------------
    # Results DataFrame
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("REGIME PERFORMANCE SUMMARY")
    print("=" * 70)

    summary_table = results_df.pivot(
        index="regime",
        columns="strategy",
        values="cumulative_return",
    )

    print(
        summary_table.to_string(
            formatters={
                strategy: "{:.2%}".format
                for strategy in summary_table.columns
            }
        )
    )

    # ---------------------------------------------------------------
    # QuantPilot relative to SPY
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("QUANTPILOT RELATIVE PERFORMANCE VS SPY")
    print("=" * 70)

    qp_results = results_df.loc[
        results_df["strategy"] == "V2 QuantPilot"
    ].set_index("regime")

    spy_results = results_df.loc[
        results_df["strategy"] == "SPY"
    ].set_index("regime")

    for regime in qp_results.index:

        qp_return = qp_results.loc[
            regime,
            "cumulative_return",
        ]

        spy_return = spy_results.loc[
            regime,
            "cumulative_return",
        ]

        difference = qp_return - spy_return

        print(
            f"{regime:25s}: "
            f"QuantPilot {qp_return:+.2%} | "
            f"SPY {spy_return:+.2%} | "
            f"Difference {difference:+.2%}"
        )

    # ---------------------------------------------------------------
    # Defensive behavior
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("DOWNSIDE / DEFENSIVE ANALYSIS")
    print("=" * 70)

    for regime in REGIMES:

        qp_row = results_df.loc[
            (
                (results_df["regime"] == regime)
                & (
                    results_df["strategy"]
                    == "V2 QuantPilot"
                )
            )
        ]

        spy_row = results_df.loc[
            (
                (results_df["regime"] == regime)
                & (
                    results_df["strategy"]
                    == "SPY"
                )
            )
        ]

        if qp_row.empty or spy_row.empty:
            continue

        qp_return = qp_row.iloc[0][
            "cumulative_return"
        ]

        spy_return = spy_row.iloc[0][
            "cumulative_return"
        ]

        qp_drawdown = qp_row.iloc[0][
            "maximum_drawdown"
        ]

        spy_drawdown = spy_row.iloc[0][
            "maximum_drawdown"
        ]

        print(
            f"\n{regime}"
        )

        print(
            f"  QuantPilot return: "
            f"{qp_return:+.2%}"
        )

        print(
            f"  SPY return:        "
            f"{spy_return:+.2%}"
        )

        print(
            f"  QuantPilot MDD:    "
            f"{qp_drawdown:.2%}"
        )

        print(
            f"  SPY MDD:           "
            f"{spy_drawdown:.2%}"
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("MARKET REGIME ROBUSTNESS COMPLETE")
    print("=" * 70)

    print("\nFixed V2 parameters:")
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

    print("\nFixed regimes:")

    for regime_name, (
        start_date,
        end_date,
    ) in REGIMES.items():

        print(
            f"  {regime_name:25s}: "
            f"{start_date} → {end_date}"
        )

    print("\nResults saved to:")
    print(OUTPUT_PATH)

    print(
        "\nIMPORTANT: Regime results are descriptive. "
        "They should not be interpreted as evidence that "
        "QuantPilot can predict future market regimes."
    )


if __name__ == "__main__":
    main()