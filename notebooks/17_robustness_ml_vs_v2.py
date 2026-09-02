"""
QuantPilot — Phase 8E
Robustness: ML vs V2 Expected Returns

Research question:
    Does the strictly out-of-sample ML expected-return model
    add value over the V2 historical-mean expected-return model?

V2:
    - Historical rolling mean
    - Window: 60 trading days
    - Cross-sectional shrinkage alpha: 0.5

ML:
    - Ridge regression
    - Minimum training size: 252 observations
    - Forecast horizon: 21 trading days
    - Ridge alpha: 1.0
    - Strictly walk-forward / out-of-sample predictions

Fixed portfolio construction:
    - Covariance window: 60 days
    - Risk aversion: 10
    - Maximum position weight: 10%
    - Monthly rebalancing
    - Transaction cost: 10 bps
    - Slippage: 5 bps

Important:
    This experiment compares ML and V2 as competing expected-return
    models. It does not tune ML parameters on the final evaluation
    period.
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

from src.features.returns import (
    calculate_return_matrix,
)

from src.features.covariance import (
    rolling_covariance,
)

from src.models.expected_returns import (
    calculate_expected_returns,
)

from src.models.ml_expected_returns import (
    calculate_ml_expected_returns,
)

from src.backtesting.engine import (
    WalkForwardBacktester,
    BacktestConfig,
)

from src.backtesting.ml_engine import (
    MLWalkForwardBacktester,
)

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

# V2
V2_EXPECTED_RETURN_WINDOW = 60
V2_EXPECTED_RETURN_ALPHA = 0.5

# ML
ML_MIN_TRAIN_SIZE = 252
ML_HORIZON = 21
ML_RIDGE_ALPHA = 1.0

# Portfolio construction
COVARIANCE_WINDOW = 60
RISK_AVERSION = 10.0
MAX_WEIGHT = 0.10

# Trading costs
TRANSACTION_COST_RATE = 0.001
SLIPPAGE_RATE = 0.0005

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "robustness_ml_vs_v2.csv"
)

ANNUAL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "robustness_ml_vs_v2_annual.csv"
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
        "cumulative_return": cumulative_return(
            returns
        ),
        "cagr": cagr(
            returns
        ),
        "annualized_volatility": annualized_volatility(
            returns
        ),
        "sharpe_ratio": sharpe_ratio(
            returns
        ),
        "sortino_ratio": sortino_ratio(
            returns
        ),
        "maximum_drawdown": maximum_drawdown(
            returns
        ),
        "observations": len(returns),
    }


# ---------------------------------------------------------------------
# Align common evaluation period
# ---------------------------------------------------------------------

def align_common_period(
    return_series: dict,
) -> pd.DataFrame:
    """
    Align multiple return series on their common
    available evaluation period.
    """

    valid_series = {
        name: series.dropna()
        for name, series in return_series.items()
    }

    common_start = max(
        series.index.min()
        for series in valid_series.values()
    )

    common_end = min(
        series.index.max()
        for series in valid_series.values()
    )

    aligned = {}

    for name, series in valid_series.items():

        aligned[name] = series.loc[
            (series.index >= common_start)
            & (series.index <= common_end)
        ]

    return pd.DataFrame(aligned)


# ---------------------------------------------------------------------
# Annual performance
# ---------------------------------------------------------------------

def calculate_annual_performance(
    returns: pd.Series,
) -> pd.DataFrame:
    """
    Calculate calendar-year cumulative returns.
    """

    returns = returns.dropna()

    if returns.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "return",
            ]
        )

    annual_returns = (
        (1.0 + returns)
        .groupby(returns.index.year)
        .prod()
        - 1.0
    )

    return pd.DataFrame(
        {
            "year": annual_returns.index,
            "return": annual_returns.values,
        }
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("QUANTPILOT — PHASE 8E")
    print("ROBUSTNESS: ML VS V2 EXPECTED RETURNS")
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
    # V2 expected returns
    # ---------------------------------------------------------------

    print("\nCalculating V2 expected returns...")

    v2_expected_returns = calculate_expected_returns(
        returns,
        window=V2_EXPECTED_RETURN_WINDOW,
        alpha=V2_EXPECTED_RETURN_ALPHA,
    )

    v2_expected_returns = (
        v2_expected_returns
        .dropna(how="any")
    )

    print(
        "V2 expected returns calculated."
    )

    print(
        f"V2 available from: "
        f"{v2_expected_returns.index.min().date()}"
    )

    # ---------------------------------------------------------------
    # ML expected returns
    # ---------------------------------------------------------------

    print("\nCalculating ML expected returns...")

    ml_expected_returns = calculate_ml_expected_returns(
        returns,
        min_train_size=ML_MIN_TRAIN_SIZE,
        horizon=ML_HORIZON,
        alpha=ML_RIDGE_ALPHA,
    )

    print(
        "ML expected returns calculated."
    )

    # ---------------------------------------------------------------
    # ML availability
    # ---------------------------------------------------------------

    ml_available = (
        ml_expected_returns
        .dropna(how="any")
    )

    if ml_available.empty:
        raise ValueError(
            "No fully valid ML expected-return dates "
            "are available."
        )

    print(
        f"ML available from: "
        f"{ml_available.index.min().date()}"
    )

    # ---------------------------------------------------------------
    # Backtest configuration
    # ---------------------------------------------------------------

    config = BacktestConfig(
        estimation_window=V2_EXPECTED_RETURN_WINDOW,
        risk_aversion=RISK_AVERSION,
        max_weight=MAX_WEIGHT,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        slippage_rate=SLIPPAGE_RATE,
    )

    # ---------------------------------------------------------------
    # V2 backtest
    # ---------------------------------------------------------------

    print("\nRunning V2 QuantPilot backtest...")

    v2_backtester = WalkForwardBacktester(
        config=config
    )

    v2_result = v2_backtester.run(
        returns=returns,
        expected_returns=v2_expected_returns,
        covariance_matrices=covariance_matrices,
    )

    print(
        "V2 backtest complete."
    )

    # ---------------------------------------------------------------
    # ML backtest
    # ---------------------------------------------------------------

    print("\nRunning ML QuantPilot backtest...")

    ml_backtester = MLWalkForwardBacktester(
        config=config
    )

    ml_result = ml_backtester.run_ml(
        returns=returns,
        ml_expected_returns=ml_expected_returns,
        covariance_matrices=covariance_matrices,
    )

    print(
        "ML backtest complete."
    )

    # ---------------------------------------------------------------
    # Return series
    # ---------------------------------------------------------------

    strategy_returns = {
        "V2 QuantPilot": v2_result.returns,
        "ML QuantPilot": ml_result.returns,
    }

    # ---------------------------------------------------------------
    # Common evaluation period
    # ---------------------------------------------------------------

    aligned = align_common_period(
        strategy_returns
    )

    print("\n" + "=" * 70)
    print("COMMON ML VS V2 EVALUATION PERIOD")
    print("=" * 70)

    print(
        f"Start: "
        f"{aligned.index.min().date()}"
    )

    print(
        f"End:   "
        f"{aligned.index.max().date()}"
    )

    print(
        f"Observations: "
        f"{len(aligned)}"
    )

    # ---------------------------------------------------------------
    # Overall performance
    # ---------------------------------------------------------------

    results = []

    for strategy_name in aligned.columns:

        series = aligned[strategy_name]

        performance = calculate_performance_summary(
            series
        )

        if strategy_name == "V2 QuantPilot":
            result = v2_result

        else:
            result = ml_result

        turnover = result.turnover.dropna()

        if turnover.empty:
            average_turnover = np.nan
            total_turnover = np.nan
        else:
            average_turnover = turnover.mean()
            total_turnover = turnover.sum()

        transaction_costs = (
            result.transaction_costs.sum()
            if not result.transaction_costs.empty
            else 0.0
        )

        slippage = (
            result.slippage.sum()
            if not result.slippage.empty
            else 0.0
        )

        results.append(
            {
                "strategy": strategy_name,
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
                "total_trading_friction": (
                    transaction_costs
                    + slippage
                ),
                "rebalances": len(
                    result.turnover
                ),
                "observations": len(series),
            }
        )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------------
    # Overall comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ML VS V2 PERFORMANCE COMPARISON")
    print("=" * 70)

    display_columns = [
        "strategy",
        "cumulative_return",
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
                "cumulative_return": "{:.2%}".format,
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
    # ML minus V2
    # ---------------------------------------------------------------

    v2_row = results_df.loc[
        results_df["strategy"]
        == "V2 QuantPilot"
    ].iloc[0]

    ml_row = results_df.loc[
        results_df["strategy"]
        == "ML QuantPilot"
    ].iloc[0]

    print("\n" + "=" * 70)
    print("ML ADVANTAGE / DISADVANTAGE VS V2")
    print("=" * 70)

    print(
        f"\nCAGR difference: "
        f"{ml_row['cagr'] - v2_row['cagr']:+.2%}"
    )

    print(
        f"Volatility difference: "
        f"{ml_row['annualized_volatility'] - v2_row['annualized_volatility']:+.2%}"
    )

    print(
        f"Sharpe difference: "
        f"{ml_row['sharpe_ratio'] - v2_row['sharpe_ratio']:+.3f}"
    )

    print(
        f"Sortino difference: "
        f"{ml_row['sortino_ratio'] - v2_row['sortino_ratio']:+.3f}"
    )

    print(
        f"Maximum drawdown difference: "
        f"{ml_row['maximum_drawdown'] - v2_row['maximum_drawdown']:+.2%}"
    )

    print(
        f"Average turnover difference: "
        f"{ml_row['average_turnover'] - v2_row['average_turnover']:+.2%}"
    )

    print(
        f"Trading friction difference: "
        f"{ml_row['total_trading_friction'] - v2_row['total_trading_friction']:+.4f}"
    )

    # ---------------------------------------------------------------
    # Annual comparison
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANNUAL RETURNS — ML VS V2")
    print("=" * 70)

    annual_table = pd.DataFrame()

    for strategy_name in aligned.columns:

        annual = calculate_annual_performance(
            aligned[strategy_name]
        )

        annual = annual.set_index("year")

        annual_table[
            strategy_name
        ] = annual["return"]

    print(
        annual_table.to_string(
            formatters={
                "V2 QuantPilot": "{:.2%}".format,
                "ML QuantPilot": "{:.2%}".format,
            }
        )
    )

    # ---------------------------------------------------------------
    # Annual ML advantage
    # ---------------------------------------------------------------

    annual_difference = (
        annual_table["ML QuantPilot"]
        - annual_table["V2 QuantPilot"]
    )

    print("\n" + "=" * 70)
    print("ANNUAL ML ADVANTAGE VS V2")
    print("=" * 70)

    for year, difference in annual_difference.items():

        print(
            f"{int(year)}: "
            f"{difference:+.2%}"
        )

    ml_wins = int(
        (annual_difference > 0).sum()
    )

    v2_wins = int(
        (annual_difference < 0).sum()
    )

    ties = int(
        (annual_difference == 0).sum()
    )

    print("\nAnnual comparison:")
    print(
        f"  ML wins: {ml_wins}"
    )
    print(
        f"  V2 wins: {v2_wins}"
    )
    print(
        f"  Ties:    {ties}"
    )

    # ---------------------------------------------------------------
    # Subperiod comparison
    # ---------------------------------------------------------------

    subperiods = {
        "2016_2019": (
            "2016-05-02",
            "2019-12-31",
        ),
        "2020_2022": (
            "2020-01-01",
            "2022-12-31",
        ),
        "2023_2024": (
            "2023-01-01",
            "2024-12-31",
        ),
        "2025_2026": (
            "2025-01-01",
            "2026-08-27",
        ),
    }

    print("\n" + "=" * 70)
    print("SUBPERIOD COMPARISON")
    print("=" * 70)

    subperiod_results = []

    for period_name, (
        start_date,
        end_date,
    ) in subperiods.items():

        print(
            f"\n{period_name}"
        )

        for strategy_name in aligned.columns:

            series = aligned[
                strategy_name
            ].loc[
                start_date:end_date
            ].dropna()

            if series.empty:
                continue

            performance = (
                calculate_performance_summary(
                    series
                )
            )

            subperiod_results.append(
                {
                    "period": period_name,
                    "strategy": strategy_name,
                    "cumulative_return": performance[
                        "cumulative_return"
                    ],
                    "cagr": performance["cagr"],
                    "volatility": performance[
                        "annualized_volatility"
                    ],
                    "sharpe": performance[
                        "sharpe_ratio"
                    ],
                    "sortino": performance[
                        "sortino_ratio"
                    ],
                    "maximum_drawdown": performance[
                        "maximum_drawdown"
                    ],
                    "observations": performance[
                        "observations"
                    ],
                }
            )

            print(
                f"  {strategy_name}: "
                f"Cumulative "
                f"{performance['cumulative_return']:+.2%}, "
                f"Sharpe "
                f"{performance['sharpe_ratio']:.3f}, "
                f"MDD "
                f"{performance['maximum_drawdown']:.2%}"
            )

    subperiod_df = pd.DataFrame(
        subperiod_results
    )

    # ---------------------------------------------------------------
    # ML consistency analysis
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ML CONSISTENCY ANALYSIS")
    print("=" * 70)

    print(
        f"\nML CAGR: "
        f"{ml_row['cagr']:.2%}"
    )

    print(
        f"V2 CAGR: "
        f"{v2_row['cagr']:.2%}"
    )

    print(
        f"\nML Sharpe: "
        f"{ml_row['sharpe_ratio']:.3f}"
    )

    print(
        f"V2 Sharpe: "
        f"{v2_row['sharpe_ratio']:.3f}"
    )

    if ml_row["cagr"] > v2_row["cagr"]:
        print(
            "\nML has higher CAGR over the common "
            "evaluation period."
        )
    else:
        print(
            "\nV2 has higher CAGR over the common "
            "evaluation period."
        )

    if ml_row["sharpe_ratio"] > v2_row["sharpe_ratio"]:
        print(
            "ML has higher Sharpe over the common "
            "evaluation period."
        )
    else:
        print(
            "V2 has higher Sharpe over the common "
            "evaluation period."
        )

    # ---------------------------------------------------------------
    # Save overall results
    # ---------------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save annual results
    # ---------------------------------------------------------------

    annual_output = annual_table.copy()

    annual_output.index.name = "year"

    annual_output.to_csv(
        ANNUAL_OUTPUT_PATH
    )

    # ---------------------------------------------------------------
    # Save subperiod results separately
    # ---------------------------------------------------------------

    subperiod_output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "robustness_ml_vs_v2_subperiods.csv"
    )

    subperiod_df.to_csv(
        subperiod_output_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ML VS V2 ROBUSTNESS EXPERIMENT COMPLETE")
    print("=" * 70)

    print("\nV2 specification:")
    print(
        f"  Expected-return window: "
        f"{V2_EXPECTED_RETURN_WINDOW}"
    )

    print(
        f"  Shrinkage alpha: "
        f"{V2_EXPECTED_RETURN_ALPHA}"
    )

    print("\nML specification:")
    print(
        f"  Minimum training size: "
        f"{ML_MIN_TRAIN_SIZE}"
    )

    print(
        f"  Forecast horizon: "
        f"{ML_HORIZON}"
    )

    print(
        f"  Ridge alpha: "
        f"{ML_RIDGE_ALPHA}"
    )

    print("\nPortfolio specification:")
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

    print(
        f"  {OUTPUT_PATH}"
    )

    print(
        f"  {ANNUAL_OUTPUT_PATH}"
    )

    print(
        f"  {subperiod_output_path}"
    )

    print(
        "\nIMPORTANT: ML parameters were not selected "
        "using the final evaluation-period performance."
    )


if __name__ == "__main__":
    main()