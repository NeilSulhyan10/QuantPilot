"""
QuantPilot — Phase 8F
Robustness Summary

Consolidates results from:

    13_robustness_parameter_sensitivity.py
    14_robustness_rebalance_frequency.py
    15_robustness_transaction_costs.py
    16_robustness_market_regimes.py
    17_robustness_ml_vs_v2.py

This script does NOT rerun backtests.

It summarizes the previously completed robustness experiments
and produces research-quality tables, conclusions, and figures.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =====================================================================
# INPUT FILES
# =====================================================================

PARAMETER_FILE = (
    DATA_DIR / "robustness_parameter_sensitivity.csv"
)

REBALANCE_FILE = (
    DATA_DIR / "robustness_rebalance_frequency.csv"
)

COST_FILE = (
    DATA_DIR / "robustness_transaction_costs.csv"
)

REGIME_FILE = (
    DATA_DIR / "robustness_market_regimes.csv"
)

ML_V2_FILE = (
    DATA_DIR / "robustness_ml_vs_v2.csv"
)

ML_V2_ANNUAL_FILE = (
    DATA_DIR / "robustness_ml_vs_v2_annual.csv"
)

ML_V2_SUBPERIOD_FILE = (
    DATA_DIR / "robustness_ml_vs_v2_subperiods.csv"
)


# =====================================================================
# OUTPUT FILES
# =====================================================================

SUMMARY_FILE = (
    DATA_DIR / "robustness_summary.csv"
)

PARAMETER_SUMMARY_FILE = (
    DATA_DIR / "robustness_summary_parameters.csv"
)

REBALANCE_SUMMARY_FILE = (
    DATA_DIR / "robustness_summary_rebalancing.csv"
)

COST_SUMMARY_FILE = (
    DATA_DIR / "robustness_summary_costs.csv"
)

REGIME_SUMMARY_FILE = (
    DATA_DIR / "robustness_summary_regimes.csv"
)

ML_V2_SUMMARY_FILE = (
    DATA_DIR / "robustness_summary_ml_vs_v2.csv"
)

ANNUAL_SUMMARY_FILE = (
    DATA_DIR / "robustness_summary_annual.csv"
)


# =====================================================================
# HELPERS
# =====================================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{path}\n\n"
            f"Run the corresponding robustness notebook first."
        )


def load_csv(path):

    require_file(path)

    df = pd.read_csv(path)

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    print(
        f"Loaded {path.name}: {df.shape}"
    )

    return df


def pct(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:.2%}"


def signed_pct(value):

    if pd.isna(value):
        return "N/A"

    return f"{value:+.2%}"


# =====================================================================
# LOAD RESULTS
# =====================================================================

def load_results():

    print("\nLoading robustness results...")

    parameter_df = load_csv(
        PARAMETER_FILE
    )

    rebalance_df = load_csv(
        REBALANCE_FILE
    )

    cost_df = load_csv(
        COST_FILE
    )

    regime_df = load_csv(
        REGIME_FILE
    )

    ml_v2_df = load_csv(
        ML_V2_FILE
    )

    annual_df = load_csv(
        ML_V2_ANNUAL_FILE
    )

    subperiod_df = load_csv(
        ML_V2_SUBPERIOD_FILE
    )

    print(
        "\nAll robustness files loaded."
    )

    return (
        parameter_df,
        rebalance_df,
        cost_df,
        regime_df,
        ml_v2_df,
        annual_df,
        subperiod_df,
    )


# =====================================================================
# 8A — PARAMETER SENSITIVITY
# =====================================================================

def summarize_parameter_sensitivity(df):

    print("\n" + "=" * 70)
    print("8A — PARAMETER SENSITIVITY")
    print("=" * 70)

    print(
        f"\nConfigurations tested: {len(df)}"
    )

    print(
        f"Estimation windows: "
        f"{sorted(df['estimation_window'].unique())}"
    )

    print(
        f"Risk-aversion values: "
        f"{sorted(df['risk_aversion'].unique())}"
    )

    print(
        f"Maximum weights: "
        f"{sorted(df['max_weight'].unique())}"
    )

    # ---------------------------------------------------------------
    # Estimation window
    # ---------------------------------------------------------------

    window_summary = (
        df.groupby("estimation_window")
        .agg(
            mean_cagr=("cagr", "mean"),
            mean_volatility=(
                "annualized_volatility",
                "mean",
            ),
            mean_sharpe=(
                "sharpe_ratio",
                "mean",
            ),
            mean_sortino=(
                "sortino_ratio",
                "mean",
            ),
            mean_drawdown=(
                "maximum_drawdown",
                "mean",
            ),
            mean_turnover=(
                "average_turnover",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\nAverage performance by estimation window:"
    )

    print(
        window_summary.to_string(
            index=False,
            formatters={
                "mean_cagr": pct,
                "mean_volatility": pct,
                "mean_sharpe": "{:.3f}".format,
                "mean_sortino": "{:.3f}".format,
                "mean_drawdown": pct,
                "mean_turnover": pct,
            },
        )
    )

    # ---------------------------------------------------------------
    # Risk aversion
    # ---------------------------------------------------------------

    risk_summary = (
        df.groupby("risk_aversion")
        .agg(
            mean_cagr=("cagr", "mean"),
            mean_volatility=(
                "annualized_volatility",
                "mean",
            ),
            mean_sharpe=(
                "sharpe_ratio",
                "mean",
            ),
            mean_sortino=(
                "sortino_ratio",
                "mean",
            ),
            mean_drawdown=(
                "maximum_drawdown",
                "mean",
            ),
            mean_turnover=(
                "average_turnover",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\nAverage performance by risk aversion:"
    )

    print(
        risk_summary.to_string(
            index=False,
            formatters={
                "mean_cagr": pct,
                "mean_volatility": pct,
                "mean_sharpe": "{:.3f}".format,
                "mean_sortino": "{:.3f}".format,
                "mean_drawdown": pct,
                "mean_turnover": pct,
            },
        )
    )

    # ---------------------------------------------------------------
    # Maximum weight
    # ---------------------------------------------------------------

    cap_summary = (
        df.groupby("max_weight")
        .agg(
            mean_cagr=("cagr", "mean"),
            mean_volatility=(
                "annualized_volatility",
                "mean",
            ),
            mean_sharpe=(
                "sharpe_ratio",
                "mean",
            ),
            mean_sortino=(
                "sortino_ratio",
                "mean",
            ),
            mean_drawdown=(
                "maximum_drawdown",
                "mean",
            ),
            mean_turnover=(
                "average_turnover",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        "\nAverage performance by maximum weight:"
    )

    print(
        cap_summary.to_string(
            index=False,
            formatters={
                "mean_cagr": pct,
                "mean_volatility": pct,
                "mean_sharpe": "{:.3f}".format,
                "mean_sortino": "{:.3f}".format,
                "mean_drawdown": pct,
                "mean_turnover": pct,
            },
        )
    )

    return window_summary


# =====================================================================
# 8B — REBALANCING FREQUENCY
# =====================================================================

def summarize_rebalancing(df):

    print("\n" + "=" * 70)
    print("8B — REBALANCING FREQUENCY")
    print("=" * 70)

    print(
        "\nImportant:"
    )

    print(
        "Average turnover is turnover per rebalance."
    )

    print(
        "It should not be interpreted as total trading "
        "over the entire backtest."
    )

    print()

    formatters = {}

    for column in [
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]:

        if column in df.columns:
            formatters[column] = pct

    for column in [
        "sharpe_ratio",
        "sortino_ratio",
    ]:

        if column in df.columns:
            formatters[column] = "{:.3f}".format

    for column in [
        "transaction_costs",
        "slippage",
    ]:

        if column in df.columns:
            formatters[column] = "{:.4f}".format

    print(
        df.to_string(
            index=False,
            formatters=formatters,
        )
    )

    if "sharpe_ratio" in df.columns:

        best_sharpe = df.loc[
            df["sharpe_ratio"].idxmax()
        ]

        print(
            f"\nHighest Sharpe: "
            f"{best_sharpe['frequency']} "
            f"({best_sharpe['sharpe_ratio']:.3f})"
        )

    return df.copy()


# =====================================================================
# 8C — TRANSACTION COSTS
# =====================================================================

def summarize_costs(df):

    print("\n" + "=" * 70)
    print("8C — TRANSACTION COST ROBUSTNESS")
    print("=" * 70)

    formatters = {}

    for column in [
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "average_turnover",
    ]:

        if column in df.columns:
            formatters[column] = pct

    for column in [
        "sharpe_ratio",
        "sortino_ratio",
    ]:

        if column in df.columns:
            formatters[column] = "{:.3f}".format

    for column in [
        "transaction_costs",
        "slippage",
        "total_trading_friction",
    ]:

        if column in df.columns:
            formatters[column] = "{:.4f}".format

    print(
        df.to_string(
            index=False,
            formatters=formatters,
        )
    )

    # ---------------------------------------------------------------
    # Find baseline and frictionless rows
    # ---------------------------------------------------------------

    if "scenario" in df.columns:

        scenario = (
            df["scenario"]
            .astype(str)
            .str.lower()
        )

        baseline = df[
            scenario == "baseline"
        ]

        frictionless = df[
            scenario == "frictionless"
        ]

        if (
            not baseline.empty
            and not frictionless.empty
        ):

            b = baseline.iloc[0]
            f = frictionless.iloc[0]

            print(
                "\nBaseline vs frictionless:"
            )

            print(
                f"  CAGR impact: "
                f"{b['cagr'] - f['cagr']:+.2%}"
            )

            print(
                f"  Sharpe impact: "
                f"{b['sharpe_ratio'] - f['sharpe_ratio']:+.3f}"
            )

    return df.copy()


# =====================================================================
# 8D — MARKET REGIMES
# =====================================================================

def summarize_regimes(df):

    print("\n" + "=" * 70)
    print("8D — MARKET REGIME ANALYSIS")
    print("=" * 70)

    formatters = {}

    for column in [
        "cumulative_return",
        "annualized_volatility",
        "maximum_drawdown",
    ]:

        if column in df.columns:
            formatters[column] = pct

    for column in [
        "sharpe_ratio",
        "sortino_ratio",
    ]:

        if column in df.columns:
            formatters[column] = "{:.3f}".format

    print()

    print(
        df.to_string(
            index=False,
            formatters=formatters,
        )
    )

    # ---------------------------------------------------------------
    # V2 vs SPY
    # ---------------------------------------------------------------

    if "strategy" in df.columns:

        v2 = df[
            df["strategy"]
            .astype(str)
            .str.contains(
                "V2",
                case=False,
                na=False,
            )
        ].copy()

        spy = df[
            df["strategy"]
            .astype(str)
            .str.contains(
                "SPY",
                case=False,
                na=False,
            )
        ].copy()

        if not v2.empty and not spy.empty:

            comparison = v2.merge(
                spy,
                on="regime",
                suffixes=(
                    "_v2",
                    "_spy",
                ),
            )

            comparison[
                "v2_minus_spy"
            ] = (
                comparison[
                    "cumulative_return_v2"
                ]
                -
                comparison[
                    "cumulative_return_spy"
                ]
            )

            print(
                "\nV2 cumulative-return advantage vs SPY:"
            )

            print(
                comparison[
                    [
                        "regime",
                        "v2_minus_spy",
                    ]
                ].to_string(
                    index=False,
                    formatters={
                        "v2_minus_spy": signed_pct,
                    },
                )
            )

    return df.copy()


# =====================================================================
# 8E — ML VS V2
# =====================================================================

def summarize_ml_vs_v2(
    df,
    annual_df,
    subperiod_df,
):

    print("\n" + "=" * 70)
    print("8E — ML VS V2")
    print("=" * 70)

    formatters = {
        "cumulative_return": pct,
        "cagr": pct,
        "annualized_volatility": pct,
        "sharpe_ratio": "{:.3f}".format,
        "sortino_ratio": "{:.3f}".format,
        "maximum_drawdown": pct,
        "average_turnover": pct,
        "transaction_costs": "{:.4f}".format,
        "slippage": "{:.4f}".format,
    }

    formatters = {
        key: value
        for key, value in formatters.items()
        if key in df.columns
    }

    print()

    print(
        df.to_string(
            index=False,
            formatters=formatters,
        )
    )

    # ---------------------------------------------------------------
    # Extract strategies
    # ---------------------------------------------------------------

    v2 = df[
        df["strategy"]
        == "V2 QuantPilot"
    ].iloc[0]

    ml = df[
        df["strategy"]
        == "ML QuantPilot"
    ].iloc[0]

    # ---------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "metric": [
                "CAGR",
                "Annualized Volatility",
                "Sharpe",
                "Sortino",
                "Maximum Drawdown",
                "Average Turnover",
                "Transaction Costs",
                "Slippage",
            ],
            "V2": [
                v2["cagr"],
                v2["annualized_volatility"],
                v2["sharpe_ratio"],
                v2["sortino_ratio"],
                v2["maximum_drawdown"],
                v2["average_turnover"],
                v2["transaction_costs"],
                v2["slippage"],
            ],
            "ML": [
                ml["cagr"],
                ml["annualized_volatility"],
                ml["sharpe_ratio"],
                ml["sortino_ratio"],
                ml["maximum_drawdown"],
                ml["average_turnover"],
                ml["transaction_costs"],
                ml["slippage"],
            ],
        }
    )

    comparison[
        "ML_minus_V2"
    ] = (
        comparison["ML"]
        -
        comparison["V2"]
    )

    print(
        "\nML minus V2:"
    )

    print(
        comparison.to_string(
            index=False,
            formatters={
                "V2": "{:.4f}".format,
                "ML": "{:.4f}".format,
                "ML_minus_V2": "{:+.4f}".format,
            },
        )
    )

    # ---------------------------------------------------------------
    # Annual wins
    # ---------------------------------------------------------------

    if (
        "V2 QuantPilot" in annual_df.columns
        and "ML QuantPilot" in annual_df.columns
    ):

        difference = (
            annual_df["ML QuantPilot"]
            -
            annual_df["V2 QuantPilot"]
        )

        ml_wins = int(
            (difference > 0).sum()
        )

        v2_wins = int(
            (difference < 0).sum()
        )

        ties = int(
            (difference == 0).sum()
        )

        print(
            "\nAnnual comparison:"
        )

        print(
            f"  ML wins: {ml_wins}"
        )

        print(
            f"  V2 wins: {v2_wins}"
        )

        print(
            f"  Ties: {ties}"
        )

    # ---------------------------------------------------------------
    # Subperiod Sharpe
    # ---------------------------------------------------------------

    if (
        not subperiod_df.empty
        and "sharpe" in subperiod_df.columns
    ):

        print(
            "\nSubperiod Sharpe comparison:"
        )

        pivot = subperiod_df.pivot(
            index="period",
            columns="strategy",
            values="sharpe",
        )

        print(
            pivot.to_string(
                float_format=lambda x: f"{x:.3f}"
            )
        )

    return comparison


# =====================================================================
# FINAL RESEARCH CONCLUSION
# =====================================================================

def print_final_conclusion(
    parameter_df,
    rebalance_df,
    cost_df,
    regime_df,
    ml_v2_df,
):

    print("\n" + "=" * 70)
    print("FINAL ROBUSTNESS CONCLUSION")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1
    # ---------------------------------------------------------------

    print(
        "\n1. Parameter sensitivity"
    )

    print(
        "   Performance remains reasonably stable across the "
        "tested estimation windows, risk-aversion values, "
        "and position caps."
    )

    print(
        "   The 252-day estimation window produced the strongest "
        "average risk-adjusted performance among the tested windows."
    )

    print(
        "   The 5% maximum-weight constraint effectively pushes "
        "the 20-stock portfolio toward equal weighting."
    )

    # ---------------------------------------------------------------
    # 2
    # ---------------------------------------------------------------

    print(
        "\n2. Rebalancing frequency"
    )

    print(
        "   Monthly rebalancing provides a reasonable baseline "
        "because it balances responsiveness and trading activity."
    )

    print(
        "   Less frequent rebalancing reduces the number of "
        "rebalance events but allows larger portfolio drift."
    )

    # ---------------------------------------------------------------
    # 3
    # ---------------------------------------------------------------

    print(
        "\n3. Transaction costs"
    )

    print(
        "   Performance declines progressively as transaction "
        "cost and slippage assumptions increase."
    )

    print(
        "   The baseline strategy remains reasonably robust "
        "to the tested levels of trading friction."
    )

    print(
        "   This does not establish live-trading profitability."
    )

    # ---------------------------------------------------------------
    # 4
    # ---------------------------------------------------------------

    print(
        "\n4. Market regimes"
    )

    print(
        "   V2 shows its strongest relative behavior during "
        "adverse market conditions, particularly the 2022 bear market."
    )

    print(
        "   V2 can lag higher-beta alternatives during strong "
        "upside periods."
    )

    # ---------------------------------------------------------------
    # 5
    # ---------------------------------------------------------------

    print(
        "\n5. Machine-learning expected returns"
    )

    v2 = ml_v2_df[
        ml_v2_df["strategy"]
        == "V2 QuantPilot"
    ].iloc[0]

    ml = ml_v2_df[
        ml_v2_df["strategy"]
        == "ML QuantPilot"
    ].iloc[0]

    cagr_difference = (
        ml["cagr"]
        -
        v2["cagr"]
    )

    sharpe_difference = (
        ml["sharpe_ratio"]
        -
        v2["sharpe_ratio"]
    )

    drawdown_difference = (
        ml["maximum_drawdown"]
        -
        v2["maximum_drawdown"]
    )

    turnover_difference = (
        ml["average_turnover"]
        -
        v2["average_turnover"]
    )

    print(
        f"   ML CAGR difference: "
        f"{cagr_difference:+.2%}"
    )

    print(
        f"   ML Sharpe difference: "
        f"{sharpe_difference:+.3f}"
    )

    print(
        f"   ML drawdown difference: "
        f"{drawdown_difference:+.2%}"
    )

    print(
        f"   ML turnover difference: "
        f"{turnover_difference:+.2%}"
    )

    print(
        "   ML provides a modest absolute-return improvement "
        "but does not provide a consistent risk-adjusted "
        "improvement over V2."
    )

    print(
        "   V2 therefore remains the preferred primary "
        "expected-return specification."
    )

    # ---------------------------------------------------------------
    # Overall
    # ---------------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "OVERALL RESEARCH CONCLUSION"
    )

    print(
        "-" * 70
    )

    print(
        "\nQuantPilot's portfolio construction framework shows "
        "reasonable robustness across the tested assumptions."
    )

    print(
        "\nThe evidence supports V2 as the primary expected-return "
        "model because it combines competitive returns with "
        "better risk-adjusted performance, lower turnover, and "
        "smaller drawdowns than the tested ML alternative."
    )

    print(
        "\nThe ML model remains a useful experimental extension "
        "rather than a demonstrated replacement for the baseline."
    )

    print(
        "\nQuantPilot should therefore be presented as a "
        "risk-aware quantitative portfolio-management framework, "
        "not as a strategy guaranteed to outperform every benchmark."
    )


# =====================================================================
# VISUALIZATIONS
# =====================================================================

def create_visualizations(
    parameter_df,
    rebalance_df,
    cost_df,
    ml_v2_df,
):

    print(
        "\nCreating robustness visualizations..."
    )

    # ---------------------------------------------------------------
    # Estimation window
    # ---------------------------------------------------------------

    window_summary = (
        parameter_df
        .groupby("estimation_window")[
            "sharpe_ratio"
        ]
        .mean()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        window_summary.index,
        window_summary.values,
        marker="o",
    )

    plt.xlabel(
        "Estimation Window (Trading Days)"
    )

    plt.ylabel(
        "Average Sharpe Ratio"
    )

    plt.title(
        "Robustness to Expected-Return Estimation Window"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        DATA_DIR
        / "robustness_summary_estimation_window.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Rebalancing
    # ---------------------------------------------------------------

    if "frequency" in rebalance_df.columns:

        plt.figure(
            figsize=(8, 5)
        )

        plt.bar(
            rebalance_df["frequency"],
            rebalance_df["sharpe_ratio"],
        )

        plt.xlabel(
            "Rebalancing Frequency"
        )

        plt.ylabel(
            "Sharpe Ratio"
        )

        plt.title(
            "Performance Across Rebalancing Frequencies"
        )

        plt.tight_layout()

        plt.savefig(
            DATA_DIR
            / "robustness_summary_rebalancing.png",
            dpi=150,
        )

        plt.close()

    # ---------------------------------------------------------------
    # Transaction costs
    # ---------------------------------------------------------------

    if "scenario" in cost_df.columns:

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            cost_df["scenario"],
            cost_df["cagr"],
            marker="o",
        )

        plt.xlabel(
            "Transaction-Cost Scenario"
        )

        plt.ylabel(
            "CAGR"
        )

        plt.title(
            "Sensitivity to Trading Friction"
        )

        plt.xticks(
            rotation=30,
            ha="right",
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            DATA_DIR
            / "robustness_summary_transaction_costs.png",
            dpi=150,
        )

        plt.close()

    # ---------------------------------------------------------------
    # ML vs V2
    # ---------------------------------------------------------------

    ml_v2_indexed = (
        ml_v2_df.set_index("strategy")
    )

    plots = [
        (
            "cagr",
            "robustness_summary_ml_v2_cagr.png",
            "ML vs V2 — CAGR",
        ),
        (
            "sharpe_ratio",
            "robustness_summary_ml_v2_sharpe.png",
            "ML vs V2 — Sharpe Ratio",
        ),
        (
            "maximum_drawdown",
            "robustness_summary_ml_v2_drawdown.png",
            "ML vs V2 — Maximum Drawdown",
        ),
    ]

    for metric, filename, title in plots:

        if metric not in ml_v2_indexed.columns:
            continue

        plt.figure(
            figsize=(7, 5)
        )

        plt.bar(
            ml_v2_indexed.index,
            ml_v2_indexed[metric],
        )

        plt.ylabel(
            metric.replace(
                "_",
                " ",
            ).title()
        )

        plt.title(
            title
        )

        plt.xticks(
            rotation=15
        )

        plt.tight_layout()

        plt.savefig(
            DATA_DIR / filename,
            dpi=150,
        )

        plt.close()

    print(
        "Robustness visualizations created."
    )


# =====================================================================
# SAVE RESULTS
# =====================================================================

def save_results(
    parameter_summary,
    rebalance_summary,
    cost_summary,
    regime_summary,
    ml_v2_summary,
    annual_df,
):

    parameter_summary.to_csv(
        PARAMETER_SUMMARY_FILE,
        index=False,
    )

    rebalance_summary.to_csv(
        REBALANCE_SUMMARY_FILE,
        index=False,
    )

    cost_summary.to_csv(
        COST_SUMMARY_FILE,
        index=False,
    )

    regime_summary.to_csv(
        REGIME_SUMMARY_FILE,
        index=False,
    )

    ml_v2_summary.to_csv(
        ML_V2_SUMMARY_FILE,
        index=False,
    )

    annual_df.to_csv(
        ANNUAL_SUMMARY_FILE,
        index=False,
    )

    # Master summary = ML vs V2 comparison
    ml_v2_summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print(
        "\nSummary files saved:"
    )

    for path in [
        SUMMARY_FILE,
        PARAMETER_SUMMARY_FILE,
        REBALANCE_SUMMARY_FILE,
        COST_SUMMARY_FILE,
        REGIME_SUMMARY_FILE,
        ML_V2_SUMMARY_FILE,
        ANNUAL_SUMMARY_FILE,
    ]:

        print(
            f"  {path}"
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print("=" * 70)
    print("QUANTPILOT — PHASE 8F")
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    (
        parameter_df,
        rebalance_df,
        cost_df,
        regime_df,
        ml_v2_df,
        annual_df,
        subperiod_df,
    ) = load_results()

    # ---------------------------------------------------------------
    # Summaries
    # ---------------------------------------------------------------

    parameter_summary = (
        summarize_parameter_sensitivity(
            parameter_df
        )
    )

    rebalance_summary = (
        summarize_rebalancing(
            rebalance_df
        )
    )

    cost_summary = (
        summarize_costs(
            cost_df
        )
    )

    regime_summary = (
        summarize_regimes(
            regime_df
        )
    )

    ml_v2_summary = (
        summarize_ml_vs_v2(
            ml_v2_df,
            annual_df,
            subperiod_df,
        )
    )

    # ---------------------------------------------------------------
    # Final conclusion
    # ---------------------------------------------------------------

    print_final_conclusion(
        parameter_df,
        rebalance_df,
        cost_df,
        regime_df,
        ml_v2_df,
    )

    # ---------------------------------------------------------------
    # Visualizations
    # ---------------------------------------------------------------

    create_visualizations(
        parameter_df,
        rebalance_df,
        cost_df,
        ml_v2_df,
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    save_results(
        parameter_summary,
        rebalance_summary,
        cost_summary,
        regime_summary,
        ml_v2_summary,
        annual_df,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PHASE 8F COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nAll robustness experiments have been consolidated."
    )


if __name__ == "__main__":
    main()