from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results"
BACKTESTS_DIR = RESULTS_DIR / "backtests"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"
FIGURES_DIR = RESULTS_DIR / "figures"


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="QuantPilot",
    page_icon="📈",
    layout="wide",
)


# =========================================================
# Data loading
# =========================================================

@st.cache_data
def load_comparison() -> pd.DataFrame:
    path = COMPARISONS_DIR / "strategy_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: {path}"
        )

    return pd.read_csv(
        path,
        index_col="strategy",
    )


@st.cache_data
def load_returns() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_returns.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: {path}"
        )

    returns = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    returns = returns.set_index("date")

    return returns


@st.cache_data
def load_weights() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_weights.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: {path}"
        )

    weights = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    weights = weights.set_index("date")

    return weights


@st.cache_data
def load_turnover() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_turnover.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: {path}"
        )

    turnover = pd.read_csv(
        path,
        parse_dates=["date"],
    )

    turnover = turnover.set_index("date")

    return turnover


@st.cache_data
def load_average_weights() -> pd.DataFrame:
    path = (
        BACKTESTS_DIR
        / "quantpilot_v2_average_weights.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: {path}"
        )

    return pd.read_csv(
        path,
        index_col="ticker",
    )


@st.cache_data
def load_effective_stocks() -> pd.DataFrame:
    path = (
        BACKTESTS_DIR
        / "quantpilot_v2_effective_stocks.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing results file: {path}"
        )

    return pd.read_csv(
        path,
        parse_dates=["date"],
    ).set_index("date")


# =========================================================
# Header
# =========================================================

st.title("QuantPilot")
st.caption(
    "Adaptive Quantitative Portfolio Manager"
)

st.markdown(
    """
QuantPilot is a quantitative portfolio management system
that combines rolling expected-return estimation,
covariance modeling, constrained mean-variance optimization,
and walk-forward backtesting.
"""
)


# =========================================================
# Load data
# =========================================================

try:
    comparison = load_comparison()
    returns = load_returns()
    weights = load_weights()
    turnover = load_turnover()
    average_weights = load_average_weights()
    effective_stocks = load_effective_stocks()

except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()


# =========================================================
# Key metrics
# =========================================================

qp = comparison.loc["QuantPilot V2"]

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "CAGR",
    f"{qp['cagr']:.2%}",
)

col2.metric(
    "Sharpe",
    f"{qp['sharpe_ratio']:.2f}",
)

col3.metric(
    "Volatility",
    f"{qp['annualized_volatility']:.2%}",
)

col4.metric(
    "Max Drawdown",
    f"{qp['maximum_drawdown']:.2%}",
)

col5.metric(
    "Cumulative Return",
    f"{qp['cumulative_return']:.2%}",
)


# =========================================================
# Performance comparison
# =========================================================

st.header("Performance Comparison")

display = comparison[
    [
        "cumulative_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
    ]
].copy()

display.columns = [
    "Cumulative Return",
    "CAGR",
    "Volatility",
    "Sharpe",
    "Sortino",
    "Max Drawdown",
]

st.dataframe(
    display.style.format(
        {
            "Cumulative Return": "{:.2%}",
            "CAGR": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe": "{:.2f}",
            "Sortino": "{:.2f}",
            "Max Drawdown": "{:.2%}",
        }
    ),
    width="stretch",
)


# =========================================================
# Cumulative returns
# =========================================================

st.header("Cumulative Portfolio Growth")

cumulative = (1.0 + returns["return"]).cumprod()

st.line_chart(
    cumulative,
    height=400,
)


# =========================================================
# Drawdown
# =========================================================

st.header("QuantPilot Drawdown")

running_max = cumulative.cummax()

drawdown = cumulative / running_max - 1.0

st.area_chart(
    drawdown,
    height=300,
)


# =========================================================
# Portfolio allocation
# =========================================================

st.header("Portfolio Allocation")

latest_weights = weights.iloc[-1].sort_values(
    ascending=False
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Latest Weights")

    latest_display = (
        latest_weights[
            latest_weights > 1e-8
        ]
        .to_frame("Weight")
    )

    latest_display["Weight"] *= 100

    st.dataframe(
        latest_display.style.format(
            {"Weight": "{:.2f}%"}
        ),
        width="stretch",
    )

with col2:
    st.subheader("Average Weights")

    average_display = (
        average_weights
        .sort_values(
            "average_weight",
            ascending=False,
        )
    )

    average_display["average_weight"] *= 100

    st.dataframe(
        average_display.style.format(
            {"average_weight": "{:.2f}%"}
        ),
        width="stretch",
    )


# =========================================================
# Portfolio characteristics
# =========================================================

st.header("Portfolio Characteristics")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Active Positions",
    f"{(weights.abs() > 1e-8).sum(axis=1).mean():.2f}",
)

col2.metric(
    "Average Effective Stocks",
    f"{effective_stocks.iloc[:, 0].mean():.2f}",
)

col3.metric(
    "Maximum Position",
    f"{weights.max().max():.2%}",
)


# =========================================================
# Turnover
# =========================================================

st.header("Portfolio Turnover")

st.line_chart(
    turnover["turnover"],
    height=300,
)


# =========================================================
# Research methodology
# =========================================================

st.header("Methodology")

st.markdown(
    """
**Universe**

20 fixed liquid US equities.

**Expected Returns**

60-day historical mean returns with
50% cross-sectional shrinkage.

**Risk Model**

60-day rolling covariance matrix.

**Optimization**

Long-only, fully invested mean-variance optimization
with a 10% maximum position constraint.

**Rebalancing**

Monthly.

**Transaction Costs**

10 basis points.

**Slippage**

5 basis points.

**Backtesting**

Walk-forward, out-of-sample evaluation with
drift-aware portfolio weights.

**Benchmarks**

Equal Weight, Buy & Hold, and SPY.
"""
)


# =========================================================
# Research limitations
# =========================================================

st.header("Research Limitations")

st.markdown(
    """
- The MVP uses a fixed current-stock universe, creating
  potential survivorship bias.
- Historical estimates are based on daily market data.
- Transaction costs and slippage are modeled assumptions,
  not live execution measurements.
- Results are historical backtests and do not guarantee
  future performance.
- QuantPilot is not selected solely because it has the
  highest historical return.
"""
)


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "QuantPilot v1.0 — Research Dashboard"
)
