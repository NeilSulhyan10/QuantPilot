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
# Application mode
# =========================================================

mode = st.radio(
    "Mode",
    ["Research Mode", "Goal Planner"],
    horizontal=True,
)

if mode == "Research Mode":
    # existing research dashboard
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

else:
    # =====================================================
    # Goal Planner
    # =====================================================

    from src.goals.goal_math import calculate_required_return

    st.header("Goal Planner")

    st.markdown(
        """
        Define your financial objective and QuantPilot will
        calculate the required return, assess feasibility,
        and construct a portfolio around your goal.
        """
    )

    # -----------------------------------------------------
    # 1. Goal definition
    # -----------------------------------------------------

    st.subheader("1. Define Your Goal")

    goal_amount = st.number_input(
        "Target Future Amount",
        min_value=1_000.0,
        value=1_000_000.0,
        step=10_000.0,
        help="The amount you want to reach at the end of the horizon.",
    )

    horizon_years = st.number_input(
        "Investment Horizon (years)",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
    )

    # -----------------------------------------------------
    # 2. Investment capacity
    # -----------------------------------------------------

    st.subheader("2. Investment Capacity")

    initial_investment = st.number_input(
        "Initial Investment",
        min_value=0.0,
        value=100_000.0,
        step=5_000.0,
    )

    monthly_contribution = st.number_input(
        "Monthly Contribution",
        min_value=0.0,
        value=2_000.0,
        step=500.0,
    )

    if initial_investment == 0 and monthly_contribution == 0:
        st.warning(
            "Enter an initial investment or monthly contribution."
        )

    # -----------------------------------------------------
    # 3. Risk tolerance
    # -----------------------------------------------------

    st.subheader("3. Risk Tolerance")

    risk_tolerance = st.selectbox(
        "How much portfolio risk are you willing to accept?",
        [
            "Conservative",
            "Moderate",
            "Aggressive",
        ],
        index=1,
    )

    risk_descriptions = {
        "Conservative": "Lower volatility target.",
        "Moderate": "Balanced risk and return.",
        "Aggressive": "Higher volatility tolerance.",
    }

    st.caption(
        risk_descriptions[risk_tolerance]
    )

    # -----------------------------------------------------
    # 4. Required return
    # -----------------------------------------------------

    st.divider()

    st.subheader("4. Required Return")

    required_return = None

    if initial_investment > 0 or monthly_contribution > 0:
        try:
            required_return = calculate_required_return(
                future_value=goal_amount,
                initial_investment=initial_investment,
                monthly_contribution=monthly_contribution,
                years=horizon_years,
            )

            return_col1, return_col2 = st.columns(2)

            return_col1.metric(
                "Required Annual Return",
                f"{required_return:.2%}",
            )

            total_contributions = (
                initial_investment
                + monthly_contribution * horizon_years * 12
            )

            return_col2.metric(
                "Total Planned Contributions",
                f"${total_contributions:,.0f}",
            )

            st.caption(
                "The required return is the effective annual return "
                "needed to reach the target under your contribution plan."
            )

        except ValueError as exc:
            st.error(str(exc))

    # -----------------------------------------------------
    # 5. Goal summary
    # -----------------------------------------------------

    st.divider()

    st.subheader("Goal Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
        "Target Amount",
        f"${goal_amount:,.0f}",
    )

    summary_col2.metric(
        "Horizon",
        f"{horizon_years} years",
    )

    summary_col3.metric(
        "Risk",
        risk_tolerance,
    )

    # -----------------------------------------------------
    # Continue
    # -----------------------------------------------------

    if st.button(
        "Check Goal Feasibility",
        type="primary",
        width="stretch",
    ):
        if required_return is None:
            st.error(
                "Please provide an initial investment "
                "or monthly contribution."
            )
        else:
            st.session_state["goal_inputs"] = {
                "target_amount": goal_amount,
                "horizon_years": horizon_years,
                "initial_investment": initial_investment,
                "monthly_contribution": monthly_contribution,
                "required_return": required_return,
                "risk_tolerance": risk_tolerance.lower(),
            }

            st.success(
                "Required return calculated. "
                "Asset selection and feasibility analysis are next."
            )