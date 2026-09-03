from pathlib import Path

import pandas as pd
import streamlit as st

from src.goals.assets import load_selected_assets
from src.goals.feasibility import (
    calculate_maximum_feasible_return,
    calculate_recommended_target_return,
)
from src.goals.goal_math import (
    calculate_required_initial_investment,
    calculate_required_monthly_contribution,
)
from src.goals.portfolio import build_goal_portfolio
from src.goals.scenarios import build_scenario_set

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
    # 2. Risk tolerance
    # -----------------------------------------------------

    st.subheader("2. Risk Tolerance")

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
        "Conservative": (
            "Lower volatility target with a more defensive portfolio."
        ),
        "Moderate": (
            "Balanced risk and return profile."
        ),
        "Aggressive": (
            "Higher volatility tolerance with greater return potential."
        ),
    }

    st.caption(risk_descriptions[risk_tolerance])

    # -----------------------------------------------------
    # 3. Company selection
    # -----------------------------------------------------

    st.subheader("3. Select Companies")

    available_tickers = [
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

    selected_tickers = st.multiselect(
        "Companies to consider",
        options=available_tickers,
        default=available_tickers,
        help=(
            "QuantPilot will construct the portfolio using only "
            "the companies selected here."
        ),
    )

    st.caption(
        "Select at least 10 companies. The current optimizer "
        "limits each position to 10%."
    )

    # -----------------------------------------------------
    # 4. Build goal plan
    # -----------------------------------------------------

    st.divider()

    st.subheader("4. Goal Plan")

    if len(selected_tickers) < 10:
        st.warning(
            f"Select at least 10 companies. "
            f"You currently selected {len(selected_tickers)}."
        )

    elif st.button(
        "Build Goal Plan",
        type="primary",
        width="stretch",
    ):

        try:
            with st.spinner("Building QuantPilot goal plan..."):

                # ---------------------------------------------
                # Load selected historical data
                # ---------------------------------------------

                asset_data = load_selected_assets(
                    selected_tickers,
                    available_tickers,
                )

                # ---------------------------------------------
                # Build return matrix and estimate the
                # maximum feasible return under the selected
                # risk profile and portfolio constraints.
                # ---------------------------------------------

                from src.goals.assets import (
                    build_selected_return_matrix,
                    validate_minimum_history,
                )

                returns = build_selected_return_matrix(asset_data)

                validate_minimum_history(
                    returns,
                    minimum_observations=252,
                )

                estimation_returns = returns.tail(60)

                daily_expected_returns = estimation_returns.mean()

                expected_returns = (
                    (1.0 + daily_expected_returns) ** 252 - 1.0
                )

                covariance = estimation_returns.cov()

                maximum_return = calculate_maximum_feasible_return(
                    expected_returns=expected_returns,
                    covariance=covariance,
                    risk_tolerance=risk_tolerance.lower(),
                    max_weight=0.10,
                )

                # ---------------------------------------------
                # Convert the feasibility ceiling into the
                # risk-profile-specific planning return.
                # ---------------------------------------------

                recommended_return = (
                    calculate_recommended_target_return(
                        maximum_feasible_return=maximum_return,
                        risk_tolerance=risk_tolerance.lower(),
                    )
                )

                # ---------------------------------------------
                # Construct the recommended portfolio.
                # ---------------------------------------------

                portfolio = build_goal_portfolio(
                    asset_data=asset_data,
                    target_return=recommended_return,
                    risk_tolerance=risk_tolerance.lower(),
                    max_weight=0.10,
                    estimation_window=60,
                )

                # ---------------------------------------------
                # Funding requirements
                #
                # These are alternative paths:
                #   A. fund everything upfront
                #   B. fund everything monthly
                # ---------------------------------------------

                required_initial = (
                    calculate_required_initial_investment(
                        future_value=goal_amount,
                        annual_return=recommended_return,
                        years=horizon_years,
                    )
                )

                required_monthly = (
                    calculate_required_monthly_contribution(
                        future_value=goal_amount,
                        annual_return=recommended_return,
                        years=horizon_years,
                    )
                )

                # ---------------------------------------------
                # Scenario analysis
                # ---------------------------------------------

                portfolio_volatility = (
                    portfolio.expected_volatility
                )

                conservative_return = max(
                    -0.99,
                    recommended_return - portfolio_volatility,
                )

                optimistic_return = (
                    recommended_return + portfolio_volatility
                )

                scenarios = build_scenario_set(
                    target_amount=goal_amount,
                    years=horizon_years,
                    conservative_return=conservative_return,
                    expected_return=recommended_return,
                    optimistic_return=optimistic_return,
                )

            st.success("Goal plan generated successfully.")

            # =================================================
            # Return & feasibility
            # =================================================

            st.subheader("Return & Feasibility")

            return_col1, return_col2, return_col3 = st.columns(3)

            return_col1.metric(
                "Maximum Feasible Return",
                f"{maximum_return:.2%}",
            )

            return_col2.metric(
                "Recommended Return",
                f"{recommended_return:.2%}",
            )

            return_col3.metric(
                "Expected Volatility",
                f"{portfolio.expected_volatility:.2%}",
            )

            st.caption(
                "The maximum feasible return is an optimization "
                "ceiling under the selected assets and current "
                "model constraints. The recommended return is a "
                "risk-profile-based planning assumption, not a "
                "guaranteed forecast."
            )

            # =================================================
            # Required funding
            # =================================================

            st.subheader("Required Funding")

            funding_col1, funding_col2 = st.columns(2)

            funding_col1.metric(
                "Required Initial Investment",
                f"${required_initial:,.0f}",
            )

            funding_col2.metric(
                "Required Monthly Contribution",
                f"${required_monthly:,.0f}",
            )

            st.caption(
                "These are alternative funding paths. You can "
                "either invest the required amount upfront or "
                "make the required monthly contribution."
            )

            # =================================================
            # Recommended portfolio
            # =================================================

            st.subheader("Recommended Portfolio")

            weights = portfolio.weights.copy()

            portfolio_table = pd.DataFrame(
                {
                    "Ticker": weights.index,
                    "Weight": weights.values,
                    "Initial Allocation": (
                        weights.values * required_initial
                    ),
                }
            )

            portfolio_table = portfolio_table[
                portfolio_table["Weight"] > 1e-6
            ].sort_values(
                "Weight",
                ascending=False,
            )

            st.dataframe(
                portfolio_table.style.format(
                    {
                        "Weight": "{:.2%}",
                        "Initial Allocation": "${:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            # =================================================
            # Scenario analysis
            # =================================================

            st.subheader("Scenario Analysis")

            scenario_rows = [
                {
                    "Scenario": scenarios.conservative.name,
                    "Annual Return": (
                        scenarios.conservative.annual_return
                    ),
                    "Required Initial Investment": (
                        scenarios.conservative.initial_investment
                    ),
                    "Required Monthly Contribution": (
                        scenarios.conservative.monthly_contribution
                    ),
                },
                {
                    "Scenario": scenarios.expected.name,
                    "Annual Return": (
                        scenarios.expected.annual_return
                    ),
                    "Required Initial Investment": (
                        scenarios.expected.initial_investment
                    ),
                    "Required Monthly Contribution": (
                        scenarios.expected.monthly_contribution
                    ),
                },
                {
                    "Scenario": scenarios.optimistic.name,
                    "Annual Return": (
                        scenarios.optimistic.annual_return
                    ),
                    "Required Initial Investment": (
                        scenarios.optimistic.initial_investment
                    ),
                    "Required Monthly Contribution": (
                        scenarios.optimistic.monthly_contribution
                    ),
                },
            ]

            scenario_table = pd.DataFrame(
                scenario_rows
            )

            st.dataframe(
                scenario_table.style.format(
                    {
                        "Annual Return": "{:.2%}",
                        "Required Initial Investment": "${:,.0f}",
                        "Required Monthly Contribution": "${:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            st.caption(
                "Scenario returns are planning assumptions around "
                "the recommended return. They are not predictions "
                "or guarantees."
            )

            # =================================================
            # Save planner state
            # =================================================

            st.session_state["goal_plan"] = {
                "target_amount": goal_amount,
                "horizon_years": horizon_years,
                "risk_tolerance": risk_tolerance.lower(),
                "selected_tickers": selected_tickers,
                "maximum_feasible_return": maximum_return,
                "recommended_return": recommended_return,
                "required_initial_investment": required_initial,
                "required_monthly_contribution": required_monthly,
                "portfolio": portfolio,
                "scenarios": scenarios,
            }

        except ValueError as exc:
            st.error(f"Goal planning failed: {exc}")

        except Exception as exc:
            st.error(
                "QuantPilot could not build the goal plan. "
                f"Details: {exc}"
            )