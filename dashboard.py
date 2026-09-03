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


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="QuantPilot",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Premium QuantPilot visual system
# =========================================================

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --qp-bg: #080b12;
        --qp-panel: #0e131d;
        --qp-panel-2: #111824;
        --qp-border: rgba(255,255,255,.08);
        --qp-text: #f4f7fb;
        --qp-muted: #8d98aa;
        --qp-accent: #7c9cff;
        --qp-accent-2: #5eead4;
        --qp-green: #5eead4;
        --qp-red: #ff7b8a;
        --qp-glow: rgba(124,156,255,.16);
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, rgba(124,156,255,.08), transparent 30%),
            radial-gradient(circle at 10% 20%, rgba(94,234,212,.045), transparent 25%),
            var(--qp-bg);
        color: var(--qp-text);
        font-family: "DM Sans", sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        font-family: "Space Grotesk", sans-serif !important;
        letter-spacing: -.025em;
        color: var(--qp-text) !important;
    }

    h1 {
        font-size: 2.7rem !important;
        margin-bottom: .15rem !important;
    }

    h2 {
        font-size: 1.55rem !important;
        margin-top: 1.7rem !important;
    }

    h3 {
        font-size: 1.12rem !important;
    }

    p, li, label, .stCaption {
        color: var(--qp-muted);
    }

    .qp-brand {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 4px;
    }

    .qp-mark {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #7c9cff, #5eead4);
        color: #071018;
        font-family: "Space Grotesk", sans-serif;
        font-size: 22px;
        font-weight: 700;
        box-shadow: 0 0 28px rgba(124,156,255,.20);
    }

    .qp-wordmark {
        font-family: "Space Grotesk", sans-serif;
        font-size: 27px;
        font-weight: 700;
        color: var(--qp-text);
    }

    .qp-subtitle {
        color: var(--qp-muted);
        font-size: 13px;
        margin-top: -2px;
    }

    .qp-hero {
        margin: 28px 0 24px 0;
        padding: 28px 30px;
        border: 1px solid var(--qp-border);
        border-radius: 20px;
        background:
            linear-gradient(135deg, rgba(124,156,255,.10), rgba(94,234,212,.035)),
            rgba(14,19,29,.88);
        box-shadow: 0 18px 60px rgba(0,0,0,.22);
    }

    .qp-hero-title {
        font-family: "Space Grotesk", sans-serif;
        font-size: 30px;
        font-weight: 700;
        color: var(--qp-text);
        margin-bottom: 7px;
    }

    .qp-hero-copy {
        max-width: 850px;
        color: var(--qp-muted);
        line-height: 1.65;
    }

    .qp-eyebrow {
        color: var(--qp-accent-2);
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .qp-card {
        border: 1px solid var(--qp-border);
        border-radius: 16px;
        padding: 19px 20px;
        background: linear-gradient(180deg, rgba(17,24,36,.96), rgba(14,19,29,.96));
        min-height: 108px;
        box-shadow: 0 10px 35px rgba(0,0,0,.14);
    }

    .qp-card-label {
        color: var(--qp-muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 600;
        margin-bottom: 9px;
    }

    .qp-card-value {
        color: var(--qp-text);
        font-family: "Space Grotesk", sans-serif;
        font-size: 25px;
        font-weight: 700;
    }

    .qp-card-note {
        color: var(--qp-muted);
        font-size: 12px;
        margin-top: 6px;
    }

    .qp-section {
        margin-top: 30px;
        margin-bottom: 13px;
    }

    .qp-section-title {
        font-family: "Space Grotesk", sans-serif;
        color: var(--qp-text);
        font-size: 20px;
        font-weight: 700;
    }

    .qp-section-copy {
        color: var(--qp-muted);
        font-size: 13px;
        margin-top: 3px;
    }

    .qp-chip {
        display: inline-block;
        padding: 6px 10px;
        border: 1px solid var(--qp-border);
        border-radius: 999px;
        background: rgba(255,255,255,.035);
        color: var(--qp-muted);
        font-size: 11px;
        margin: 3px 5px 3px 0;
    }

    .qp-highlight {
        border-left: 3px solid var(--qp-accent);
        padding: 13px 16px;
        border-radius: 0 10px 10px 0;
        background: rgba(124,156,255,.07);
        color: var(--qp-muted);
        font-size: 13px;
        line-height: 1.55;
    }

    .qp-success {
        border-left: 3px solid var(--qp-green);
        padding: 13px 16px;
        border-radius: 0 10px 10px 0;
        background: rgba(94,234,212,.06);
        color: var(--qp-muted);
        font-size: 13px;
    }

    .qp-warning {
        border-left: 3px solid #fbbf24;
        padding: 13px 16px;
        border-radius: 0 10px 10px 0;
        background: rgba(251,191,36,.055);
        color: var(--qp-muted);
        font-size: 13px;
        line-height: 1.55;
    }

    .qp-scenario {
        border: 1px solid var(--qp-border);
        border-radius: 16px;
        padding: 18px;
        background: var(--qp-panel);
        min-height: 185px;
    }

    .qp-scenario-name {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: var(--qp-muted);
        font-weight: 700;
    }

    .qp-scenario-return {
        font-family: "Space Grotesk", sans-serif;
        font-size: 27px;
        font-weight: 700;
        color: var(--qp-text);
        margin: 8px 0 14px;
    }

    .qp-scenario-line {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 7px 0;
        border-top: 1px solid var(--qp-border);
        font-size: 12px;
    }

    .qp-scenario-line span:first-child {
        color: var(--qp-muted);
    }

    .qp-scenario-line span:last-child {
        color: var(--qp-text);
        font-weight: 600;
    }

    .qp-footer {
        margin-top: 40px;
        padding-top: 18px;
        border-top: 1px solid var(--qp-border);
        color: #697386;
        font-size: 11px;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(17,24,36,.96), rgba(14,19,29,.96));
        border: 1px solid var(--qp-border);
        border-radius: 16px;
        padding: 17px 18px;
        box-shadow: 0 10px 35px rgba(0,0,0,.14);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--qp-muted) !important;
    }

    div[data-testid="stMetricValue"] {
        color: var(--qp-text) !important;
        font-family: "Space Grotesk", sans-serif;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(124,156,255,.35);
        background: linear-gradient(135deg, rgba(124,156,255,.22), rgba(94,234,212,.12));
        color: white;
        font-weight: 700;
        min-height: 44px;
        box-shadow: 0 8px 25px rgba(0,0,0,.15);
    }

    .stButton > button:hover {
        border-color: rgba(124,156,255,.75);
        color: white;
        transform: translateY(-1px);
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div,
    .stMultiSelect div[data-baseweb="select"] {
        border-radius: 10px !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--qp-border);
        border-radius: 14px;
        overflow: hidden;
    }

    hr {
        border-color: var(--qp-border) !important;
    }

    .stAlert {
        border-radius: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Data loading
# =========================================================

@st.cache_data
def load_comparison() -> pd.DataFrame:
    path = COMPARISONS_DIR / "strategy_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, index_col="strategy")


@st.cache_data
def load_returns() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


@st.cache_data
def load_weights() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


@st.cache_data
def load_turnover() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_turnover.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


@st.cache_data
def load_average_weights() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_average_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, index_col="ticker")


@st.cache_data
def load_effective_stocks() -> pd.DataFrame:
    path = BACKTESTS_DIR / "quantpilot_v2_effective_stocks.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


# =========================================================
# Reusable UI helpers
# =========================================================

def section(title: str, copy: str | None = None):
    text = f'<div class="qp-section"><div class="qp-section-title">{title}</div>'
    if copy:
        text += f'<div class="qp-section-copy">{copy}</div>'
    text += "</div>"
    st.markdown(text, unsafe_allow_html=True)


def card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="qp-card">
            <div class="qp-card-label">{label}</div>
            <div class="qp-card-value">{value}</div>
            {"<div class='qp-card-note'>" + note + "</div>" if note else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def scenario_card(name: str, annual_return: float, initial: float, monthly: float):
    st.markdown(
        f"""
        <div class="qp-scenario">
            <div class="qp-scenario-name">{name}</div>
            <div class="qp-scenario-return">{annual_return:.2%}</div>
            <div class="qp-scenario-line">
                <span>Initial investment</span>
                <span>${initial:,.0f}</span>
            </div>
            <div class="qp-scenario-line">
                <span>Monthly contribution</span>
                <span>${monthly:,.0f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_metric(metric_name: str, value: float) -> str:
    if metric_name in {
        "cagr",
        "annualized_volatility",
        "maximum_drawdown",
        "cumulative_return",
    }:
        return f"{value:.2%}"
    return f"{value:.2f}"


# =========================================================
# Header
# =========================================================

st.markdown(
    """
    <div class="qp-brand">
        <div class="qp-mark">Q</div>
        <div>
            <div class="qp-wordmark">QuantPilot</div>
            <div class="qp-subtitle">Adaptive Quantitative Portfolio Manager</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Application mode
# =========================================================

mode = st.radio(
    "Mode",
    ["Research Mode", "Goal Planner"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Research Mode": 

    st.markdown(
        """
        <div class="qp-hero">
            <div class="qp-eyebrow">Research Terminal</div>
            <div class="qp-hero-title">Portfolio intelligence, backed by data.</div>
            <div class="qp-hero-copy">
                Walk-forward quantitative research using rolling expected returns,
                covariance estimation, constrained mean-variance optimization,
                realistic portfolio turnover, transaction costs and benchmark analysis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    qp = comparison.loc["QuantPilot V2"]

    section("QuantPilot V2", "Core out-of-sample performance")

    cols = st.columns(5)
    with cols[0]:
        card("CAGR", f"{qp['cagr']:.2%}")
    with cols[1]:
        card("Sharpe", f"{qp['sharpe_ratio']:.2f}")
    with cols[2]:
        card("Volatility", f"{qp['annualized_volatility']:.2%}")
    with cols[3]:
        card("Max Drawdown", f"{qp['maximum_drawdown']:.2%}")
    with cols[4]:
        card("Cumulative Return", f"{qp['cumulative_return']:.2%}")

    section("Performance Comparison", "Risk-adjusted comparison across the research universe")

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
        use_container_width=True,
        hide_index=False,
    )

    section("Cumulative Portfolio Growth")
    cumulative = (1.0 + returns["return"]).cumprod()
    st.line_chart(cumulative, height=400)

    section("QuantPilot Drawdown")
    drawdown = cumulative / cumulative.cummax() - 1.0
    st.area_chart(drawdown, height=300)

    section("Portfolio Allocation", "Current and average capital allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Latest Weights**")
        latest_weights = weights.iloc[-1].sort_values(ascending=False)
        latest_display = latest_weights[latest_weights > 1e-8].to_frame("Weight")
        st.dataframe(
            latest_display.style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Average Weights**")
        average_display = average_weights.sort_values(
            "average_weight", ascending=False
        ).copy()
        st.dataframe(
            average_display.style.format({"average_weight": "{:.2%}"}),
            use_container_width=True,
        )

    section("Portfolio Characteristics")

    cols = st.columns(3)
    with cols[0]:
        card(
            "Average Active Positions",
            f"{(weights.abs() > 1e-8).sum(axis=1).mean():.2f}",
        )
    with cols[1]:
        card(
            "Average Effective Stocks",
            f"{effective_stocks.iloc[:, 0].mean():.2f}",
        )
    with cols[2]:
        card("Maximum Position", f"{weights.max().max():.2%}")

    section("Portfolio Turnover")
    st.line_chart(turnover["turnover"], height=300)

    section("Methodology")

    st.markdown(
        """
        <div class="qp-highlight">
        <b>Universe</b> — 20 fixed liquid US equities.<br><br>
        <b>Expected Returns</b> — 60-day historical mean returns with
        50% cross-sectional shrinkage.<br><br>
        <b>Risk Model</b> — 60-day rolling covariance matrix.<br><br>
        <b>Optimization</b> — Long-only, fully invested mean-variance optimization
        with a 10% maximum position constraint.<br><br>
        <b>Rebalancing</b> — Monthly.<br><br>
        <b>Costs</b> — 10 bps transaction cost + 5 bps slippage.<br><br>
        <b>Backtesting</b> — Walk-forward, out-of-sample evaluation with
        drift-aware portfolio weights.<br><br>
        <b>Benchmarks</b> — Equal Weight, Buy &amp; Hold, and SPY.
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("Research Limitations")

    st.markdown(
        """
        <div class="qp-warning">
        • The MVP uses a fixed current-stock universe, creating potential survivorship bias.<br>
        • Historical estimates are based on daily market data.<br>
        • Transaction costs and slippage are modeled assumptions, not live execution measurements.<br>
        • Backtest results are historical and do not guarantee future performance.<br>
        • QuantPilot is not selected solely because it has the highest historical return.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="qp-footer">QuantPilot v1.0 · Research Mode</div>',
        unsafe_allow_html=True,
    )


else:

    # =====================================================
    # Goal Planner
    # =====================================================

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
    from src.goals.backtest import backtest_goal_portfolio
    from src.goals.assets import (
        build_selected_return_matrix,
        load_selected_assets,
        validate_minimum_history,
    )
    from src.data.benchmark_reader import load_benchmark
    from src.backtesting.benchmark import EqualWeightBacktester
    from src.backtesting.config import BacktestConfig
    from src.evaluation.metrics import evaluate_returns

    st.markdown(
        """
        <div class="qp-hero">
            <div class="qp-eyebrow">Goal Planner</div>
            <div class="qp-hero-title">Turn a future target into a portfolio plan.</div>
            <div class="qp-hero-copy">
                Set a target, choose your horizon and risk tolerance, select the
                companies you want QuantPilot to consider, and let the optimizer
                translate those constraints into a model-based funding plan.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # Inputs
    # -----------------------------------------------------

    section("01 · Define Your Goal", "The destination QuantPilot is planning toward")

    goal_col1, goal_col2 = st.columns(2)

    with goal_col1:
        goal_amount = st.number_input(
            "Target Future Amount",
            min_value=1_000.0,
            value=1_000_000.0,
            step=10_000.0,
            help="The amount you want to reach at the end of the horizon.",
        )

    with goal_col2:
        horizon_years = st.number_input(
            "Investment Horizon (years)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

    section("02 · Choose Risk", "Your risk tolerance determines the portfolio volatility constraint")

    risk_tolerance = st.selectbox(
        "Risk profile",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        label_visibility="collapsed",
    )

    risk_descriptions = {
        "Conservative": "Lower volatility target with a more defensive portfolio.",
        "Moderate": "Balanced risk and return profile.",
        "Aggressive": "Higher volatility tolerance with greater return potential.",
    }

    st.markdown(
        f'<div class="qp-chip">{risk_tolerance}</div>'
        f'<span style="color:#8d98aa;font-size:13px;">{risk_descriptions[risk_tolerance]}</span>',
        unsafe_allow_html=True,
    )

    section("03 · Select Companies", "QuantPilot will optimize only across your selected universe")

    available_tickers = [
        "AAPL", "MSFT", "NVDA", "AVGO", "GOOGL",
        "AMZN", "META", "JPM", "V", "MA",
        "JNJ", "UNH", "XOM", "CVX", "PG",
        "KO", "COST", "CAT", "WMT", "HD",
    ]

    selected_tickers = st.multiselect(
        "Companies to consider",
        options=available_tickers,
        default=available_tickers,
        help="QuantPilot constructs the portfolio using only the companies selected here.",
    )

    selected_count = len(selected_tickers)

    if selected_count < 10:
        st.markdown(
            f'<div class="qp-warning">Select at least 10 companies. '
            f'You currently have {selected_count}. The 10% position cap requires '
            f'at least 10 assets for a fully invested portfolio.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="qp-success">{selected_count} companies selected · '
            f'10% maximum position constraint satisfied.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    build = st.button(
        "Build QuantPilot Goal Plan  →",
        type="primary",
        use_container_width=True,
        disabled=selected_count < 10,
    )

    if selected_count >= 10 and build:

        try:
            with st.spinner("Estimating returns, risk and portfolio feasibility..."):

                asset_data = load_selected_assets(
                    selected_tickers,
                    available_tickers,
                )

                returns = build_selected_return_matrix(asset_data)

                # -------------------------------------------------
                # Goal Planner validation
                # -------------------------------------------------

                if goal_amount <= 0:
                    raise ValueError(
                        "Target future amount must be greater than zero."
                    )

                if horizon_years <= 0:
                    raise ValueError(
                        "Investment horizon must be greater than zero."
                    )

                if returns.empty:
                    raise ValueError(
                        "No historical return data is available for the selected companies."
                    )

                if returns.shape[1] < 10:
                    raise ValueError(
                        "At least 10 companies with valid historical data are required "
                        "because the optimizer has a 10% maximum position constraint."
                    )

                validate_minimum_history(
                    returns,
                    minimum_observations=252,
                )

                if returns.tail(60).isna().any().any():
                    raise ValueError(
                        "The latest estimation window contains missing return observations. "
                        "Try selecting companies with longer continuous price histories."
                    )

                estimation_returns = returns.tail(60)
                daily_expected_returns = estimation_returns.mean()

                expected_returns = (
                    (1.0 + daily_expected_returns) ** 252 - 1.0
                )

                if not expected_returns.replace(
                    [float("inf"), float("-inf")],
                    pd.NA,
                ).notna().all():
                    raise ValueError(
                        "The return model produced invalid expected-return estimates."
                    )

                covariance = estimation_returns.cov()

                maximum_return = calculate_maximum_feasible_return(
                    expected_returns=expected_returns,
                    covariance=covariance,
                    risk_tolerance=risk_tolerance.lower(),
                    max_weight=0.10,
                )

                if not pd.api.types.is_number(maximum_return):
                    raise ValueError(
                        "QuantPilot could not determine a feasible return for "
                        "the selected universe and risk profile."
                    )

                recommended_return = calculate_recommended_target_return(
                    maximum_feasible_return=maximum_return,
                    risk_tolerance=risk_tolerance.lower(),
                )

                if recommended_return <= -1.0:
                    raise ValueError(
                        "The resulting planning return is not mathematically usable "
                        "for the selected goal."
                    )

                portfolio = build_goal_portfolio(
                    asset_data=asset_data,
                    target_return=recommended_return,
                    risk_tolerance=risk_tolerance.lower(),
                    max_weight=0.10,
                    estimation_window=60,
                )

                required_initial = calculate_required_initial_investment(
                    future_value=goal_amount,
                    annual_return=recommended_return,
                    years=horizon_years,
                )

                required_monthly = calculate_required_monthly_contribution(
                    future_value=goal_amount,
                    annual_return=recommended_return,
                    years=horizon_years,
                )

                if required_initial < 0 or required_monthly < 0:
                    raise ValueError(
                        "QuantPilot produced an invalid funding requirement."
                    )

                if not pd.Series(
                    [required_initial, required_monthly]
                ).replace(
                    [float("inf"), float("-inf")],
                    pd.NA,
                ).notna().all():
                    raise ValueError(
                        "The funding calculation produced an invalid value. "
                        "Try a longer investment horizon."
                    )

                historical_returns = build_selected_return_matrix(asset_data)

                spy_data = load_benchmark("SPY")
                spy_returns = spy_data["Close"].pct_change().dropna()

                goal_backtest = backtest_goal_portfolio(
                    returns=historical_returns,
                    portfolio=portfolio,
                )

                equal_weight_backtest = EqualWeightBacktester(
                    config=BacktestConfig()
                ).run(returns=historical_returns)

                comparison_returns = pd.concat(
                    {
                        "QuantPilot Goal Portfolio": goal_backtest.returns,
                        "Equal Weight": equal_weight_backtest.returns,
                        "SPY": spy_returns,
                    },
                    axis=1,
                ).dropna()

                if len(comparison_returns) < 252:
                    raise ValueError(
                        "The common historical evaluation period is shorter than "
                        "one trading year. QuantPilot requires at least 252 common "
                        "observations for meaningful performance statistics."
                    )

                goal_metrics = evaluate_returns(
                    comparison_returns["QuantPilot Goal Portfolio"]
                )
                equal_weight_metrics = evaluate_returns(
                    comparison_returns["Equal Weight"]
                )
                spy_metrics = evaluate_returns(
                    comparison_returns["SPY"]
                )

                portfolio_volatility = portfolio.expected_volatility

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

            section(
                "04 · Return & Feasibility",
                "How the selected universe behaves under QuantPilot's constraints",
            )

            cols = st.columns(3)
            with cols[0]:
                card(
                    "Maximum Feasible Return",
                    f"{maximum_return:.2%}",
                    "Optimization ceiling",
                )
            with cols[1]:
                card(
                    "Recommended Return",
                    f"{recommended_return:.2%}",
                    "Risk-profile planning assumption",
                )
            with cols[2]:
                card(
                    "Expected Volatility",
                    f"{portfolio.expected_volatility:.2%}",
                    f"{risk_tolerance} profile",
                )

            st.markdown(
                """
                <div class="qp-highlight">
                <b>What this means:</b> QuantPilot first estimates the highest
                return that can satisfy the selected assets, 10% position cap and
                risk constraint. It then applies the selected risk profile to choose
                a more conservative planning return. This is a model-based assumption,
                <b>not a guaranteed future return</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # =================================================
            # Required funding
            # =================================================

            section(
                "05 · Required Funding",
                "Two alternative ways to finance the same target",
            )

            cols = st.columns(2)
            with cols[0]:
                card(
                    "Invest Upfront",
                    f"${required_initial:,.0f}",
                    f"Target: ${goal_amount:,.0f} in {horizon_years} years",
                )
            with cols[1]:
                card(
                    "Invest Monthly",
                    f"${required_monthly:,.0f}",
                    f"End-of-month contributions for {horizon_years} years",
                )

            st.markdown(
                """
                <div class="qp-highlight">
                These are <b>alternative funding paths</b>, not amounts that
                need to be combined. The first assumes the capital is invested
                upfront; the second assumes regular monthly contributions.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # =================================================
            # Recommended portfolio
            # =================================================

            section(
                "06 · Recommended Portfolio",
                "Model allocation generated from the selected companies and risk constraints",
            )

            weights = portfolio.weights.copy()
            weights = weights[weights > 1e-6].sort_values(ascending=False)

            allocation = pd.DataFrame(
                {
                    "Ticker": weights.index,
                    "Weight": weights.values,
                    "Initial Allocation": weights.values * required_initial,
                }
            )

            left, right = st.columns([1.05, 1.35])

            with left:
                st.markdown("**Allocation by company**")
                st.bar_chart(
                    weights,
                    height=390,
                )

            with right:
                st.markdown("**Capital allocation**")
                st.dataframe(
                    allocation.style.format(
                        {
                            "Weight": "{:.2%}",
                            "Initial Allocation": "${:,.0f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=390,
                )

            allocation_total = allocation["Initial Allocation"].sum()

            st.caption(
                f"Portfolio weights sum to {weights.sum():.2%}. "
                f"Displayed initial allocations total approximately ${allocation_total:,.0f}."
            )

            # =================================================
            # Scenario analysis
            # =================================================

            section(
                "07 · Scenario Analysis",
                "How the funding requirement changes if the assumed return changes",
            )

            scenario_cols = st.columns(3)

            with scenario_cols[0]:
                scenario_card(
                    scenarios.conservative.name,
                    scenarios.conservative.annual_return,
                    scenarios.conservative.initial_investment,
                    scenarios.conservative.monthly_contribution,
                )

            with scenario_cols[1]:
                scenario_card(
                    scenarios.expected.name,
                    scenarios.expected.annual_return,
                    scenarios.expected.initial_investment,
                    scenarios.expected.monthly_contribution,
                )

            with scenario_cols[2]:
                scenario_card(
                    scenarios.optimistic.name,
                    scenarios.optimistic.annual_return,
                    scenarios.optimistic.initial_investment,
                    scenarios.optimistic.monthly_contribution,
                )

            st.caption(
                "Scenario returns are planning assumptions around the recommended "
                "return. They are not statistical forecasts or guarantees."
            )

            # =================================================
            # Historical hypothetical performance
            # =================================================

            section(
                "08 · Historical Hypothetical Performance",
                "A historical stress test of the current recommended allocation",
            )

            st.markdown(
                """
                <div class="qp-warning">
                This analysis applies the <b>current recommended portfolio weights</b>
                to historical data. It is hypothetical and is not a walk-forward
                simulation of what an investor would actually have held at each point
                in history.
                </div>
                """,
                unsafe_allow_html=True,
            )

            evaluation_start = comparison_returns.index[0]
            evaluation_end = comparison_returns.index[-1]

            st.caption(
                f"Common evaluation period: {evaluation_start.date()} → {evaluation_end.date()}"
            )

            comparison_table = pd.DataFrame(
                {
                    "Metric": [
                        "CAGR",
                        "Annualized Volatility",
                        "Sharpe Ratio",
                        "Sortino Ratio",
                        "Maximum Drawdown",
                    ],
                    "QuantPilot Goal Portfolio": [
                        f"{goal_metrics['cagr']:.2%}",
                        f"{goal_metrics['annualized_volatility']:.2%}",
                        f"{goal_metrics['sharpe_ratio']:.2f}",
                        f"{goal_metrics['sortino_ratio']:.2f}",
                        f"{goal_metrics['maximum_drawdown']:.2%}",
                    ],
                    "Equal Weight": [
                        f"{equal_weight_metrics['cagr']:.2%}",
                        f"{equal_weight_metrics['annualized_volatility']:.2%}",
                        f"{equal_weight_metrics['sharpe_ratio']:.2f}",
                        f"{equal_weight_metrics['sortino_ratio']:.2f}",
                        f"{equal_weight_metrics['maximum_drawdown']:.2%}",
                    ],
                    "SPY": [
                        f"{spy_metrics['cagr']:.2%}",
                        f"{spy_metrics['annualized_volatility']:.2%}",
                        f"{spy_metrics['sharpe_ratio']:.2f}",
                        f"{spy_metrics['sortino_ratio']:.2f}",
                        f"{spy_metrics['maximum_drawdown']:.2%}",
                    ],
                }
            )

            st.dataframe(
                comparison_table,
                use_container_width=True,
                hide_index=True,
            )

            equity_comparison = (1.0 + comparison_returns).cumprod()

            st.markdown("**Growth of $1**")
            st.line_chart(equity_comparison, height=380)

            drawdown_comparison = (
                equity_comparison / equity_comparison.cummax()
            ) - 1.0

            st.markdown("**Drawdown**")
            st.area_chart(drawdown_comparison, height=280)

            # =================================================
            # Methodology note
            # =================================================

            section("Model Notes")

            st.markdown(
                """
                <div class="qp-highlight">
                <b>Estimation window:</b> latest 60 trading days.<br>
                <b>Position constraint:</b> maximum 10% per company.<br>
                <b>Portfolio:</b> long-only and fully invested.<br>
                <b>Goal return:</b> selected from a feasibility ceiling using the
                risk-profile rule.<br>
                <b>Historical comparison:</b> current goal weights versus Equal Weight
                and SPY on their common historical dates.
                </div>
                """,
                unsafe_allow_html=True,
            )

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

    st.markdown(
        '<div class="qp-footer">QuantPilot v2.0 · Goal Planner · '
        'Model-based planning only — not financial advice</div>',
        unsafe_allow_html=True,
    )
