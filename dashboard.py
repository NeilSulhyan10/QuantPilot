from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.markets import Market, get_market_config
from src.goals.assets import (
    build_selected_return_matrix,
    load_selected_assets,
    validate_minimum_history,
)
from src.goals.feasibility import (
    calculate_maximum_feasible_return,
    calculate_recommended_target_return,
)
from src.goals.goal_math import (
    calculate_required_initial_investment,
    calculate_required_monthly_contribution,
    parse_horizon,
    format_horizon,
)
from src.goals.planner import (
    build_goal_plan_from_selection,
    evaluate_company_subsets,
    evaluate_subsets_with_investment,
    SubsetResult,
    SubsetInvestmentResult,
    RiskProfileResult,
)
from src.goals.backtest import backtest_goal_portfolio
from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.config import BacktestConfig
from src.evaluation.metrics import evaluate_returns
from src.data.market_data import MarketDataAdapter
from src.goals.scenarios import build_scenario_set

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
BACKTESTS_DIR = RESULTS_DIR / "backtests"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"


st.set_page_config(
    page_title="QuantPilot",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg: #070a0f;
    --panel: #0d1219;
    --panel2: #111821;
    --border: rgba(255,255,255,.08);
    --text: #f2f5f7;
    --muted: #8e99a8;
    --green: #22c55e;
    --green2: #16a34a;
    --green-soft: rgba(34,197,94,.09);
    --red: #ef6b73;
    --amber: #eab308;
}

.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(34,197,94,.055), transparent 28%),
        radial-gradient(circle at 5% 15%, rgba(34,197,94,.035), transparent 24%),
        var(--bg);
    color: var(--text);
    font-family: "DM Sans", sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: hidden; }

.block-container {
    max-width: 1500px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1,h2,h3,h4 {
    font-family: "Space Grotesk", sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.025em;
}

h1 { font-size: 2.65rem !important; }
h2 { font-size: 1.55rem !important; margin-top: 1.7rem !important; }
h3 { font-size: 1.1rem !important; }

p,li,label,.stCaption { color: var(--muted); }

.qp-brand {
    display:flex;
    align-items:center;
    gap:14px;
}

.qp-mark {
    width:42px;
    height:42px;
    border-radius:11px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:var(--green);
    color:#041008;
    font-family:"Space Grotesk",sans-serif;
    font-size:22px;
    font-weight:700;
    box-shadow:0 0 28px rgba(34,197,94,.16);
}

.qp-wordmark {
    font-family:"Space Grotesk",sans-serif;
    font-size:27px;
    font-weight:700;
    color:var(--text);
}

.qp-subtitle {
    color:var(--muted);
    font-size:13px;
}

.qp-hero {
    margin:27px 0 24px;
    padding:28px 30px;
    border:1px solid var(--border);
    border-radius:20px;
    background:
        linear-gradient(135deg,rgba(34,197,94,.075),rgba(17,24,33,.75)),
        var(--panel);
    box-shadow:0 18px 60px rgba(0,0,0,.20);
}

.qp-eyebrow {
    color:#4ade80;
    text-transform:uppercase;
    letter-spacing:.14em;
    font-size:11px;
    font-weight:700;
    margin-bottom:7px;
}

.qp-hero-title {
    font-family:"Space Grotesk",sans-serif;
    font-size:30px;
    font-weight:700;
    color:var(--text);
    margin-bottom:7px;
}

.qp-hero-copy {
    max-width:900px;
    color:var(--muted);
    line-height:1.65;
}

.qp-section {
    margin-top:30px;
    margin-bottom:13px;
}

.qp-section-title {
    font-family:"Space Grotesk",sans-serif;
    color:var(--text);
    font-size:20px;
    font-weight:700;
}

.qp-section-copy {
    color:var(--muted);
    font-size:13px;
    margin-top:3px;
}

.qp-card {
    border:1px solid var(--border);
    border-radius:16px;
    padding:19px 20px;
    background:linear-gradient(180deg,rgba(17,24,33,.97),rgba(13,18,25,.97));
    min-height:108px;
}

.qp-card-label {
    color:var(--muted);
    font-size:12px;
    text-transform:uppercase;
    letter-spacing:.08em;
    font-weight:600;
}

.qp-card-value {
    color:var(--text);
    font-family:"Space Grotesk",sans-serif;
    font-size:25px;
    font-weight:700;
    margin-top:8px;
}

.qp-card-note {
    color:var(--muted);
    font-size:12px;
    margin-top:6px;
}

.qp-chip {
    display:inline-block;
    padding:6px 10px;
    border:1px solid rgba(34,197,94,.24);
    border-radius:999px;
    background:var(--green-soft);
    color:#86efac;
    font-size:11px;
    margin:3px 5px 3px 0;
}

.qp-highlight,.qp-success,.qp-warning {
    border-left:3px solid var(--green);
    padding:13px 16px;
    border-radius:0 10px 10px 0;
    background:var(--green-soft);
    color:var(--muted);
    font-size:13px;
    line-height:1.55;
}

.qp-warning {
    border-left-color:var(--amber);
    background:rgba(234,179,8,.055);
}

.qp-scenario {
    border:1px solid var(--border);
    border-radius:16px;
    padding:18px;
    background:var(--panel);
    min-height:185px;
}

.qp-scenario-name {
    font-size:12px;
    text-transform:uppercase;
    letter-spacing:.08em;
    color:var(--muted);
    font-weight:700;
}

.qp-scenario-return {
    font-family:"Space Grotesk",sans-serif;
    font-size:27px;
    font-weight:700;
    color:var(--text);
    margin:8px 0 14px;
}

.qp-scenario-line {
    display:flex;
    justify-content:space-between;
    padding:7px 0;
    border-top:1px solid var(--border);
    font-size:12px;
}

.qp-scenario-line span:first-child { color:var(--muted); }
.qp-scenario-line span:last-child { color:var(--text); font-weight:600; }

.qp-footer {
    margin-top:40px;
    padding-top:18px;
    border-top:1px solid var(--border);
    color:#667180;
    font-size:11px;
    text-align:center;
}

div[data-testid="stMetric"] {
    background:linear-gradient(180deg,rgba(17,24,33,.97),rgba(13,18,25,.97));
    border:1px solid var(--border);
    border-radius:16px;
    padding:17px 18px;
}

div[data-testid="stMetricLabel"] { color:var(--muted) !important; }
div[data-testid="stMetricValue"] {
    color:var(--text) !important;
    font-family:"Space Grotesk",sans-serif;
}

.stButton > button {
    border-radius:10px;
    border:1px solid rgba(34,197,94,.55);
    background:linear-gradient(180deg,var(--green),var(--green2));
    color:#031108;
    font-weight:700;
    min-height:44px;
    box-shadow:0 8px 24px rgba(34,197,94,.12);
}

.stButton > button:hover {
    border-color:#4ade80;
    background:linear-gradient(180deg,#4ade80,var(--green));
    color:#031108;
}

.stButton > button:disabled {
    background:#1b2420;
    border-color:#29352e;
    color:#66716a;
}

.stTextInput input,.stNumberInput input,.stSelectbox div,
.stMultiSelect div[data-baseweb="select"] {
    border-radius:10px !important;
}

[data-testid="stDataFrame"] {
    border:1px solid var(--border);
    border-radius:14px;
    overflow:hidden;
}

hr { border-color:var(--border) !important; }
.stAlert { border-radius:12px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_comparison():
    path = COMPARISONS_DIR / "strategy_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, index_col="strategy")


@st.cache_data
def load_returns():
    path = BACKTESTS_DIR / "quantpilot_v2_returns.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


@st.cache_data
def load_weights():
    path = BACKTESTS_DIR / "quantpilot_v2_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


@st.cache_data
def load_turnover():
    path = BACKTESTS_DIR / "quantpilot_v2_turnover.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


@st.cache_data
def load_average_weights():
    path = BACKTESTS_DIR / "quantpilot_v2_average_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, index_col="ticker")


@st.cache_data
def load_effective_stocks():
    path = BACKTESTS_DIR / "quantpilot_v2_effective_stocks.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def section(title, copy=None):
    html = f'<div class="qp-section"><div class="qp-section-title">{title}</div>'
    if copy:
        html += f'<div class="qp-section-copy">{copy}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def card(label, value, note=""):
    note_html = f'<div class="qp-card-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="qp-card">
            <div class="qp-card-label">{label}</div>
            <div class="qp-card-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def scenario_card(name, annual_return, initial, monthly, currency_symbol="$"):
    st.markdown(
        f"""
        <div class="qp-scenario">
            <div class="qp-scenario-name">{name}</div>
            <div class="qp-scenario-return">{annual_return:.2%}</div>
            <div class="qp-scenario-line">
                <span>Initial investment</span>
                <span>{currency_symbol}{initial:,.0f}</span>
            </div>
            <div class="qp-scenario-line">
                <span>Monthly contribution</span>
                <span>{currency_symbol}{monthly:,.0f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
                realistic turnover, transaction costs and benchmark analysis.
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
    for col, label, value in zip(
        cols,
        ["CAGR", "Sharpe", "Volatility", "Max Drawdown", "Cumulative Return"],
        [
            f"{qp['cagr']:.2%}",
            f"{qp['sharpe_ratio']:.2f}",
            f"{qp['annualized_volatility']:.2%}",
            f"{qp['maximum_drawdown']:.2%}",
            f"{qp['cumulative_return']:.2%}",
        ],
    ):
        with col:
            card(label, value)

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
    st.line_chart((1.0 + returns["return"]).cumprod(), height=400)

    section("QuantPilot Drawdown")
    cumulative = (1.0 + returns["return"]).cumprod()
    st.area_chart(cumulative / cumulative.cummax() - 1.0, height=300)

    section("Portfolio Allocation", "Current and average capital allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Latest Weights**")
        latest = weights.iloc[-1].sort_values(ascending=False)
        st.dataframe(
            latest[latest > 1e-8].to_frame("Weight").style.format({"Weight": "{:.2%}"}),
            use_container_width=True,
        )

    with col2:
        st.markdown("**Average Weights**")
        avg = average_weights.sort_values("average_weight", ascending=False)
        st.dataframe(
            avg.style.format({"average_weight": "{:.2%}"}),
            use_container_width=True,
        )

    section("Portfolio Characteristics")

    cols = st.columns(3)
    with cols[0]:
        card("Average Active Positions", f"{(weights.abs() > 1e-8).sum(axis=1).mean():.2f}")
    with cols[1]:
        card("Average Effective Stocks", f"{effective_stocks.iloc[:, 0].mean():.2f}")
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
        The MVP uses a fixed current-stock universe, creating potential survivorship bias.<br>
        Historical estimates are based on daily market data.<br>
        Transaction costs and slippage are modeled assumptions, not live execution measurements.<br>
        Backtest results are historical and do not guarantee future performance.<br>
        QuantPilot is not selected solely because it has the highest historical return.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="qp-footer">QuantPilot · Research Mode</div>',
        unsafe_allow_html=True,
    )


else:
    st.markdown(
        """
        <div class="qp-hero">
            <div class="qp-eyebrow">Portfolio Builder</div>
            <div class="qp-hero-title">Build your portfolio from an investment amount.</div>
            <div class="qp-hero-copy">
                Choose a market, enter your investment amount and horizon, select companies,
                and QuantPilot finds the top 3 portfolio combinations with risk/return projections.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section("01 · Choose Market", "Market selection determines ticker conventions, currency and benchmark")

    market_label = st.selectbox(
        "Market",
        ["United States", "India"],
    )

    market = (
        Market.US
        if market_label == "United States"
        else Market.INDIA
    )
    market_config = get_market_config(market)

    currency_symbol = "$" if market_config.currency == "USD" else "₹"

    st.markdown(
        f'<span class="qp-chip">{market_config.name}</span>'
        f'<span class="qp-chip">{market_config.currency}</span>'
        f'<span class="qp-chip">Benchmark: {market_config.benchmark}</span>',
        unsafe_allow_html=True,
    )

    section("02 · Investment Plan", "How much and how long")

    invest_col1, invest_col2 = st.columns(2)

    with invest_col1:
        investment_mode = st.radio(
            "Investment type",
            ["Lump Sum", "Monthly SIP"],
            horizontal=True,
            help="Lump Sum: invest all at once. Monthly SIP: invest fixed amount each month.",
        )

    with invest_col2:
        if investment_mode == "Lump Sum":
            investment_amount = st.number_input(
                f"Initial Investment ({currency_symbol})",
                min_value=100.0,
                value=1_000_000.0,
                step=100.0,
                help="Amount to invest upfront. Minimum 100.",
            )
            monthly_sip = 0.0
        else:
            investment_amount = st.number_input(
                f"Monthly SIP ({currency_symbol})",
                min_value=100.0,
                value=10_000.0,
                step=100.0,
                help="Amount to invest each month (end of month). Minimum 100.",
            )
            monthly_sip = investment_amount
            investment_amount = 0.0  # No lump sum for SIP mode

    horizon_input = st.text_input(
        "Investment Horizon",
        value="10 years",
        help=(
            "Examples: '10 years', '5 years 6 months', '18 months', '2.5 years', "
            "'1y 6m', '6m'. Use years, months, or a combination."
        ),
    )

    # Parse horizon immediately for validation feedback
    try:
        horizon_years = parse_horizon(horizon_input)
        st.caption(f"Parsed as: {format_horizon(horizon_years)} ({horizon_years:.2f} years)")
    except Exception as exc:
        st.error(str(exc))
        horizon_years = 0.0

    section("03 · Risk Profiles", "All three profiles will be shown for each portfolio combination")

    risk_descriptions = {
        "Conservative": "Lower volatility target (max 12% annual vol) — more defensive.",
        "Moderate": "Balanced risk and return (max 18% annual vol).",
        "Aggressive": "Higher volatility tolerance (max 25% annual vol) — more return potential.",
    }

    cols = st.columns(3)
    for col, (profile, desc) in zip(cols, risk_descriptions.items()):
        with col:
            st.markdown(f'<div class="qp-chip">{profile}</div>', unsafe_allow_html=True)
            st.caption(desc)

    # We'll use all three risk profiles internally
    risk_tolerance = "moderate"  # default for feasibility calculations

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
        f'<span style="color:#8e99a8;font-size:13px;">{risk_descriptions[risk_tolerance]}</span>',
        unsafe_allow_html=True,
    )

    section("04 · Estimation Window", "Historical lookback for expected return estimation")

    estimation_window = st.selectbox(
        "Estimation window",
        options=[60, 120, 252, 504, 756],
        index=2,
        format_func=lambda x: f"{x} days ({'~2 months' if x==60 else '~4 months' if x==120 else '~1 year' if x==252 else '~2 years' if x==504 else '~3 years'})",
        help="Longer windows provide more stable estimates but may not reflect recent market conditions.",
    )

    section("05 · Select Companies", "Enter the companies you want QuantPilot to consider")

    ticker_input = st.text_input(
        "Ticker symbols",
        value="AAPL, MSFT, NVDA, AVGO, GOOGL, AMZN, META, JPM, V, MA"
        if market == Market.US
        else "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT, HINDUNILVR, MARUTI",
        help=(
            "Enter Yahoo Finance ticker symbols separated by commas. "
            "Indian symbols can be entered without the .NS suffix. "
            "Minimum 3, maximum 15 companies. "
            "Position cap auto-relaxes to 1/N per subset."
        ),
    )

    raw_tickers = [
        ticker.strip()
        for ticker in ticker_input.split(",")
        if ticker.strip()
    ]

    try:
        from src.data.markets import normalize_tickers

        selected_tickers = normalize_tickers(
            raw_tickers,
            market,
        )
    except ValueError as exc:
        selected_tickers = []
        st.error(str(exc))

    selected_count = len(selected_tickers)

    if selected_count < 3:
        st.markdown(
            f'<div class="qp-warning">Select at least 3 companies. '
            f'You currently have {selected_count}.</div>',
            unsafe_allow_html=True,
        )
    elif selected_count > 15:
        st.markdown(
            f'<div class="qp-warning">Maximum 15 companies allowed. '
            f'You have {selected_count}. Please reduce selection.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="qp-success">{selected_count} companies selected · '
            f'Subsets of size 3 to {selected_count} will be evaluated.</div>',
            unsafe_allow_html=True,
        )

    if selected_tickers:
        st.caption("Normalized symbols: " + ", ".join(selected_tickers))

    st.markdown("<br>", unsafe_allow_html=True)

    section("06 · Ranking Method", "How to rank the top portfolio combinations")

    ranking_metric = st.selectbox(
        "Rank portfolios by",
        options=[
            "Recommended Return",
            "Expected Return",
            "Sharpe Ratio",
            "Max Feasible Return",
            "Lowest Volatility",
        ],
        index=1,
        help=(
            "Recommended Return: risk-adjusted target (Conservative=50%, Moderate=70%, Aggressive=85% of max feasible). "
            "Expected Return: portfolio's estimated annual return. "
            "Sharpe Ratio: risk-adjusted return (expected_return / volatility). "
            "Max Feasible Return: highest return achievable under risk constraint. "
            "Lowest Volatility: minimum risk portfolio."
        ),
    )

    st.markdown("<br>", unsafe_allow_html=True)

    build = st.button(
        "Build QuantPilot Goal Plan",
        type="primary",
        use_container_width=True,
        disabled=not (3 <= selected_count <= 15) or horizon_years <= 0,
    )

    if 3 <= selected_count <= 15 and horizon_years > 0 and build:
        try:
            with st.spinner(
                f"Evaluating all company subsets (size 3 to {selected_count})..."
            ):
                subset_results = evaluate_subsets_with_investment(
                    all_tickers=selected_tickers,
                    market=market,
                    lump_sum=investment_amount,
                    monthly_sip=monthly_sip,
                    years=horizon_years,
                    minimum_observations=252,
                    estimation_window=estimation_window,
                    shrinkage_alpha=0.5,
                    allow_download=True,
                )

            feasible_results = [r for r in subset_results if r.feasible]
            infeasible_results = [r for r in subset_results if not r.feasible]

            top3 = feasible_results[:3]

            st.success(
                f"Evaluated {len(subset_results)} subsets · "
                f"{len(feasible_results)} feasible · "
                f"Showing top 3 by projected future value"
            )

            # ---- TOP 3 PORTFOLIOS ----
            section("04 · Top 3 Portfolio Combinations", "Ranked by best projected future value")

            for rank, result in enumerate(top3, 1):
                # Show all 3 risk profiles in tabs
                with st.expander(
                    f"#{rank} · {result.subset_size} companies · "
                    f"Cap: {result.max_weight:.0%} · "
                    f"Best FV: {currency_symbol}{result.best_future_value:,.0f}",
                    expanded=(rank == 1),
                ):
                    # Portfolio weights (from moderate profile as reference)
                    if result.moderate and result.moderate.feasible:
                        ref_portfolio = result.moderate
                    elif result.conservative and result.conservative.feasible:
                        ref_portfolio = result.conservative
                    else:
                        ref_portfolio = result.aggressive

                    # We need to get the actual portfolio weights - use the first feasible one
                    # For display, we'll show the moderate profile portfolio
                    if result.moderate and result.moderate.feasible:
                        # Need to rebuild portfolio for weights
                        pass

                    # Risk Profile Comparison Table
                    st.markdown("**Risk Profile Comparison**")
                    
                    risk_data = []
                    for rp_name, rpr in [("Conservative", result.conservative), 
                                          ("Moderate", result.moderate), 
                                          ("Aggressive", result.aggressive)]:
                        if rpr and rpr.feasible:
                            fv_lump = rpr.future_value_lump_sum if investment_amount > 0 else 0
                            fv_sip = rpr.future_value_monthly if monthly_sip > 0 else 0
                            total_fv = fv_lump + fv_sip
                            risk_data.append({
                                "Profile": rp_name,
                                "Max Vol": f"{rpr.max_allowed_volatility:.1%}",
                                "Rec. Return": f"{rpr.recommended_return:.2%}",
                                "Exp. Return": f"{rpr.expected_return:.2%}",
                                "Exp. Vol": f"{rpr.expected_volatility:.2%}",
                                "Sharpe": f"{rpr.sharpe_ratio:.2f}",
                                "Future Value (Lump)": f"{currency_symbol}{fv_lump:,.0f}" if investment_amount > 0 else "—",
                                "Future Value (SIP)": f"{currency_symbol}{fv_sip:,.0f}" if monthly_sip > 0 else "—",
                                "Total Future Value": f"{currency_symbol}{total_fv:,.0f}",
                            })
                        else:
                            risk_data.append({
                                "Profile": rp_name,
                                "Max Vol": "—",
                                "Rec. Return": "—",
                                "Exp. Return": "—",
                                "Exp. Vol": "—",
                                "Sharpe": "—",
                                "Future Value (Lump)": "Infeasible",
                                "Future Value (SIP)": "Infeasible",
                                "Total Future Value": "Infeasible",
                            })

                    risk_df = pd.DataFrame(risk_data)
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)

                    # Allocation for Moderate profile (if feasible)
                    if result.moderate and result.moderate.feasible:
                        # Use stored weights from evaluation (avoids rebuild issues)
                        weights = result.moderate.weights[result.moderate.weights > 1e-6].sort_values(ascending=False)

                        col1, col2 = st.columns([1, 1.5])
                        with col1:
                            st.markdown("**Allocation (Moderate Profile)**")
                            st.bar_chart(weights, height=250)
                        with col2:
                            alloc_df = pd.DataFrame({
                                "Ticker": weights.index,
                                "Weight": weights.values,
                                "Amount": weights.values * investment_amount if investment_amount > 0 else weights.values * monthly_sip * 12,
                            })
                            st.markdown("**Capital Allocation**")
                            st.dataframe(
                                alloc_df.style.format({
                                    "Weight": "{:.2%}",
                                    "Amount": f"{currency_symbol}" + "{:,.0f}",
                                }),
                                use_container_width=True,
                                hide_index=True,
                                height=250,
                            )

                    # Warning if any
                    warnings = []
                    for rpr in [result.conservative, result.moderate, result.aggressive]:
                        if rpr and rpr.warning:
                            warnings.append(f"{rpr.risk_profile}: {rpr.warning}")
                    for w in warnings:
                        st.markdown(f'<div class="qp-warning">⚠️ {w}</div>', unsafe_allow_html=True)

            # ---- INFEASIBLE SUBSETS ----
            if infeasible_results:
                with st.expander(f"⚠️ {len(infeasible_results)} Infeasible Subsets", expanded=False):
                    for result in infeasible_results[:20]:
                        st.markdown(
                            f"**{' + '.join(result.tickers)}** "
                            f"({result.subset_size} companies, cap {result.max_weight:.0%}) — "
                            f"{result.error or 'Infeasible under risk/return constraints'}"
                        )

            # ---- DETAILED ANALYSIS FOR BEST PORTFOLIO (Moderate) ----
            if top3:
                best_result = top3[0]
                if best_result.moderate and best_result.moderate.feasible:
                    # Use stored weights from evaluation (avoids rebuild issues)
                    from src.goals.portfolio import GoalPortfolioResult
                    from src.goals.assets import load_selected_assets, build_selected_return_matrix, validate_minimum_history
                    from src.data.markets import normalize_tickers

                    normalized = normalize_tickers(list(best_result.tickers), market)
                    asset_data = load_selected_assets(
                        tickers=normalized,
                        market=market,
                        allow_download=True,
                    )
                    returns = build_selected_return_matrix(asset_data)
                    validate_minimum_history(returns, minimum_observations=252)

                    # Reconstruct portfolio from stored weights
                    best_portfolio = GoalPortfolioResult(
                        weights=best_result.moderate.weights,
                        expected_return=best_result.moderate.expected_return,
                        expected_volatility=best_result.moderate.expected_volatility,
                        target_return=best_result.moderate.recommended_return,
                        risk_tolerance="moderate",
                        maximum_allowed_volatility=best_result.moderate.max_allowed_volatility,
                        feasible=True,
                    )

                    benchmark_adapter = MarketDataAdapter(market)
                    benchmark_data = benchmark_adapter.load_benchmark(allow_download=True)
                    benchmark_returns = benchmark_data["Close"].pct_change().dropna()

                    goal_backtest = backtest_goal_portfolio(
                        returns=returns,
                        portfolio=best_portfolio,
                    )

                    equal_weight_backtest = EqualWeightBacktester(
                        config=BacktestConfig()
                    ).run(returns=returns)

                    comparison_returns = pd.concat(
                        {
                            "QuantPilot Portfolio (Moderate)": goal_backtest.returns,
                            "Equal Weight": equal_weight_backtest.returns,
                            market_config.benchmark: benchmark_returns,
                        },
                        axis=1,
                    ).dropna()

                    if len(comparison_returns) >= 252:
                        goal_metrics = evaluate_returns(
                            comparison_returns["QuantPilot Portfolio (Moderate)"]
                        )
                        equal_weight_metrics = evaluate_returns(
                            comparison_returns["Equal Weight"]
                        )
                        benchmark_metrics = evaluate_returns(
                            comparison_returns[market_config.benchmark]
                        )

                        section("05 · Best Portfolio: Detailed Analysis (Moderate Profile)", "Full analysis of the top-ranked combination")

                        cols = st.columns(3)
                        with cols[0]:
                            card("Conservative FV", f"{currency_symbol}{best_result.conservative.future_value_lump_sum + best_result.conservative.future_value_monthly:,.0f}" if best_result.conservative else "—", f"Max vol {best_result.conservative.max_allowed_volatility:.0%}")
                        with cols[1]:
                            card("Moderate FV", f"{currency_symbol}{best_result.moderate.future_value_lump_sum + best_result.moderate.future_value_monthly:,.0f}" if best_result.moderate else "—", f"Max vol {best_result.moderate.max_allowed_volatility:.0%}")
                        with cols[2]:
                            card("Aggressive FV", f"{currency_symbol}{best_result.aggressive.future_value_lump_sum + best_result.aggressive.future_value_monthly:,.0f}" if best_result.aggressive else "—", f"Max vol {best_result.aggressive.max_allowed_volatility:.0%}")

                        if best_result.moderate.warning:
                            st.markdown(
                                f"""
                                <div class="qp-warning">
                                ⚠️ {best_result.moderate.warning}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        section("06 · Historical Performance Comparison", "Backtest vs Equal Weight & Benchmark")

                        comparison_table = pd.DataFrame(
                            {
                                "Metric": [
                                    "CAGR",
                                    "Annualized Volatility",
                                    "Sharpe Ratio",
                                    "Sortino Ratio",
                                    "Maximum Drawdown",
                                ],
                                "QuantPilot (Moderate)": [
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
                                market_config.benchmark: [
                                    f"{benchmark_metrics['cagr']:.2%}",
                                    f"{benchmark_metrics['annualized_volatility']:.2%}",
                                    f"{benchmark_metrics['sharpe_ratio']:.2f}",
                                    f"{benchmark_metrics['sortino_ratio']:.2f}",
                                    f"{benchmark_metrics['maximum_drawdown']:.2%}",
                                ],
                            }
                        )

                        st.dataframe(
                            comparison_table,
                            use_container_width=True,
                            hide_index=True,
                        )

                        equity_comparison = (1.0 + comparison_returns).cumprod()

                        st.markdown("**Growth of 1 unit**")
                        st.line_chart(equity_comparison, height=380)

                        st.markdown("**Drawdown**")
                        st.area_chart(
                            equity_comparison / equity_comparison.cummax() - 1.0,
                            height=280,
                        )

                        section("Model Notes")

                        st.markdown(
                            f"""
                            <div class="qp-highlight">
                            <b>Market:</b> {market_config.name} ({market_config.currency}).<br>
                            <b>Benchmark:</b> {market_config.benchmark}.<br>
                            <b>Estimation window:</b> latest {estimation_window} trading days.<br>
                            <b>Position constraint:</b> maximum {best_result.max_weight:.0%} per company (1/N).<br>
                            <b>Portfolio:</b> long-only and fully invested.<br>
                            <b>Companies in portfolio:</b> {best_result.subset_size} of {selected_count}.<br>
                            <b>Historical comparison:</b> best subset vs Equal Weight
                            and the market-specific benchmark.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        except ValueError as exc:
            st.error(f"Portfolio building failed: {exc}")
        except Exception as exc:
            st.error(
                "QuantPilot could not build the portfolio. "
                f"Details: {exc}"
            )

    st.markdown(
        '<div class="qp-footer">QuantPilot · Goal Planner · '
        'Model-based planning only — not financial advice</div>',
        unsafe_allow_html=True,
    )
