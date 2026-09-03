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
)
from src.goals.planner import build_goal_plan_from_selection
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
            <div class="qp-eyebrow">Goal Planner</div>
            <div class="qp-hero-title">Turn a future target into a portfolio plan.</div>
            <div class="qp-hero-copy">
                Choose a market, select the companies you want QuantPilot to consider,
                define your target and horizon, and generate a constrained portfolio plan.
                Market-specific ticker rules and benchmarks are handled automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    section("02 · Choose Market", "Market selection determines ticker conventions, currency and benchmark")

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

    section("03 · Choose Risk", "Your risk tolerance determines the portfolio volatility constraint")

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

    section("04 · Select Companies", "Enter the companies you want QuantPilot to consider")

    ticker_input = st.text_input(
        "Ticker symbols",
        value="AAPL, MSFT, NVDA, AVGO, GOOGL, AMZN, META, JPM, V, MA"
        if market == Market.US
        else "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT, HINDUNILVR, MARUTI",
        help=(
            "Enter Yahoo Finance ticker symbols separated by commas. "
            "Indian symbols can be entered without the .NS suffix."
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

    if selected_tickers:
        st.caption("Normalized symbols: " + ", ".join(selected_tickers))

    st.markdown("<br>", unsafe_allow_html=True)

    build = st.button(
        "Build QuantPilot Goal Plan",
        type="primary",
        use_container_width=True,
        disabled=selected_count < 10,
    )

    if selected_count >= 10 and build:
        try:
            with st.spinner(
                "Loading market data and building the constrained goal portfolio..."
            ):
                plan = build_goal_plan_from_selection(
                    tickers=selected_tickers,
                    market=market,
                    target_amount=goal_amount,
                    years=horizon_years,
                    risk_tolerance=risk_tolerance.lower(),
                    max_weight=0.10,
                    minimum_observations=252,
                    estimation_window=60,
                    allow_download=True,
                )

                asset_data = load_selected_assets(
                    selected_tickers,
                    market=market,
                    allow_download=True,
                )

                returns = build_selected_return_matrix(asset_data)
                validate_minimum_history(
                    returns,
                    minimum_observations=252,
                )

                portfolio = plan.portfolio
                recommended_return = plan.recommended_return
                maximum_return = plan.maximum_feasible_return

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

                historical_returns = returns

                benchmark_adapter = MarketDataAdapter(market)
                benchmark_data = benchmark_adapter.load_benchmark(
                    allow_download=True,
                )
                benchmark_returns = benchmark_data["Close"].pct_change().dropna()

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
                        market_config.benchmark: benchmark_returns,
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
                benchmark_metrics = evaluate_returns(
                    comparison_returns[market_config.benchmark]
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

            section(
                "05 · Return & Feasibility",
                "How the selected universe behaves under QuantPilot's constraints",
            )

            cols = st.columns(3)
            with cols[0]:
                card("Maximum Feasible Return", f"{maximum_return:.2%}", "Optimization ceiling")
            with cols[1]:
                card("Recommended Return", f"{recommended_return:.2%}", "Risk-profile planning assumption")
            with cols[2]:
                card("Expected Volatility", f"{portfolio.expected_volatility:.2%}", f"{risk_tolerance} profile")

            st.markdown(
                """
                <div class="qp-highlight">
                QuantPilot first estimates the highest return that can satisfy the
                selected universe, position cap and risk constraint. It then applies
                the selected risk profile to choose a more conservative planning return.
                This is a model-based assumption, not a guaranteed future return.
                </div>
                """,
                unsafe_allow_html=True,
            )

            section("06 · Required Funding", "Alternative ways to finance the same target")

            cols = st.columns(2)
            with cols[0]:
                card(
                    "Invest Upfront",
                    f"{currency_symbol}{required_initial:,.0f}",
                    f"Target: {currency_symbol}{goal_amount:,.0f} in {horizon_years} years",
                )
            with cols[1]:
                card(
                    "Invest Monthly",
                    f"{currency_symbol}{required_monthly:,.0f}",
                    f"End-of-month contributions for {horizon_years} years",
                )

            st.markdown(
                """
                <div class="qp-highlight">
                These are alternative funding paths. The first assumes capital is
                invested upfront; the second assumes regular monthly contributions.
                </div>
                """,
                unsafe_allow_html=True,
            )

            section("07 · Recommended Portfolio", "Model allocation generated from the selected companies")

            weights = portfolio.weights[portfolio.weights > 1e-6].sort_values(ascending=False)

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
                st.bar_chart(weights, height=390)

            with right:
                st.markdown("**Capital allocation**")
                st.dataframe(
                    allocation.style.format(
                        {
                            "Weight": "{:.2%}",
                            "Initial Allocation": f"{currency_symbol}" + "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=390,
                )

            st.caption(
                f"Portfolio weights sum to {weights.sum():.2%}. "
                f"Displayed initial allocations total approximately "
                f"{currency_symbol}{allocation['Initial Allocation'].sum():,.0f}."
            )

            section("08 · Scenario Analysis", "How funding requirements change across planning assumptions")

            scenario_cols = st.columns(3)

            with scenario_cols[0]:
                scenario_card(
                    scenarios.conservative.name,
                    scenarios.conservative.annual_return,
                    scenarios.conservative.initial_investment,
                    scenarios.conservative.monthly_contribution,
                    currency_symbol,
                )

            with scenario_cols[1]:
                scenario_card(
                    scenarios.expected.name,
                    scenarios.expected.annual_return,
                    scenarios.expected.initial_investment,
                    scenarios.expected.monthly_contribution,
                    currency_symbol,
                )

            with scenario_cols[2]:
                scenario_card(
                    scenarios.optimistic.name,
                    scenarios.optimistic.annual_return,
                    scenarios.optimistic.initial_investment,
                    scenarios.optimistic.monthly_contribution,
                    currency_symbol,
                )

            st.caption(
                "Scenario returns are planning assumptions around the recommended return, "
                "not statistical forecasts or guarantees."
            )

            section(
                "09 · Historical Hypothetical Performance",
                "Historical stress test of the current recommended allocation",
            )

            st.markdown(
                """
                <div class="qp-warning">
                This applies the current recommended portfolio weights to historical data.
                It is hypothetical and is not a walk-forward simulation.
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
                <b>Estimation window:</b> latest 60 trading days.<br>
                <b>Position constraint:</b> maximum 10% per company.<br>
                <b>Portfolio:</b> long-only and fully invested.<br>
                <b>Historical comparison:</b> current goal weights versus Equal Weight
                and the market-specific benchmark.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.session_state["goal_plan"] = {
                "target_amount": goal_amount,
                "horizon_years": horizon_years,
                "risk_tolerance": risk_tolerance.lower(),
                "market": market.value,
                "currency": market_config.currency,
                "benchmark": market_config.benchmark,
                "selected_tickers": list(selected_tickers),
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
        '<div class="qp-footer">QuantPilot · Goal Planner · '
        'Model-based planning only — not financial advice</div>',
        unsafe_allow_html=True,
    )
