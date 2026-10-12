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
    build_allocation_weights,
)
from src.goals.backtest import backtest_goal_portfolio
from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.config import BacktestConfig
from src.evaluation.metrics import evaluate_returns
from src.data.market_data import MarketDataAdapter
from src.goals.scenarios import build_scenario_set
from src.goals.returns import (
    analyze_stock,
    StockAnalysis,
    Trend,
)

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
    --bg: #f5f8fc;
    --bg-2: #eef4fb;
    --panel: #ffffff;
    --panel-solid: #ffffff;
    --panel-hover: #f7faff;
    --border: #dbe4ef;
    --border-strong: #b8d0ed;
    --text: #172033;
    --muted: #64748b;

    --blue: #2563eb;
    --blue-bright: #3b82f6;
    --blue-soft: #eff6ff;
    --blue-glow: rgba(37,99,235,.13);
    --blue-deep: #1d4ed8;

    --green: #16a34a;
    --amber: #d97706;
    --red: #dc2626;

    --shadow: 0 10px 30px rgba(30,64,175,.08);
    --shadow-sm: 0 4px 14px rgba(15,23,42,.06);
    --shadow-lg: 0 12px 28px rgba(37,99,235,.09);
}

.stApp {
    background: var(--bg) !important;
    color: var(--text);
    font-family: "DM Sans", sans-serif;
    overflow-x: hidden;
}

/* Remove Streamlit default header/nav chrome */
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    backdrop-filter: none !important;
}
[data-testid="stToolbar"] { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; }

/* Main container */
.block-container {
    max-width: 1480px;
    width: 100%;
    box-sizing: border-box;
    padding: 1.25rem 2rem 3rem;
    position: relative;
    z-index: 1;
}

/* Typography */
h1,h2,h3,h4 {
    font-family: "Space Grotesk", sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.035em;
}
h1 { font-size: 2.55rem !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.12rem !important; }
p, li, label, .stCaption { color: var(--muted); }
.stMarkdown, .stText { color: var(--text); }

/* ===== BRAND ===== */
.qp-brand {
    display:flex; align-items:center; gap:13px;
    margin: 2px 0 18px; padding-bottom:15px;
    border-bottom:1px solid #e5ebf3;
}
.qp-mark {
    width:42px; height:42px; border-radius:12px;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(135deg, #60a5fa, #2563eb);
    color:#fff; font-family:"Space Grotesk",sans-serif;
    font-size:20px; font-weight:700;
    box-shadow:0 6px 18px rgba(37,99,235,.18);
}
.qp-wordmark { font-family:"Space Grotesk",sans-serif; font-size:24px; font-weight:700; color:var(--text); letter-spacing:-.04em; }
.qp-subtitle { color:#7b8ba1; font-size:10px; margin-top:1px; letter-spacing:.02em; }

/* ===== MODE SELECTOR ===== */
div[role="radiogroup"] {
    gap:4px !important; padding:4px !important; width:fit-content;
    background:#ffffff; border:1px solid var(--border);
    border-radius:11px; box-shadow:var(--shadow-sm);
    margin-bottom:20px;
}
div[role="radiogroup"] label {
    border-radius:8px !important; padding:7px 16px !important;
    color:var(--muted) !important; background:transparent !important;
    transition:all .18s ease; font-weight:600; font-size:13px;
}
div[role="radiogroup"] label:hover { color:var(--blue-deep) !important; background:var(--blue-soft) !important; }
div[role="radiogroup"] label:has(input:checked) {
    color:var(--blue-deep) !important; background:#eaf2ff !important;
    box-shadow:inset 0 0 0 1px #c7dcfa;
}

/* ===== HERO ===== */
.qp-hero {
    position:relative; overflow:hidden;
    margin:6px 0 26px; padding:28px 31px;
    border:1px solid #d5e3f3; border-radius:18px;
    background: linear-gradient(135deg, #ffffff, #f3f7fc);
    box-shadow:var(--shadow);
}
.qp-eyebrow { color:var(--blue); text-transform:uppercase; letter-spacing:.16em; font-size:9px; font-weight:700; margin-bottom:8px; }
.qp-hero-title { font-family:"Space Grotesk",sans-serif; font-size:28px; font-weight:700; color:#172033; margin-bottom:7px; letter-spacing:-.045em; overflow-wrap:anywhere; }
.qp-hero-copy { max-width:920px; color:var(--muted); line-height:1.65; font-size:13px; }

/* ===== SECTIONS ===== */
.qp-section { margin-top:28px; margin-bottom:12px; }
.qp-section-title {
    font-family:"Space Grotesk",sans-serif; color:#172033; font-size:17px; font-weight:700;
    display:flex; align-items:center; gap:10px; letter-spacing:-.025em;
}
.qp-section-title::before { content:""; width:3px; height:17px; border-radius:99px; background:linear-gradient(180deg,#60a5fa,#2563eb); box-shadow:0 0 8px rgba(37,99,235,.18); }
.qp-section-copy { color:#718096; font-size:11px; margin-top:4px; }

/* ===== CARDS / KPIs ===== */
.qp-card { position:relative; overflow:hidden; border:1px solid var(--border); border-radius:14px; padding:16px 18px; background:#ffffff; min-height:94px; box-shadow:var(--shadow-sm); transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
.qp-card::before { content:""; position:absolute; left:0; top:0; width:100%; height:2px; background:linear-gradient(90deg, transparent, rgba(59,130,246,.48), transparent); }
.qp-card:hover { transform:translateY(-2px); border-color:#bfd5ef; box-shadow:var(--shadow-lg); }
.qp-card-label { color:#718096; font-size:9px; text-transform:uppercase; letter-spacing:.09em; font-weight:700; }
.qp-card-value { color:#172033; font-family:"Space Grotesk",sans-serif; font-size:24px; font-weight:700; margin-top:6px; letter-spacing:-.035em; }
.qp-card-note { color:#718096; font-size:10px; margin-top:4px; }

/* ===== CHIPS ===== */
.qp-chip { display:inline-flex; align-items:center; padding:5px 10px; border:1px solid #c8dcf5; border-radius:999px; background:#eff6ff; color:#2563eb; font-size:10px; font-weight:600; margin:2px 4px 2px 0; }
.qp-chip-blue { border-color:#c8dcf5; background:#eff6ff; color:#2563eb; }
.qp-chip-purple { border-color:#ddd0fb; background:#f5f3ff; color:#6d28d9; }
.qp-chip-amber { border-color:#f4d7a1; background:#fffbeb; color:#b45309; }

/* ===== CALLOUTS ===== */
.qp-highlight,.qp-success,.qp-warning,.qp-info { border:1px solid #cfe0f4; border-left:3px solid #2563eb; padding:12px 16px; border-radius:10px; background:#f4f8fd; color:#5f7087; font-size:12px; line-height:1.6; }
.qp-success { border-color:#c9e8d3; border-left-color:#16a34a; background:#f1faf4; }
.qp-warning { border-color:#f1d8a8; border-left-color:#d97706; background:#fffaf0; }
.qp-info { border-color:#ddd0fb; border-left-color:#7c3aed; background:#f8f6ff; }

/* ===== SCENARIO CARDS ===== */
.qp-scenario { border:1px solid var(--border); border-radius:14px; padding:16px; background:#ffffff; min-height:155px; box-shadow:var(--shadow-sm); transition:transform .18s ease, border-color .18s ease; }
.qp-scenario:hover { transform:translateY(-2px); border-color:#bfd5ef; }
.qp-scenario-name { font-size:9px; text-transform:uppercase; letter-spacing:.09em; color:#718096; font-weight:700; }
.qp-scenario-return { font-family:"Space Grotesk",sans-serif; font-size:25px; font-weight:700; color:#172033; margin:7px 0 13px; }
.qp-scenario-line { display:flex; justify-content:space-between; padding:7px 0; border-top:1px solid #edf1f6; font-size:10px; }
.qp-scenario-line span:first-child { color:#718096; }
.qp-scenario-line span:last-child { color:#334155; font-weight:600; }

/* ===== STREAMLIT METRICS ===== */
div[data-testid="stMetric"] { background:#ffffff; border:1px solid var(--border); border-radius:14px; padding:14px 17px; box-shadow:var(--shadow-sm); }
div[data-testid="stMetricLabel"] { color:#718096 !important; }
div[data-testid="stMetricValue"] { color:#172033 !important; font-family:"Space Grotesk",sans-serif; }

/* ===== BUTTONS ===== */
.stButton > button { border-radius:10px; border:1px solid var(--blue); background:linear-gradient(135deg,var(--blue-bright),var(--blue)); color:#fff; font-weight:700; min-height:43px; box-shadow:0 7px 18px rgba(37,99,235,.15); transition:transform .16s ease, box-shadow .16s ease, filter .16s ease; }
.stButton > button:hover { filter:brightness(1.04); transform:translateY(-1px); box-shadow:0 10px 24px rgba(37,99,235,.21); }
.stButton > button:disabled { background:#e8edf4; border-color:#d5dde8; color:#94a3b8; box-shadow:none; }
.stButton > button[kind="secondary"] { background:#ffffff; border-color:var(--border); color:#334155; box-shadow:none; }
.stButton > button[kind="secondary"]:hover { border-color:#b8d0ed; background:#f5f8fc; }

/* ===== INPUTS - ALL WHITE BACKGROUNDS, BLUE FOCUS, BLUE SELECTED ===== */
.stTextInput input, .stNumberInput input,
.stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"],
.stTextArea textarea {
    border-radius:10px !important;
    background:#ffffff !important;
    border:1px solid #d5deea !important;
    color:#172033 !important;
    box-shadow:0 2px 6px rgba(15,23,42,.025);
}
.stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
    border-color:#5b9af7 !important;
    box-shadow:0 0 0 3px rgba(37,99,235,.09) !important;
    outline: none !important;
}

/* Selectbox / MultiSelect dropdown */
[data-baseweb="popover"] [data-baseweb="menu"] {
    background:#ffffff !important; border:1px solid var(--border) !important;
    border-radius:11px !important; box-shadow:0 12px 30px rgba(15,23,42,.12);
}
[data-baseweb="option"] { color:#334155 !important; }
[data-baseweb="option"]:hover, [data-baseweb="option"][aria-selected="true"] { background:#eff6ff !important; color:#2563eb !important; }
[data-baseweb="option"][aria-selected="true"] { font-weight:600; }

/* Select trigger */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background:#ffffff !important; color:#334155 !important;
}
.stSelectbox [data-baseweb="select"] svg,
.stMultiSelect [data-baseweb="select"] svg { fill:#64748b !important; }
.stSelectbox [data-baseweb="select"] [data-baseweb="tag"],
.stMultiSelect [data-baseweb="select"] [data-baseweb="tag"] {
    background:#eff6ff !important; color:#2563eb !important; border:1px solid #c8dcf5 !important;
}

/* Placeholders */
.stTextInput input::placeholder, .stNumberInput input::placeholder, .stTextArea textarea::placeholder { color:#94a3b8 !important; }

/* Number input buttons */
.stNumberInput button { background:#f8fafc !important; color:#475569 !important; border-color:#dbe4ef !important; }
.stNumberInput button:hover { background:#eff6ff !important; color:#2563eb !important; }

/* Input wrappers */
.stTextInput > div > div, .stNumberInput > div > div, .stTextArea > div > div { background:#ffffff !important; }

/* Labels */
.stSelectbox label, .stMultiSelect label, .stTextInput label,
.stNumberInput label, .stTextArea label, .stRadio > label,
.stCheckbox > label, .stSlider > label {
    color:var(--muted) !important; font-weight:500;
}

/* ===== RADIO BUTTONS - BLUE SELECTED, WHITE BG ===== */
.stRadio > div { gap:6px; }
.stRadio [data-baseweb="radio"] {
    background:#ffffff !important; border:1px solid var(--border) !important;
    border-radius:9px !important; padding:8px 13px !important;
}
.stRadio [data-baseweb="radio"]:hover { border-color:#b8d0ed !important; }
.stRadio [data-baseweb="radio"][aria-checked="true"] {
    background:#eff6ff !important; border-color:#9fc1ed !important;
}
.stRadio label { color:#475569 !important; font-weight:600 !important; font-size:12px !important; }

/* Force inner radio dot to be blue when selected */
.stRadio [data-baseweb="radio"] [role="radio"] { background:#ffffff !important; border-color:#94a3b8 !important; }
.stRadio [data-baseweb="radio"][aria-checked="true"] [role="radio"] {
    background:#2563eb !important; border-color:#2563eb !important;
    box-shadow:inset 0 0 0 3px #ffffff !important;
}
.stRadio [data-baseweb="radio"][aria-checked="true"] [role="radio"]::after { background:#ffffff !important; }

/* ===== CHECKBOXES ===== */
.stCheckbox [data-baseweb="checkbox"] { background:#ffffff !important; border-color:#cbd5e1 !important; }
.stCheckbox [data-baseweb="checkbox"][data-checked="true"] { background:#2563eb !important; border-color:#2563eb !important; }

/* ===== SLIDER ===== */
.stProgress > div > div { background:linear-gradient(90deg,var(--blue),#60a5fa) !important; border-radius:99px; }
.stSlider [data-baseweb="slider"] { color:var(--blue) !important; }

/* ===== DATAFRAMES / TABLES - LIGHT THEME ===== */
[data-testid="stDataFrame"] {
    border:1px solid var(--border); border-radius:12px; overflow:hidden;
    box-shadow:var(--shadow-sm); background:#ffffff;
}
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] [role="grid"],
[data-testid="stDataFrame"] canvas,
[data-testid="stDataFrame"] [data-testid="stDataFrameResizable"],
[data-testid="stDataFrame"] [data-testid="stDataFrameGlideDataEditor"] {
    background:#ffffff !important;
}
[data-testid="stDataFrame"] table { font-family:"DM Sans",sans-serif !important; }
[data-testid="stDataFrame"] th {
    background:#f1f5f9 !important; color:#64748b !important;
    font-weight:700 !important; font-size:9px !important;
    text-transform:uppercase; letter-spacing:.06em;
    border-bottom:1px solid #e2e8f0 !important;
}
[data-testid="stDataFrame"] td {
    color:#334155 !important; background:#ffffff !important;
    font-size:11px !important; border-bottom:1px solid #f1f5f9 !important;
}
[data-testid="stDataFrame"] [role="gridcell"]:hover { background:#f8fbff !important; }
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"] { background:#f8fbff !important; }

/* ===== TABS ===== */
[data-baseweb="tab-list"] { gap:3px; background:#f1f5f9; border:1px solid var(--border); border-radius:10px; padding:3px; }
[data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important; border-radius:7px !important; font-weight:600 !important; font-size:11px !important; padding:8px 14px !important; }
[data-baseweb="tab"][aria-selected="true"] { background:#ffffff !important; color:var(--blue) !important; box-shadow:var(--shadow-sm); }

/* ===== EXPANDERS ===== */
[data-testid="stExpander"] { background:#ffffff; border:1px solid var(--border); border-radius:12px; overflow:hidden; box-shadow:var(--shadow-sm); }
[data-testid="stExpander"] summary { color:#334155 !important; font-weight:600; }
[data-testid="stExpanderDetails"] { background:#ffffff !important; padding:12px 16px; }
.streamlit-expanderHeader { background:transparent !important; border:none !important; font-weight:600 !important; }

/* ===== CHARTS - WHITE BG, SLATE AXES, BLUE ACCENT, NO WHEEL ZOOM ===== */
.qp-chart-container,
[data-testid="stLineChart"],
[data-testid="stAreaChart"],
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stDeckGlJsonChart"] {
    background:#ffffff !important;
    border:1px solid var(--border) !important;
    border-radius:12px !important;
    padding:16px !important;
    box-shadow:var(--shadow-sm);
}

/* Vega/Altair canvas and SVG surfaces */
[data-testid="stVegaLiteChart"] > div,
[data-testid="stVegaLiteChart"] canvas,
[data-testid="stVegaLiteChart"] svg,
[data-testid="stArrowVegaLiteChart"] > div,
[data-testid="stArrowVegaLiteChart"] canvas,
[data-testid="stArrowVegaLiteChart"] svg {
    background:#ffffff !important;
}

/* Chart axis/legend/grid text - slate gray */
[data-testid="stVegaLiteChart"] text,
[data-testid="stArrowVegaLiteChart"] text,
[data-testid="stLineChart"] text,
[data-testid="stAreaChart"] text {
    fill:#64748b !important;
}

/* Remove dark chart wrappers */
.stElementContainer:has([data-testid="stVegaLiteChart"]),
.stElementContainer:has([data-testid="stLineChart"]),
.stElementContainer:has([data-testid="stAreaChart"]) {
    background:transparent !important;
}

/* Disable wheel zoom on charts - keep buttons functional */
[data-testid="stVegaLiteChart"],
[data-testid="stArrowVegaLiteChart"],
[data-testid="stLineChart"],
[data-testid="stAreaChart"] {
    overflow: hidden !important;
}
[data-testid="stVegaLiteChart"] > div,
[data-testid="stArrowVegaLiteChart"] > div,
[data-testid="stLineChart"] > div,
[data-testid="stAreaChart"] > div {
    pointer-events: none !important;
}
[data-testid="stVegaLiteChart"] button,
[data-testid="stArrowVegaLiteChart"] button,
[data-testid="stLineChart"] button,
[data-testid="stAreaChart"] button {
    pointer-events: auto !important;
}
[data-testid="stVegaLiteChart"] canvas,
[data-testid="stArrowVegaLiteChart"] canvas,
[data-testid="stLineChart"] canvas,
[data-testid="stAreaChart"] canvas {
    pointer-events: none !important;
}

/* ===== DIVIDERS ===== */
hr { border-color:#e2e8f0 !important; }
.qp-divider { display:flex; align-items:center; gap:12px; margin:20px 0; color:#94a3b8; font-size:9px; text-transform:uppercase; letter-spacing:.1em; }
.qp-divider::before,.qp-divider::after { content:""; flex:1; height:1px; background:#e2e8f0; }

/* ===== STATUS / BADGES ===== */
.qp-status,.qp-badge { display:inline-flex; align-items:center; gap:5px; border-radius:999px; font-size:9px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; }
.qp-status { padding:4px 9px; } .qp-badge { padding:3px 8px; }
.qp-status-green,.qp-badge-green { background:#f0fdf4; color:#15803d; border:1px solid #bbf7d0; }
.qp-status-amber,.qp-badge-amber { background:#fffbeb; color:#b45309; border:1px solid #fde68a; }
.qp-status-red,.qp-badge-red { background:#fef2f2; color:#b91c1c; border:1px solid #fecaca; }
.qp-status-blue,.qp-badge-blue { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; }

/* ===== FOOTER ===== */
.qp-footer { margin-top:38px; padding-top:17px; border-top:1px solid #e2e8f0; color:#94a3b8; font-size:9px; text-align:center; letter-spacing:.03em; }

/* ===== FOCUS + SCROLLBAR ===== */
*:focus-visible { outline:2px solid rgba(37,99,235,.55) !important; outline-offset:2px !important; }
::-webkit-scrollbar { width:8px; height:8px; }
::-webkit-scrollbar-track { background:#eef2f7; }
::-webkit-scrollbar-thumb { background:#c5d1df; border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background:#aebdce; }

/* ===== RESPONSIVE ===== */
@media (max-width: 900px) {
    .block-container { padding-left:1rem; padding-right:1rem; }
    .qp-hero { padding:23px; }
    .qp-hero-title { font-size:23px; }
}
</style>
<script>
(function() {
    function disableChartWheelZoom() {
        const selectors = [
            '[data-testid="stVegaLiteChart"]',
            '[data-testid="stArrowVegaLiteChart"]',
            '[data-testid="stLineChart"]',
            '[data-testid="stAreaChart"]'
        ];

        document.querySelectorAll(selectors.join(',')).forEach(function(chart) {
            if (chart.dataset.qpWheelBound === "1") return;
            chart.dataset.qpWheelBound = "1";

            chart.addEventListener("wheel", function(event) {
                const button = event.target.closest("button");
                if (button) return;
                event.stopPropagation();
            }, {capture: true, passive: true});
        });
    }

    disableChartWheelZoom();
    new MutationObserver(disableChartWheelZoom).observe(document.body, {
        childList: true,
        subtree: true
    });
})();
</script>

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
    ["Portfolio Analytics", "Research Lab"],
    horizontal=True,
    label_visibility="collapsed",
)


if mode == "Portfolio Analytics":

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
        '<div class="qp-footer">QuantPilot · Portfolio Analytics</div>',
        unsafe_allow_html=True,
    )


else:
    st.markdown(
        """
        <div class="qp-hero">
            <div class="qp-eyebrow">Research Lab</div>
            <div class="qp-hero-title">Plan your portfolio from an investment amount.</div>
            <div class="qp-hero-copy">
                Choose a market, enter your investment amount and horizon, select companies,
                and QuantPilot finds the top portfolio combinations with risk/return projections.
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
            "Position cap defaults to 25% and only rises when needed for smaller groups."
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
                            total_fv = fv_lump + fv_sip if fv_lump is not None and fv_sip is not None else None
                            risk_data.append({
                                "Profile": rp_name,
                                "Max Vol": f"{rpr.max_allowed_volatility:.1%}",
                                "Rec. Return": f"{rpr.recommended_return:.2%}",
                                "Exp. Return": f"{rpr.expected_return:.2%}" if rpr.expected_return is not None else "—",
                                "Exp. Vol": f"{rpr.expected_volatility:.2%}" if rpr.expected_volatility is not None else "—",
                                "Sharpe": f"{rpr.sharpe_ratio:.2f}" if rpr.sharpe_ratio is not None else "—",
                                "Future Value (Lump)": f"{currency_symbol}{fv_lump:,.0f}" if investment_amount > 0 and fv_lump is not None else "—",
                                "Future Value (SIP)": f"{currency_symbol}{fv_sip:,.0f}" if monthly_sip > 0 and fv_sip is not None else "—",
                                "Total Future Value": f"{currency_symbol}{total_fv:,.0f}" if total_fv is not None else "—",
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
                    st.caption("Sharpe is unavailable when annualized volatility is below 0.01%; this avoids unstable ratios at effectively zero risk.")

                    # Allocation for Moderate profile (if feasible)
                    if result.moderate and result.moderate.feasible:
                        # Keep zero-weight stocks visible: the selected combination
                        # and the optimizer's label-to-weight mapping are distinct.
                        weights = build_allocation_weights(
                            result.tickers,
                            result.moderate.weights,
                            result.moderate.cash_weight,
                        ).sort_values(ascending=False)

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
                        profile_failures = [
                            f"{profile.risk_profile}: {profile.warning or 'risk/return constraints not met'}"
                            for profile in (result.conservative, result.moderate, result.aggressive)
                            if profile is not None and not profile.feasible
                        ]
                        reason = result.error or "; ".join(profile_failures) or "No risk profile met its constraints"
                        st.markdown(
                            f"**{' + '.join(result.tickers)}** "
                            f"({result.subset_size} companies, cap {result.max_weight:.0%}) — "
                            f"{reason}"
                        )

            # ---- STOCK TREND ANALYSIS (Post-Build) ----
            st.markdown("<br>", unsafe_allow_html=True)
            section("Stock Trend Analysis", "Multi-horizon returns, technicals, and trend classification for all selected companies")

            try:
                from src.data.market_data import MarketDataAdapter
                adapter = MarketDataAdapter(market)

                trend_data = []
                progress_bar = st.progress(0, text="Analyzing stock trends...")

                for i, ticker in enumerate(selected_tickers):
                    try:
                        data = adapter.load_asset(ticker, allow_download=True)
                        prices = data["Close"].dropna()

                        if len(prices) >= 2:
                            analysis = analyze_stock(prices, ticker)
                            trend_data.append(analysis)
                        else:
                            trend_data.append(None)
                    except Exception as e:
                        trend_data.append(None)
                        st.warning(f"Could not analyze {ticker}: {e}")

                    progress_bar.progress((i + 1) / len(selected_tickers), text=f"Analyzing {ticker}...")

                progress_bar.empty()

                if trend_data:
                    rows = []
                    for analysis in trend_data:
                        if analysis:
                            mh = analysis.multi_horizon
                            tech = analysis.technical

                            trend_color = {
                                Trend.BULLISH: "🟢",
                                Trend.NEUTRAL: "🟡",
                                Trend.BEARISH: "🔴",
                            }
                            trend_badge = f"{trend_color.get(analysis.trend, '⚪')} {analysis.trend.value.upper()}"

                            rows.append({
                                "Ticker": analysis.ticker,
                                "Price": f"{analysis.current_price:,.2f}",
                                "Trend": trend_badge,
                                "Score": analysis.trend_score,
                                "30D": f"{mh.return_30d:.2%}" if mh.return_30d is not None else "—",
                                "90D": f"{mh.return_90d:.2%}" if mh.return_90d is not None else "—",
                                "180D": f"{mh.return_180d:.2%}" if mh.return_180d is not None else "—",
                                "1Y": f"{mh.return_1y:.2%}" if mh.return_1y is not None else "—",
                                "SMA50": f"{tech.sma_50:,.2f}" if tech.sma_50 is not None else "—",
                                "SMA200": f"{tech.sma_200:,.2f}" if tech.sma_200 is not None else "—",
                                "vs 50": tech.price_vs_sma50 or "—",
                                "vs 200": tech.price_vs_sma200 or "—",
                            })

                    if rows:
                        trend_df = pd.DataFrame(rows)

                        def style_trend(val):
                            if "BULLISH" in str(val):
                                return "background-color: #dcfce7; color: #166534; font-weight: 600;"
                            elif "BEARISH" in str(val):
                                return "background-color: #fee2e2; color: #991b1b; font-weight: 600;"
                            elif "NEUTRAL" in str(val):
                                return "background-color: #fef9c3; color: #854d0e; font-weight: 600;"
                            return ""

                        def style_vs_ma(val):
                            if val == "above":
                                return "color: #16a34a; font-weight: 600;"
                            elif val == "below":
                                return "color: #dc2626; font-weight: 600;"
                            return ""

                        # Use .map() instead of deprecated .applymap()
                        styled = trend_df.style.map(style_trend, subset=["Trend"]).map(style_vs_ma, subset=["vs 50", "vs 200"])
                        st.dataframe(styled, width="stretch", hide_index=True, height=min(400, len(rows) * 35 + 40))

                        bullish_count = sum(1 for a in trend_data if a and a.trend == Trend.BULLISH)
                        bearish_count = sum(1 for a in trend_data if a and a.trend == Trend.BEARISH)
                        neutral_count = sum(1 for a in trend_data if a and a.trend == Trend.NEUTRAL)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="qp-card"><div class="qp-card-label">Bullish</div><div class="qp-card-value">{bullish_count}</div></div>', unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'<div class="qp-card"><div class="qp-card-label">Neutral</div><div class="qp-card-value">{neutral_count}</div></div>', unsafe_allow_html=True)
                        with col3:
                            st.markdown(f'<div class="qp-card"><div class="qp-card-label">Bearish</div><div class="qp-card-value">{bearish_count}</div></div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Trend analysis failed: {e}")

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
                        cash_weight=best_result.moderate.cash_weight,
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
                            card("Conservative FV", f"{currency_symbol}{best_result.conservative.future_value_lump_sum + best_result.conservative.future_value_monthly:,.0f}" if best_result.conservative and best_result.conservative.future_value_lump_sum is not None and best_result.conservative.future_value_monthly is not None else "—", f"Max vol {best_result.conservative.max_allowed_volatility:.0%}" if best_result.conservative else "")
                        with cols[1]:
                            card("Moderate FV", f"{currency_symbol}{best_result.moderate.future_value_lump_sum + best_result.moderate.future_value_monthly:,.0f}" if best_result.moderate and best_result.moderate.future_value_lump_sum is not None and best_result.moderate.future_value_monthly is not None else "—", f"Max vol {best_result.moderate.max_allowed_volatility:.0%}" if best_result.moderate else "")
                        with cols[2]:
                            card("Aggressive FV", f"{currency_symbol}{best_result.aggressive.future_value_lump_sum + best_result.aggressive.future_value_monthly:,.0f}" if best_result.aggressive and best_result.aggressive.future_value_lump_sum is not None and best_result.aggressive.future_value_monthly is not None else "—", f"Max vol {best_result.aggressive.max_allowed_volatility:.0%}" if best_result.aggressive else "")

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
                            <b>Position constraint:</b> maximum {best_result.max_weight:.0%} per company.<br>
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
        '<div class="qp-footer">QuantPilot · Portfolio Analytics · '
        'Model-based planning only — not financial advice</div>',
        unsafe_allow_html=True,
    )
