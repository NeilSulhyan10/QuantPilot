# QuantPilot — Goal Planner

QuantPilot's **Goal Planner** is an interactive quantitative portfolio-planning module that translates a future financial target into a model-based portfolio and funding plan.

It is designed as a planning and research tool rather than a return-prediction system or a source of guaranteed investment recommendations.

---

## Overview

The Goal Planner asks the user for:

- Target future amount
- Investment horizon
- Risk tolerance
- Companies to consider

QuantPilot then:

1. Estimates expected returns and covariance from historical data.
2. Determines the maximum feasible portfolio return under the selected assets and risk constraints.
3. Converts that feasibility ceiling into a risk-profile-specific recommended planning return.
4. Constructs a constrained portfolio.
5. Calculates the required initial investment and/or monthly contribution.
6. Generates conservative, expected, and optimistic funding scenarios.
7. Evaluates the generated portfolio historically against Equal Weight and SPY.
8. Presents the results through the interactive Streamlit dashboard.

---

## Goal Planner Workflow

```text
User Goal
   │
   ├── Target Future Amount
   ├── Investment Horizon
   ├── Risk Tolerance
   └── Selected Companies
            │
            ▼
     Historical Market Data
            │
            ▼
    Return & Risk Estimation
            │
      ┌─────┴─────┐
      ▼           ▼
 Expected       Covariance
 Returns          Model
      │           │
      └─────┬─────┘
            ▼
   Feasibility Analysis
            │
            ▼
 Maximum Feasible Return
            │
            ▼
 Risk-Profile Planning Rule
            │
            ▼
   Recommended Return
            │
            ▼
 Constrained Portfolio
            │
      ┌─────┴────────────┐
      ▼                  ▼
 Funding Plan       Historical Analysis
      │                  │
      ▼                  ▼
 Initial / Monthly   Equal Weight / SPY
      │                  │
      └────────┬─────────┘
               ▼
          Goal Planner UI
```

---

## User Inputs

### Target Future Amount

The amount the user wants to accumulate at the end of the investment horizon.

### Investment Horizon

The number of years available to reach the target.

### Risk Tolerance

The planner currently supports three risk profiles:

| Profile | Maximum Annual Volatility |
|---|---:|
| Conservative | 12% |
| Moderate | 18% |
| Aggressive | 25% |

### Selected Companies

The user selects the companies that QuantPilot is allowed to consider when constructing the portfolio.

The current dashboard uses the 20-stock QuantPilot MVP universe:

```text
AAPL  MSFT  NVDA  AVGO  GOOGL
AMZN  META  JPM   V     MA
JNJ   UNH   XOM   CVX   PG
KO    COST  CAT   WMT   HD
```

The portfolio optimizer currently enforces a **10% maximum weight per company**. Therefore, at least 10 valid companies are required for a fully invested long-only portfolio.

---

# Return Estimation

The Goal Planner currently uses the latest **60 trading days** for its return and covariance estimation.

Daily expected returns are estimated from the selected assets and annualized using 252 trading periods per year.

Conceptually:

```text
Daily historical returns
        │
        ▼
60-day estimation window
        │
        ├── Mean daily returns
        │
        └── Daily covariance
                │
                ▼
        Annualized estimates
```

The Goal Planner is therefore sensitive to the most recent historical estimation window.

---

# Risk Model

The covariance matrix is estimated from the same 60-day historical window.

The covariance matrix is used by the optimizer to measure portfolio risk and enforce the maximum annual volatility associated with the selected risk profile.

The optimization is performed using the portfolio covariance structure rather than simply ranking companies by historical return.

---

# Portfolio Constraints

The Goal Planner constructs a portfolio subject to:

- Long-only positions
- Fully invested portfolio
- 100% total portfolio weight
- Maximum 10% weight per company
- Risk-profile-specific annual volatility constraint
- Minimum historical data requirement

The optimizer seeks a portfolio that satisfies the selected target return while controlling portfolio variance.

---

# Maximum Feasible Return

Before constructing the recommended portfolio, QuantPilot determines the **maximum feasible return** under the selected assets and portfolio constraints.

This represents an optimization ceiling:

> The highest expected return that the model can achieve while satisfying the selected risk and portfolio constraints.

It is **not** a forecast of the return the portfolio will actually earn.

For example, if the selected universe and risk constraints allow a maximum feasible return of 20%, that does not mean QuantPilot expects the portfolio to earn 20% in the future.

It only means that 20% is the estimated return ceiling produced by the optimization problem under the current model assumptions.

---

# Recommended Planning Return

The Goal Planner does not directly use the maximum feasible return as its planning assumption.

Instead, the feasibility ceiling is converted into a risk-profile-specific recommended return.

Current planning rules:

| Risk Profile | Fraction of Maximum Feasible Return |
|---|---:|
| Conservative | 50% |
| Moderate | 70% |
| Aggressive | 85% |

Conceptually:

```text
Maximum Feasible Return
          │
          ▼
 Risk Profile
          │
          ▼
Planning Fraction
          │
          ▼
Recommended Return
```

These fractions are **product-level planning conventions**, not statistically estimated forecasts.

They are deliberately separated from the optimization engine so that future versions can replace or calibrate them without rewriting the portfolio optimizer.

---

# Funding Calculations

Once a recommended planning return has been determined, QuantPilot calculates two alternative funding paths.

## 1. Required Initial Investment

This answers:

> How much capital would need to be invested today to reach the target under the assumed annual return?

The calculation uses compound growth.

## 2. Required Monthly Contribution

This answers:

> How much would need to be invested at the end of each month to reach the target under the assumed annual return?

The monthly contribution calculation assumes **end-of-month contributions** and converts the annual planning return into an effective monthly return.

These are alternative funding paths.

They should not be added together.

---

# Important Planning Distinction

The Goal Planner performs deterministic financial mathematics using a model-derived planning return.

Therefore:

```text
Target + Horizon
        ↓
Recommended Planning Return
        ↓
Required Capital
```

The funding requirement is mathematically correct **conditional on the assumed return**.

It does not mean the market will actually deliver that return.

For this reason, the planner explicitly distinguishes between:

- Mathematical funding requirement
- Model-based planning assumption
- Actual future investment performance

---

# Scenario Analysis

The planner generates three funding scenarios:

### Conservative

A lower assumed return than the recommended planning return.

### Expected

The risk-profile-specific recommended planning return.

### Optimistic

A higher assumed return than the recommended planning return.

The scenario returns are currently constructed around the portfolio's expected volatility:

```text
Conservative Return
    = Recommended Return − Portfolio Volatility

Expected Return
    = Recommended Return

Optimistic Return
    = Recommended Return + Portfolio Volatility
```

The lower scenario is bounded at -99%.

These scenarios are **planning assumptions**, not probability distributions, confidence intervals, or forecasts.

They are intended to show how sensitive the required funding amount is to changes in the assumed return.

---

# Recommended Portfolio

The planner outputs portfolio weights for the selected companies.

For the required initial investment, the dashboard also converts each weight into an approximate capital allocation:

```text
Initial Allocation
=
Portfolio Weight × Required Initial Investment
```

The allocation therefore provides two views:

- Portfolio percentage weight
- Approximate initial capital allocation

Tiny numerical optimizer weights are excluded from the displayed allocation.

---

# Historical Hypothetical Performance

The Goal Planner includes a historical analysis of the generated portfolio.

This analysis is intentionally different from the main QuantPilot research backtest.

## How it works

1. The Goal Planner constructs the current portfolio using the latest estimation window.
2. The resulting portfolio weights are held as the generated portfolio.
3. Those weights are evaluated against historical selected-company returns.
4. Equal Weight and SPY are evaluated over the common available historical period.
5. CAGR, volatility, Sharpe ratio, Sortino ratio, and maximum drawdown are displayed.

---

# Important Historical Backtest Limitation

The Goal Planner's historical analysis is **not a walk-forward goal optimization**.

It answers:

> **"How would the currently generated portfolio have behaved historically?"**

It does **not** answer:

> **"How would QuantPilot have generated the portfolio historically at each point in time?"**

The current recommended weights are generated using the latest estimation information and then applied historically.

Therefore, this analysis should be interpreted as a **hypothetical historical stress test**, not as evidence of historical out-of-sample portfolio-selection performance.

The main QuantPilot research engine provides the stronger walk-forward backtest for that purpose.

---

# Benchmark Comparison

The Goal Planner compares the generated portfolio with:

### Equal Weight

An equal-weight portfolio constructed from the selected companies.

### SPY

The SPY benchmark is loaded from the project's benchmark data source.

All three return series are aligned to a **common evaluation period** before calculating comparative performance metrics.

This prevents differences in available dates from creating an inconsistent comparison.

---

# Historical Metrics

The Goal Planner reports:

### CAGR

Compound annual growth rate over the evaluation period.

### Annualized Volatility

Annualized standard deviation of daily returns using 252 trading periods per year.

### Sharpe Ratio

Risk-adjusted return relative to the configured zero risk-free-rate assumption.

### Sortino Ratio

Downside-risk-adjusted return.

### Maximum Drawdown

Largest historical peak-to-trough decline.

These metrics describe historical behavior only.

---

# Main Research Backtest vs Goal Planner

QuantPilot contains two distinct portfolio-analysis concepts.

| Feature | Research Mode | Goal Planner |
|---|---|---|
| Portfolio generation | Walk-forward | Current/latest model |
| Historical evaluation | Out-of-sample | Hypothetical |
| Re-optimization | Yes | No |
| Drift-aware | Yes | Yes for generated portfolio |
| Transaction costs | Modeled | Modeled for generated portfolio |
| Goal funding | No | Yes |
| Risk-profile constraints | Research configuration | Yes |
| Benchmark comparison | Yes | Yes |

This separation is intentional.

The Research Mode evaluates the quantitative strategy.

The Goal Planner translates a current model portfolio into a financial planning scenario.

---

# Methodological Limitations

## Survivorship Bias

The MVP uses a fixed current-stock universe.

Companies that disappeared, were acquired, failed, or were removed from the universe historically are not represented.

This can introduce survivorship bias into historical analyses.

## Parameter Sensitivity

Results can change depending on:

- Expected-return window
- Covariance window
- Risk-aversion parameters
- Position limits
- Rebalancing frequency
- Transaction costs
- Slippage assumptions

These parameters should therefore not be interpreted as universally optimal.

## Historical Estimates

Expected returns and covariance are estimated from historical market data.

Future market behavior may differ substantially from historical behavior.

## Transaction Costs and Slippage

Transaction costs and slippage are modeled assumptions rather than measurements from actual live executions.

## Goal Planner Return Assumptions

The recommended return is a model-derived planning assumption.

The risk-profile fractions are product conventions rather than statistically validated forecasts.

## Goal Planner Historical Analysis

The Goal Planner historical analysis uses the currently generated portfolio weights and is not a walk-forward historical optimization.

## Future Performance

Historical backtest performance does not guarantee future results.

---

# Machine Learning Component

QuantPilot also contains an experimental Ridge regression component for expected-return estimation.

The ML model uses lagged historical features and forward return targets.

Training is restricted to information available before the prediction period to reduce look-ahead leakage.

The ML component is intentionally treated as an experimental component rather than the central decision engine.

The portfolio optimizer and risk model remain the primary decision-making components.

The ML comparison did not demonstrate a sufficiently clear improvement in risk-adjusted performance to justify replacing the baseline expected-return model.

---

# Look-Ahead Bias

Avoiding look-ahead bias is a core design principle of QuantPilot.

The main research backtest follows a walk-forward structure:

```text
Historical Information
        │
        ▼
Estimate Model Inputs
        │
        ▼
Generate Portfolio
        │
        ▼
Future / Out-of-Sample Period
        │
        ▼
Evaluate Performance
```

Information from future evaluation periods should not be used to construct historical portfolio decisions.

The Goal Planner's static historical analysis is explicitly labeled differently because it is not a walk-forward simulation.

---

# Current Goal Planner Architecture

The Goal Planner is implemented through dedicated modules:

```text
src/goals/
├── assets.py
├── backtest.py
├── feasibility.py
├── goal_math.py
├── optimizer.py
├── planner.py
├── portfolio.py
└── scenarios.py
```

The dashboard integrates these components into an interactive Streamlit interface.

---

# Product Philosophy

QuantPilot intentionally separates:

```text
Prediction
     ≠
Planning
     ≠
Optimization
     ≠
Historical Evaluation
```

The system does not claim that an optimizer can predict future market returns with certainty.

Instead, it uses historical estimates and explicit mathematical constraints to construct portfolios and answer conditional planning questions.

The central question is:

> **Given these assumptions, constraints, assets and risk tolerance, what portfolio and funding path does the model produce?**

---

# Disclaimer

QuantPilot is an educational, research, and portfolio-planning project.

Its outputs are model-based calculations and historical analyses. They are not personalized financial advice, investment guarantees, or predictions of future market performance.

Users should independently evaluate assumptions, risks, taxes, liquidity requirements, inflation, and other financial considerations before making investment decisions.
