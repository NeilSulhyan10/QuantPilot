# QuantPilot — Adaptive Quantitative Portfolio Manager

QuantPilot is a quantitative portfolio management research system that combines statistical estimation, risk modeling, mathematical portfolio optimization, walk-forward backtesting, and an experimental machine-learning component.

The project is designed around a portfolio-management question rather than a stock-price prediction question:

> Can systematic portfolio construction produce competitive risk-adjusted performance while controlling portfolio concentration, volatility, drawdown, turnover, and trading friction?

The core QuantPilot strategy uses historical expected returns and a rolling covariance matrix inside a constrained mean-variance optimizer.

Machine learning is treated as an experimental expected-return component, not as the central portfolio construction mechanism.

---

## 1. Research Objective

QuantPilot investigates a complete systematic portfolio-management pipeline:

1. Market-data acquisition
2. Data validation
3. Return calculation
4. Expected-return estimation
5. Covariance estimation
6. Constrained portfolio optimization
7. Portfolio drift simulation
8. Periodic rebalancing
9. Transaction-cost modeling
10. Slippage modeling
11. Walk-forward backtesting
12. Benchmark comparison
13. Portfolio diagnostics
14. Robustness analysis
15. Experimental machine learning
16. Visualization and dashboard reporting

The project emphasizes realistic research methodology and attempts to avoid:

- look-ahead bias
- target leakage
- unrealistic portfolio constraints
- frictionless trading assumptions
- in-sample-only evaluation
- selecting a strategy solely because of maximum historical return

---

## 2. System Architecture

```text
                         Market Data
                              |
                              v
                       Data Validation
                              |
                              v
                       Price Processing
                              |
                              v
                       Return Calculation
                              |
                +-------------+-------------+
                |                           |
                v                           v
        Expected Returns                Covariance
                |                           |
                +-------------+-------------+
                              |
                              v
                  Mean-Variance Optimization
                              |
                              v
                     Portfolio Weights
                              |
                              v
                  Walk-Forward Backtesting
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
          Weight Drift   Transaction Costs   Slippage
             |                |                |
             +----------------+----------------+
                              |
                              v
                         Net Returns
                              |
                              v
                    Performance Evaluation
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
      Benchmarks         Diagnostics         Visualizations
          |                                       |
          +-------------------+-------------------+
                              |
                              v
                       Streamlit Dashboard
```

---

## 3. Research Universe

The MVP uses a fixed universe of 20 liquid US equities:

```text
AAPL  MSFT  NVDA  AVGO  GOOGL
AMZN  META  JPM   V     MA
JNJ   UNH   XOM   CVX   PG
KO    COST  CAT   WMT   HD
```

The canonical dataset begins in January 2015.

The research data cutoff is:

```text
2015-01-01 -> 2026-09-01
```

Daily market data is used throughout the core backtest.

### Survivorship Bias

The universe is a fixed set of selected current liquid US equities rather than a point-in-time historical universe.

Consequently, the research has a survivorship-bias limitation.

This is explicitly documented rather than treated as if the universe had been historically unbiased.

A future version could use historical point-in-time index constituents or another survivorship-bias-controlled universe.

---

## 4. Data Pipeline

Raw market data is downloaded and stored separately from processed data.

```text
data/
├── raw/
└── processed/
```

The processing pipeline validates:

- required OHLCV fields
- date ordering
- missing values
- positive prices
- valid trading dates
- volume values

The processed data is then consumed by the feature and portfolio-construction layers.

Price data is never forward-filled to fabricate missing observations.

For cross-sectional return calculations, the common intersection of valid asset observations is used.

---

## 5. Return Modeling

The baseline uses simple daily returns.

For a price series P:

```text
r_t = P_t / P_(t-1) - 1
```

Log returns are also implemented for feature and research flexibility.

The return matrix contains one column per asset and is aligned across the research universe.

---

## 6. Expected-Return Model

The baseline QuantPilot V2 expected-return estimator uses:

- 60-trading-day historical mean returns
- 50% cross-sectional shrinkage

The historical mean provides a simple statistical estimate while shrinkage pulls individual asset estimates toward the cross-sectional universe mean.

The objective is to reduce the instability associated with estimating expected returns independently for every asset.

Baseline configuration:

```text
Expected-return window: 60 trading days
Shrinkage alpha:         0.50
```

This is deliberately interpretable rather than relying on a highly complex forecasting model.

---

## 7. Covariance Model

Portfolio risk is estimated using a:

```text
60-trading-day rolling covariance matrix
```

The covariance matrix is constructed from historical returns available at each rebalance date.

Covariance matrices are checked for:

- finite values
- symmetry
- numerical validity
- positive semi-definiteness

The covariance matrix is then passed to the portfolio optimizer.

---

## 8. Portfolio Optimization

The central portfolio construction engine is a constrained mean-variance optimizer implemented using CVXPY.

Conceptually, the optimizer balances expected portfolio return against portfolio variance.

The optimization is performed at each walk-forward rebalance date.

### Portfolio Constraints

QuantPilot uses:

```text
Long-only
Fully invested
Maximum individual position = 10%
```

Therefore:

```text
w_i >= 0

sum(w_i) = 1

w_i <= 0.10
```

These constraints prevent short positions, ensure that all capital is allocated, and limit concentration in any single security.

The optimizer uses the CLARABEL solver with tightened feasibility and optimality tolerances.

Solver-level floating-point noise is treated separately from economically meaningful portfolio constraints.

---

## 9. Rebalancing

The baseline strategy rebalances monthly.

Rebalance dates correspond to the last available trading day of each month, subject to the availability of valid expected-return and covariance estimates.

Baseline configuration:

```text
Rebalancing frequency: Monthly
```

Between rebalance dates, the portfolio is not artificially kept at its target weights.

Instead, holdings drift according to realized asset returns.

This creates a more realistic portfolio simulation.

---

## 10. Portfolio Drift

Suppose an asset has beginning-of-day weight w_i and return r_i.

After the day's returns, portfolio weights are updated according to the relative growth of each holding.

Conceptually:

```text
Target Portfolio
       |
       v
Daily Asset Returns
       |
       v
Portfolio Return
       |
       v
Drifted Portfolio Weights
       |
       v
Next Trading Day
       |
       v
Rebalance when scheduled
```

This distinction is important because a monthly-rebalanced portfolio should not behave like a continuously rebalanced portfolio.

---

## 11. Transaction Costs

The baseline transaction-cost assumption is:

```text
10 basis points
```

Transaction costs are calculated from portfolio turnover.

The purpose is to prevent the strategy from receiving an unrealistic frictionless trading advantage.

---

## 12. Slippage

The baseline slippage assumption is:

```text
5 basis points
```

Slippage is modeled separately from transaction costs.

Therefore the baseline trading-friction assumptions are:

```text
Transaction costs: 10 bps
Slippage:           5 bps
Total modeled cost: 15 bps
```

These are research assumptions and are not intended to represent guaranteed real-world execution costs.

---

## 13. Walk-Forward Backtesting

QuantPilot uses a walk-forward design.

At each rebalance date:

```text
Historical information available at t
                |
                v
       Expected returns
                |
                v
          Covariance
                |
                v
       Portfolio optimizer
                |
                v
        Target weights
                |
                v
      Portfolio holding period
                |
                v
       Weight drift + costs
                |
                v
       Next rebalance date
```

The optimizer does not receive future realized returns when constructing historical portfolios.

This design is intended to avoid look-ahead bias.

---

## 14. Machine-Learning Leakage Prevention

The ML component uses a strict walk-forward prediction process.

For a prediction date t, the training data is restricted to observations whose forward-return targets are fully known by t.

The implementation therefore prevents future target information from entering the training set.

This is particularly important for financial machine learning because future-return targets naturally overlap the prediction date.

The ML implementation was tested specifically for this leakage condition.

---

## 15. Benchmarks

QuantPilot is compared against three benchmark strategies.

### Equal Weight

Equal allocation across the same 20-stock universe.

The implementation is drift-aware and rebalances according to the same baseline schedule.

### Buy & Hold

The portfolio starts equally weighted and then remains un-rebalanced.

Weights therefore drift naturally as asset prices change.

### SPY

SPY is used as a broad US equity-market benchmark.

All strategies are aligned to a common evaluation period before the final comparison.

---

## 16. Evaluation Metrics

QuantPilot evaluates:

### Cumulative Return

Total growth of the portfolio over the evaluation period.

### CAGR

Compound annual growth rate.

### Annualized Volatility

Annualized standard deviation of daily returns.

### Sharpe Ratio

Risk-adjusted return relative to return volatility.

### Sortino Ratio

Risk-adjusted return using downside volatility.

### Maximum Drawdown

Largest peak-to-trough decline during the evaluation period.

Additional portfolio statistics include:

- turnover
- transaction costs
- slippage
- number of active positions
- effective number of stocks
- average weights
- maximum weights
- position-cap frequency

---

## 17. Canonical Performance

Evaluation period:

```text
2015-04-01 -> 2026-08-27
```

| Strategy | Cumulative Return | CAGR | Volatility | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|---:|
| QuantPilot V2 | 719.09% | 20.25% | 16.45% | 1.26 | 1.19 | -24.71% |
| Equal Weight | 996.09% | 23.36% | 17.90% | 1.32 | 1.24 | -31.18% |
| Buy & Hold | 2945.41% | 34.92% | 28.57% | 1.25 | 1.21 | -37.87% |
| SPY | 355.56% | 14.22% | 17.76% | 0.88 | 0.82 | -33.44% |

### Interpretation

The Buy & Hold portfolio achieved the highest historical CAGR.

However, it also experienced:

- the highest volatility
- the largest maximum drawdown

QuantPilot's CAGR was lower, but its volatility was considerably lower and its maximum drawdown was approximately 13 percentage points smaller than Buy & Hold.

The Sharpe ratio of QuantPilot was also slightly higher than Buy & Hold.

Equal Weight performed very strongly and exceeded QuantPilot on both CAGR and Sharpe in this particular experiment.

Therefore, the evidence does not support a claim that the optimizer is universally superior to simple strategies.

Instead, it demonstrates that the portfolio construction process can produce a materially different risk/return profile.

---

## 18. QuantPilot Portfolio Diagnostics

The canonical QuantPilot portfolio has approximately:

```text
Average active positions: 11.94
Average effective stocks: 10.57
Position cap:             10%
```

Largest average portfolio weights:

| Asset | Average Weight |
|---|---:|
| JNJ | 7.44% |
| KO | 6.96% |
| PG | 6.96% |
| WMT | 6.72% |
| COST | 6.36% |
| UNH | 6.12% |
| JPM | 5.42% |
| HD | 5.27% |
| CAT | 4.87% |
| XOM | 4.86% |

The optimizer therefore does not continuously allocate the entire portfolio to a handful of assets.

The position cap and risk model produce a diversified but selective portfolio.

---

## 19. Robustness Analysis

The project does not rely solely on one parameter configuration.

Multiple experiments were conducted across:

- estimation window
- risk aversion
- position cap
- rebalance frequency
- transaction costs
- market regimes

The goal is to understand the behavior of the system rather than simply identify the highest historical return.

---

## 20. Estimation-Window Sensitivity

Tested estimation windows included:

```text
60 days
120 days
252 days
```

The 252-day configuration produced the strongest average risk-adjusted results among the tested estimation-window configurations.

Approximate average results:

| Window | Avg CAGR | Avg Volatility | Avg Sharpe | Avg Max DD | Avg Turnover |
|---|---:|---:|---:|---:|---:|
| 60 days | 21.35% | 17.04% | 1.278 | -27.81% | 21.13% |
| 120 days | 21.41% | 17.05% | 1.280 | -27.52% | 17.74% |
| 252 days | 21.72% | 16.65% | 1.320 | -26.34% | 16.33% |

The 60-day window remains the baseline because the objective is not to select the single best historical parameter configuration.

---

## 21. Risk-Aversion Sensitivity

Tested risk-aversion values included:

```text
5
10
20
```

Approximate average behavior:

| Risk Aversion | Avg CAGR | Avg Volatility | Avg Sharpe |
|---|---:|---:|---:|
| 5 | 22.93% | 17.77% | 1.311 |
| 10 | 21.46% | 16.83% | 1.298 |
| 20 | 20.09% | 16.15% | 1.269 |

Higher risk aversion reduced volatility but also reduced return.

The baseline value of 10 therefore represents a reasonable compromise rather than a parameter chosen solely because it maximized CAGR.

---

## 22. Position-Cap Sensitivity

Position caps tested included:

```text
5%
10%
15%
```

A 5% cap is effectively an equal-weight portfolio because:

```text
20 stocks x 5% = 100%
```

Therefore, the 5% configuration should not be interpreted as evidence that mean-variance optimization itself is superior.

The 10% cap is retained as the baseline because it permits meaningful optimization while controlling concentration.

---

## 23. Rebalancing Sensitivity

The baseline monthly schedule was compared with:

```text
Monthly
Quarterly
Semiannual
Annual
```

Approximate results:

| Frequency | CAGR | Sharpe | Max Drawdown | Avg Turnover |
|---|---:|---:|---:|---:|
| Monthly | 20.25% | 1.261 | -24.71% | 26.10% |
| Quarterly | 17.47% | 1.045 | -31.76% | 47.40% |
| Semiannual | 20.66% | 1.158 | -31.76% | 48.17% |
| Annual | 21.73% | 1.195 | -31.76% | 51.41% |

Less frequent rebalancing reduces the number of trading events but allows greater portfolio drift.

The monthly schedule is retained as the baseline because it provides a stronger balance between portfolio control and trading activity in this experiment.

---

## 24. Transaction-Cost Sensitivity

The strategy was evaluated under multiple transaction-cost and slippage assumptions.

| Cost Scenario | Transaction Cost | Slippage | CAGR | Sharpe |
|---|---:|---:|---:|---:|
| Frictionless | 0 bps | 0 bps | 20.82% | 1.291 |
| Low | 5 bps | 2.5 bps | 20.53% | 1.276 |
| Baseline | 10 bps | 5 bps | 20.25% | 1.261 |
| High | 20 bps | 10 bps | 19.68% | 1.231 |
| Very High | 50 bps | 25 bps | 17.99% | 1.140 |

Compared with the frictionless case, the baseline assumptions reduce CAGR by approximately 0.57 percentage points and Sharpe by approximately 0.03.

This confirms that trading friction has a measurable effect on the strategy.

---

## 25. Market-Regime Analysis

QuantPilot was also examined across several historical market regimes.

| Period | QuantPilot Return | QuantPilot Vol | QuantPilot Sharpe | QuantPilot MDD | SPY Return |
|---|---:|---:|---:|---:|---:|
| 2018 Correction | -3.71% | 18.57% | -0.122 | -20.48% | -4.57% |
| 2020 COVID | 29.59% | 31.75% | 1.013 | -24.71% | 18.33% |
| 2022 Bear Market | 6.33% | 17.83% | 0.452 | -12.72% | -18.18% |
| 2023-24 Bull Market | 54.20% | 12.77% | 1.852 | -11.27% | 57.58% |
| 2025-26 | 19.03% | 13.53% | 0.889 | -18.90% | 33.82% |

The strategy showed particularly strong relative behavior during adverse
conditions, most notably during 2022.

It also lagged the market during some strong upside periods.

This is consistent with the strategy's objective of balancing return and
risk rather than maximizing equity-market beta.

---

## 26. Machine-Learning Experiment

An experimental Ridge-regression expected-return model was implemented.

Features include:

- 5-day return
- 20-day return
- 20-day volatility
- 60-day volatility
- 60-day mean return

The model is standardized using a pipeline and trained using strictly
walk-forward observations.

The ML model predicts a forward return over a 21-trading-day horizon and
converts the prediction into a daily-equivalent expected return for portfolio
optimization.

### ML vs V2

Common evaluation period:

```text
2016-05-02 -> 2026-08-27
```

| Strategy | CAGR | Volatility | Sharpe | Sortino | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| QuantPilot V2 | 21.20% | 16.45% | 1.312 | 1.233 | -24.71% |
| ML Expected Return | 21.44% | 16.91% | 1.293 | 1.220 | -32.64% |

The ML model produced a slightly higher CAGR but also:

- higher volatility
- lower Sharpe
- lower Sortino
- materially larger drawdown
- higher turnover

Therefore, ML is currently treated as an experimental component rather than
the production baseline.

This result is an important part of the research: more predictive complexity
does not automatically produce a better portfolio.

---

## 27. ML Annual Comparison

The ML model outperformed V2 in 7 of the 11 evaluated calendar periods and
underperformed in 4.

Approximate annual differences in ML return relative to V2:

| Period | ML vs V2 |
|---|---:|
| 2016 | +0.51 pp |
| 2017 | +7.45 pp |
| 2018 | +2.73 pp |
| 2019 | -3.34 pp |
| 2020 | -5.16 pp |
| 2021 | +0.39 pp |
| 2022 | -5.72 pp |
| 2023 | +5.92 pp |
| 2024 | +2.34 pp |
| 2025 | -4.12 pp |
| 2026 | +2.56 pp |

The mixed results reinforce the decision to keep the statistical V2 model as
the primary baseline.

---

## 28. Project Structure

```text
quantpilot/
│
├── configs/
│   ├── quantpilot.yaml
│   └── universe.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── results/
│   ├── backtests/
│   ├── comparisons/
│   └── figures/
│
├── src/
│   ├── backtesting/
│   │   ├── benchmark.py
│   │   ├── buy_and_hold.py
│   │   ├── config.py
│   │   ├── costs.py
│   │   ├── engine.py
│   │   ├── returns.py
│   │   └── slippage.py
│   │
│   ├── data/
│   │   ├── benchmark_loader.py
│   │   ├── benchmark_reader.py
│   │   ├── check_coverage.py
│   │   ├── loader.py
│   │   ├── processor.py
│   │   ├── reader.py
│   │   ├── validate_universe.py
│   │   └── validation.py
│   │
│   ├── evaluation/
│   │   ├── benchmark_comparison.py
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── portfolio_diagnostics.py
│   │
│   ├── features/
│   │   ├── covariance.py
│   │   ├── returns.py
│   │   ├── state.py
│   │   └── statistics.py
│   │
│   ├── models/
│   │   ├── expected_returns.py
│   │   ├── ml_features.py
│   │   └── ml_expected_returns.py
│   │
│   ├── optimization/
│   │   └── mean_variance.py
│   │
│   ├── portfolio/
│   │   ├── drift.py
│   │   ├── rebalance.py
│   │   ├── schedule.py
│   │   └── weights.py
│   │
│   └── config.py
│
├── tests/
│
├── dashboard.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 29. Technology Stack

QuantPilot is implemented in Python.

Core technologies:

- Python
- pandas
- NumPy
- SciPy
- CVXPY
- scikit-learn
- yfinance
- PyArrow
- matplotlib
- Plotly
- Streamlit
- pytest
- Git

The project separates data, features, models, optimization, backtesting,
evaluation, and portfolio logic into dedicated modules.

---

## 30. Configuration

The canonical experiment is controlled through:

```text
configs/quantpilot.yaml
```

Important baseline parameters include:

```yaml
returns:
  method: "simple"

expected_returns:
  model: "v2"
  window: 60
  shrinkage_alpha: 0.5

covariance:
  window: 60

optimizer:
  method: "mean_variance"
  risk_aversion: 10.0
  max_weight: 0.10
  long_only: true
  fully_invested: true

rebalancing:
  frequency: "monthly"

costs:
  transaction_cost_bps: 10
  slippage_bps: 5

backtest:
  walk_forward: true
  drift_aware: true

ml:
  enabled: true
  model: "ridge"
  min_train_size: 252
  horizon: 21
  alpha: 1.0
```

The configuration is loaded through typed Python dataclasses and validated
before the backtest begins.

---

## 31. Installation

Clone the repository and enter the project directory.

```bash
git clone https://github.com/NeilSulhyan10/QuantPilot
cd quantpilot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 32. Data Preparation

The data pipeline downloads the configured universe and stores raw data under:

```text
data/raw/
```

Processed data is stored under:

```text
data/processed/
```

The benchmark SPY data is handled through the benchmark data pipeline.

After downloading and processing the data, the project can run the
canonical backtest.

---

## 33. Running the Canonical Backtest

Run:

```bash
python main.py
```

The script:

1. Loads the canonical configuration
2. Loads the 20-stock universe
3. Builds the return matrix
4. Calculates expected returns
5. Builds rolling covariance matrices
6. Runs the walk-forward QuantPilot strategy
7. Runs Equal Weight
8. Runs Buy & Hold
9. Loads SPY
10. Aligns all strategies
11. Calculates performance metrics
12. Generates portfolio diagnostics
13. Generates figures
14. Saves persistent research outputs

The expected canonical evaluation window is:

```text
Start: 2015-04-01
End:   2026-08-27
```

---

## 34. Running the Dashboard

The project includes a Streamlit dashboard.

Run:

```bash
streamlit run dashboard.py
```

The dashboard reads persisted research results rather than rerunning the
entire backtest.

This separates:

```text
Research computation
```

from:

```text
Result visualization
```

The dashboard can be used to inspect portfolio performance and diagnostics.

---

## 35. Generated Results

The canonical backtest produces files under:

```text
results/
├── backtests/
├── comparisons/
└── figures/
```

### Backtest Outputs

```text
quantpilot_v2_returns.csv
quantpilot_v2_weights.csv
quantpilot_v2_turnover.csv
quantpilot_v2_transaction_costs.csv
quantpilot_v2_slippage.csv
quantpilot_v2_average_weights.csv
quantpilot_v2_maximum_weights.csv
quantpilot_v2_active_positions.csv
quantpilot_v2_effective_stocks.csv
quantpilot_v2_cap_hit_frequency.csv
quantpilot_v2_turnover_statistics.csv
```

### Comparison Output

```text
strategy_comparison.csv
```

### Figures

```text
cumulative_returns.png
drawdown.png
portfolio_weights.png
turnover.png
```

These outputs make the research process inspectable without requiring every
result to be recomputed interactively.

---

## 36. Testing

The project includes a comprehensive test suite covering:

- configuration validation
- data validation
- return calculations
- covariance calculations
- expected returns
- optimization
- portfolio constraints
- portfolio drift
- rebalancing
- transaction costs
- slippage
- benchmark behavior
- walk-forward behavior
- ML leakage prevention
- evaluation metrics
- plotting
- reproducibility
- persisted output validation

Run the complete test suite with:

```bash
python -m pytest
```

The current project test suite contains:

```text
156 tests
```

All tests pass for the canonical project state.

---

## 37. Reproducibility

The project includes explicit reproducibility checks.

The canonical configuration uses a frozen data cutoff:

```text
2026-09-01
```

The final output files are validated for:

- existence
- expected strategy names
- required metrics
- non-empty returns
- valid portfolio weights
- weight-sum constraints
- position-cap constraints
- increasing dates

The canonical pipeline was executed twice and produced identical console
output.

This provides a basic reproducibility check for the final research pipeline.

---

## 38. Numerical Stability

Portfolio optimization is subject to floating-point solver tolerances.

The project therefore uses tightened CLARABEL tolerances for the final
optimizer.

The implementation avoids arbitrary post-solver clipping or normalization
because the 10% position cap is tight:

```text
20 assets × 10% = 100%
```

Artificially modifying solver weights after optimization can therefore
violate the mathematical constraints.

Instead, numerical validity is handled at the optimizer and validation
boundaries.

---

## 39. Research Limitations

QuantPilot is a research project and has several important limitations.

### 39.1 Survivorship Bias

The universe consists of selected current equities rather than a historical
point-in-time universe.

### 39.2 Expected-Return Estimation

Historical mean returns are noisy estimates.

Shrinkage reduces instability but does not make expected returns predictable.

### 39.3 Covariance Estimation

Rolling covariance estimates can be unstable, especially during regime
changes.

### 39.4 Transaction Costs

The cost model uses fixed assumptions rather than security-specific
execution costs.

### 39.5 Slippage

Slippage is modeled as a fixed rate and does not explicitly model:

- bid-ask spread
- market impact
- liquidity
- order size
- intraday execution

### 39.6 Market Impact

The current implementation does not model nonlinear market impact.

### 39.7 Taxation

Taxes are not modeled.

### 39.8 Corporate Actions

The research relies on adjusted market data and therefore does not attempt
to reproduce every operational detail of historical portfolio accounting.

### 39.9 Parameter Stability

The optimal configuration can vary across market environments.

Robustness experiments reduce the risk of relying on one configuration but
do not eliminate parameter uncertainty.

### 39.10 Backtest-to-Live Gap

A historical backtest is not equivalent to live trading.

Real-world execution can differ because of:

- liquidity
- spreads
- latency
- market impact
- trading hours
- execution quality
- data errors
- operational constraints

---

## 40. What QuantPilot Is Not

QuantPilot is not intended to be:

- a guaranteed profitable trading system
- a stock-price prediction engine
- financial advice
- a production brokerage system
- a high-frequency trading strategy
- evidence that machine learning always improves portfolio performance

The project is primarily a quantitative research and engineering system.

---

## 41. Why Mean-Variance Optimization?

Mean-variance optimization provides a clear mathematical framework for
portfolio construction.

It explicitly represents the trade-off between:

```text
Expected Return
       vs.
Portfolio Risk
```

It also makes portfolio constraints explicit.

This provides an interpretable foundation on which more advanced techniques
can later be evaluated.

The optimizer therefore acts as the portfolio-construction layer, while
statistical or machine-learning models can be substituted into the
expected-return estimation layer.

---

## 42. Why Machine Learning Is Experimental

A common failure mode in quantitative projects is to make machine learning
the center of the system without first establishing a strong financial
baseline.

QuantPilot follows the opposite approach.

The research progression is:

```text
Statistical Baseline
        |
        v
Risk Model
        |
        v
Portfolio Optimizer
        |
        v
Realistic Backtest
        |
        v
Benchmark Comparison
        |
        v
ML Experiment
```

This makes it possible to determine whether ML adds genuine portfolio value
rather than simply improving a prediction metric.

The current ML experiment does not clearly outperform the V2 baseline on
risk-adjusted performance, so it remains experimental.

---

## 43. Research Philosophy

The central philosophy of QuantPilot is:

> Better quantitative finance is not simply better prediction.

A portfolio strategy must also consider:

- estimation uncertainty
- diversification
- risk
- concentration
- turnover
- transaction costs
- slippage
- rebalancing
- drawdowns
- robustness
- out-of-sample evaluation

The project therefore treats portfolio construction as a complete system.

---

## 44. Future Improvements

Potential future extensions include:

### Data

- point-in-time historical universes
- additional asset classes
- sector metadata
- liquidity measures
- fundamentals

### Risk

- covariance shrinkage
- factor risk models
- exponentially weighted covariance
- downside-risk models
- CVaR
- volatility targeting

### Optimization

- turnover constraints
- sector constraints
- factor exposure constraints
- transaction-cost-aware optimization
- robust optimization
- risk parity

### Expected Returns

- factor models
- momentum signals
- quality signals
- valuation signals
- Bayesian estimation
- ensemble forecasting

### Machine Learning

- gradient boosting
- random forests
- temporal models
- regime classification
- model ensembles
- probability-calibrated forecasts

### Execution

- spread-aware costs
- liquidity constraints
- market-impact models
- order sizing
- execution simulation

### Portfolio Management

- dynamic risk budgets
- volatility targeting
- cash allocation
- multi-asset portfolios
- live paper trading

---

## 45. Project Roadmap

The project has been developed in phases:

```text
Phase 1  — Repository and Environment
Phase 2  — Market Data Pipeline
Phase 3  — Features and Risk Inputs
Phase 4  — Portfolio Optimization
Phase 5  — Backtesting Engine
Phase 6  — Benchmarks and Evaluation
Phase 7  — Robustness Analysis
Phase 8  — Machine Learning Experiment
Phase 9  — Integration and Productionization
```

The final integration phase focuses on making the system reproducible,
inspectable, testable, and presentable rather than continuously adding
strategies without validating the existing pipeline.

---

## 46. Final Takeaways

The current QuantPilot research demonstrates several important points.

### 1. Portfolio construction matters

The portfolio optimizer produces a substantially different risk/return
profile from both Buy & Hold and Equal Weight.

### 2. Higher return is not automatically better

Buy & Hold produced the highest CAGR but also the highest volatility and
largest drawdown.

### 3. Simple benchmarks are difficult to beat

Equal Weight performed extremely well and exceeded QuantPilot on CAGR and
Sharpe in the canonical experiment.

### 4. Risk control has a measurable effect

QuantPilot achieved substantially lower volatility and drawdown than
Buy & Hold.

### 5. Trading friction matters

Adding transaction costs and slippage reduced both CAGR and Sharpe.

### 6. Robustness matters more than one backtest

Parameter sensitivity and regime analysis are necessary before drawing
strong conclusions.

### 7. Machine learning is not automatically superior

The experimental ML model produced a slightly higher CAGR but worse
risk-adjusted performance and materially larger drawdown.

### 8. Backtesting discipline matters

Walk-forward estimation, leakage prevention, portfolio drift, transaction
costs, slippage, and realistic rebalancing are central components of the
system.

---

## 47. Disclaimer

QuantPilot is an educational and research project.

Historical backtest results do not guarantee future performance.

Nothing in this repository constitutes financial, investment, or trading
advice.

The strategy has not been validated for live capital deployment and should
not be assumed to be profitable in real-world trading.

---

## 48. Author

### Neil Sulhyan

**Walchand College of Engineering - (CSE)**  

QuantPilot was developed as a quantitative finance and software-engineering research project.

The project combines concepts from:

- quantitative finance
- portfolio theory
- statistical modeling
- optimization
- machine learning
- software engineering
- data engineering

The objective is to demonstrate an end-to-end understanding of how a quantitative portfolio system can be designed, implemented, tested, and evaluated.

### Connect

- **GitHub:** https://github.com/NeilSulhyan10
- **LinkedIn:** https://www.linkedin.com/in/neil-sulhyan-091ba82aa/
- **Instagram:** https://www.instagram.com/neilsulhyan10/
- **Portfolio:** https://neilsulhyan10.github.io/Portfolio/

---

## License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.
