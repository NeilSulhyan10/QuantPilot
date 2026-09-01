# QuantPilot

Adaptive Quantitative Portfolio Manager.

## Objective

QuantPilot is a quantitative portfolio management system for US individual equities.

The system combines:

- Statistical return estimation
- Risk and covariance modeling
- Mathematical portfolio optimization
- Machine learning as a supporting component
- Portfolio constraints
- Periodic rebalancing
- Transaction costs and slippage
- Walk-forward out-of-sample backtesting
- Performance evaluation
- Paper portfolio management

## MVP

- ~20 liquid US equities
- Daily OHLCV data
- Rolling expected returns
- Rolling covariance estimation
- Mean-variance optimization
- Long-only portfolio
- Full-investment constraint
- Position limits
- Weekly/monthly rebalancing
- Transaction costs
- Slippage
- Walk-forward backtesting
- Benchmark comparison
- One interpretable ML component
- Performance dashboard

## Project Structure

```text
quantpilot/
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── risk/
│   ├── models/
│   ├── optimization/
│   ├── backtesting/
│   ├── evaluation/
│   └── portfolio/
├── tests/
├── configs/
├── requirements.txt
└── README.md
