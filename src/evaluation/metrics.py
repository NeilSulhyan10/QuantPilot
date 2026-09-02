import pandas as pd
import numpy as np


def cumulative_return(returns: pd.Series) -> float:
    if returns.empty:
        raise ValueError("Returns series is empty.")

    if returns.isna().any():
        raise ValueError("Returns series contains missing values.")

    return float((1 + returns).prod() - 1)


def cagr(returns: pd.Series) -> float:
    if returns.empty:
        raise ValueError("Returns series is empty.")

    if returns.isna().any():
        raise ValueError("Returns series contains missing values.")

    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")

    start_date = returns.index[0]
    end_date = returns.index[-1]

    years = (end_date - start_date).days / 365.25

    if years <= 0:
        raise ValueError("Return series must span a positive amount of time.")

    growth = (1 + returns).prod()

    return float(growth ** (1 / years) - 1)

def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    if returns.empty:
        raise ValueError("Returns series is empty.")

    if returns.isna().any():
        raise ValueError("Returns series contains missing values.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    return float(returns.std() * np.sqrt(periods_per_year))

def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if returns.empty:
        raise ValueError("Returns series is empty.")

    if returns.isna().any():
        raise ValueError("Returns series contains missing values.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    daily_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / periods_per_year)
    ) - 1

    excess_returns = returns - daily_risk_free_rate

    volatility = excess_returns.std()

    if volatility == 0:
        raise ValueError(
            "Sharpe ratio is undefined when return volatility is zero."
        )

    return float(
        excess_returns.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )

def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    if returns.empty:
        raise ValueError("Returns series is empty.")

    if returns.isna().any():
        raise ValueError("Returns series contains missing values.")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    daily_risk_free_rate = (
        (1 + risk_free_rate) ** (1 / periods_per_year)
    ) - 1

    excess_returns = returns - daily_risk_free_rate

    downside_returns = excess_returns[excess_returns < 0]

    if downside_returns.empty:
        raise ValueError(
            "Sortino ratio is undefined when there are no downside returns."
        )

    downside_deviation = (
        (downside_returns ** 2).mean() ** 0.5
    )

    if downside_deviation == 0:
        raise ValueError(
            "Sortino ratio is undefined when downside deviation is zero."
        )

    return float(
        excess_returns.mean()
        / downside_deviation
        * np.sqrt(periods_per_year)
    )

def maximum_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        raise ValueError("Returns series is empty.")

    if returns.isna().any():
        raise ValueError("Returns series contains missing values.")

    wealth = (1 + returns).cumprod()

    running_peak = wealth.cummax()

    drawdown = wealth / running_peak - 1

    return float(drawdown.min())

def evaluate_returns(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    if returns.empty:
        raise ValueError("Returns series is empty.")

    return {
        "cumulative_return": cumulative_return(returns),
        "cagr": cagr(returns),
        "annualized_volatility": annualized_volatility(
            returns,
            periods_per_year=periods_per_year,
        ),
        "sharpe_ratio": sharpe_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "sortino_ratio": sortino_ratio(
            returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        ),
        "maximum_drawdown": maximum_drawdown(returns),
    }