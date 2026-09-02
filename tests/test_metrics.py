import pandas as pd
import pytest

from src.evaluation.metrics import (
    cumulative_return,
    cagr,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    maximum_drawdown,
    evaluate_returns
)


def test_cumulative_return():
    returns = pd.Series(
        [0.10, 0.10]
    )

    result = cumulative_return(returns)

    assert result == pytest.approx(0.21)


def test_cumulative_return_empty():
    with pytest.raises(ValueError):
        cumulative_return(pd.Series(dtype=float))


def test_cumulative_return_missing_values():
    returns = pd.Series([0.01, None, 0.02])

    with pytest.raises(ValueError):
        cumulative_return(returns)

def test_cagr():
    dates = pd.to_datetime(
        [
            "2020-01-01",
            "2021-01-01",
        ]
    )

    returns = pd.Series(
        [0.10, 0.10],
        index=dates,
    )

    result = cagr(returns)

    expected = (1.21 ** (1 / (366 / 365.25))) - 1

    assert result == pytest.approx(expected)


def test_cagr_requires_multiple_observations():
    returns = pd.Series(
        [0.10],
        index=pd.to_datetime(["2020-01-01"]),
    )

    with pytest.raises(ValueError):
        cagr(returns)


def test_cagr_requires_positive_time_span():
    dates = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-01",
        ]
    )

    returns = pd.Series(
        [0.10, 0.10],
        index=dates,
    )

    with pytest.raises(ValueError):
        cagr(returns)

def test_annualized_volatility():
    returns = pd.Series(
        [0.01, -0.01, 0.01, -0.01]
    )

    result = annualized_volatility(returns)

    expected = returns.std() * (252 ** 0.5)

    assert result == pytest.approx(expected)


def test_annualized_volatility_empty():
    with pytest.raises(ValueError):
        annualized_volatility(pd.Series(dtype=float))


def test_annualized_volatility_invalid_periods():
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError):
        annualized_volatility(
            returns,
            periods_per_year=0,
        )

def test_sharpe_ratio():
    returns = pd.Series(
        [0.01, 0.02, 0.00, 0.01]
    )

    result = sharpe_ratio(returns)

    expected = (
        returns.mean()
        / returns.std()
        * (252 ** 0.5)
    )

    assert result == pytest.approx(expected)


def test_sharpe_ratio_with_risk_free_rate():
    returns = pd.Series(
        [0.01, 0.02, 0.00, 0.01]
    )

    result = sharpe_ratio(
        returns,
        risk_free_rate=0.04,
    )

    assert isinstance(result, float)


def test_sharpe_ratio_zero_volatility():
    returns = pd.Series(
        [0.01, 0.01, 0.01]
    )

    with pytest.raises(ValueError):
        sharpe_ratio(returns)


def test_sharpe_ratio_invalid_periods():
    returns = pd.Series(
        [0.01, 0.02]
    )

    with pytest.raises(ValueError):
        sharpe_ratio(
            returns,
            periods_per_year=0,
        )

def test_sortino_ratio():
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.02]
    )

    result = sortino_ratio(returns)

    downside = returns[returns < 0]
    downside_deviation = (
        (downside ** 2).mean() ** 0.5
    )

    expected = (
        returns.mean()
        / downside_deviation
        * (252 ** 0.5)
    )

    assert result == pytest.approx(expected)


def test_sortino_ratio_with_risk_free_rate():
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.02]
    )

    result = sortino_ratio(
        returns,
        risk_free_rate=0.04,
    )

    assert isinstance(result, float)


def test_sortino_ratio_no_downside_returns():
    returns = pd.Series(
        [0.01, 0.02, 0.03]
    )

    with pytest.raises(ValueError):
        sortino_ratio(returns)


def test_sortino_ratio_invalid_periods():
    returns = pd.Series(
        [0.01, -0.01]
    )

    with pytest.raises(ValueError):
        sortino_ratio(
            returns,
            periods_per_year=0,
        )

def test_maximum_drawdown():
    returns = pd.Series(
        [0.10, 0.20, -0.10, -0.20, 0.10]
    )

    result = maximum_drawdown(returns)

    wealth = (1 + returns).cumprod()
    running_peak = wealth.cummax()
    expected = (wealth / running_peak - 1).min()

    assert result == pytest.approx(expected)


def test_maximum_drawdown_never_negative():
    returns = pd.Series(
        [0.01, 0.02, 0.03]
    )

    result = maximum_drawdown(returns)

    assert result == pytest.approx(0.0)


def test_maximum_drawdown_empty():
    with pytest.raises(ValueError):
        maximum_drawdown(pd.Series(dtype=float))


def test_maximum_drawdown_missing_values():
    returns = pd.Series(
        [0.01, None, -0.02]
    )

    with pytest.raises(ValueError):
        maximum_drawdown(returns)

def test_evaluate_returns():
    returns = pd.Series(
        [0.02, -0.01, 0.03, -0.02],
        index=pd.date_range(
            "2020-01-01",
            periods=4,
            freq="B",
        ),
    )

    result = evaluate_returns(returns)

    expected_keys = {
        "cumulative_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
    }

    assert set(result.keys()) == expected_keys

    for value in result.values():
        assert isinstance(value, float)


def test_evaluate_returns_empty():
    with pytest.raises(ValueError):
        evaluate_returns(pd.Series(dtype=float))