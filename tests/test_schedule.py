import pandas as pd

from src.portfolio.schedule import monthly_rebalance_dates


def test_monthly_rebalance_dates():
    dates = pd.DatetimeIndex([
        "2024-01-02",
        "2024-01-15",
        "2024-01-31",
        "2024-02-01",
        "2024-02-28",
        "2024-03-01",
        "2024-03-29",
    ])

    result = monthly_rebalance_dates(dates)

    expected = pd.DatetimeIndex([
        "2024-01-31",
        "2024-02-28",
        "2024-03-29",
    ])

    assert result.equals(expected)


def test_unsorted_dates_are_handled():
    dates = pd.DatetimeIndex([
        "2024-02-28",
        "2024-01-31",
        "2024-01-02",
        "2024-03-29",
    ])

    result = monthly_rebalance_dates(dates)

    expected = pd.DatetimeIndex([
        "2024-01-31",
        "2024-02-28",
        "2024-03-29",
    ])

    assert result.equals(expected)


def test_empty_dates_raise_error():
    dates = pd.DatetimeIndex([])

    try:
        monthly_rebalance_dates(dates)
        assert False
    except ValueError:
        pass