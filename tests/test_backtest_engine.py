import pandas as pd
import pytest

from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.config import BacktestConfig
from src.backtesting.engine import WalkForwardBacktester
from src.backtesting.results import BacktestResult
from src.backtesting.returns import apply_rebalance_costs
from tests.test_benchmark_engine import make_returns


def test_get_rebalance_dates_uses_available_feature_dates():
    dates = pd.date_range("2020-01-01", "2020-04-30", freq="B")

    returns = pd.DataFrame(
        {
            "AAPL": 0.01,
            "MSFT": 0.02,
        },
        index=dates,
    )

    expected_returns = returns.copy()

    covariance_matrices = {
        pd.Timestamp("2020-01-31"): pd.DataFrame(
            [[0.01, 0.0], [0.0, 0.01]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        ),
        pd.Timestamp("2020-02-28"): pd.DataFrame(
            [[0.01, 0.0], [0.0, 0.01]],
            index=["AAPL", "MSFT"],
            columns=["AAPL", "MSFT"],
        ),
    }

    backtester = WalkForwardBacktester(BacktestConfig())

    rebalance_dates = backtester._get_rebalance_dates(
        returns,
        expected_returns,
        covariance_matrices,
    )

    assert list(rebalance_dates) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-28"),
    ]

def test_calculate_target_weights():
    date = pd.Timestamp("2020-01-31")

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "GOOGL",
        "AMZN",
        "META",
        "JPM",
        "V",
        "MA",
    ]

    expected_returns = pd.DataFrame(
        [[0.001] * len(tickers)],
        index=[date],
        columns=tickers,
    )

    covariance = pd.DataFrame(
        0.002,
        index=tickers,
        columns=tickers,
    )

    for ticker in tickers:
        covariance.loc[ticker, ticker] = 0.01

    covariance_matrices = {
        date: covariance,
    }

    backtester = WalkForwardBacktester(BacktestConfig())

    weights = backtester._calculate_target_weights(
        date=date,
        expected_returns=expected_returns,
        covariance_matrices=covariance_matrices,
    )

    assert isinstance(weights, pd.Series)
    assert list(weights.index) == tickers
    assert abs(weights.sum() - 1.0) < 1e-8
    assert (weights >= -1e-8).all()
    assert (weights <= 0.10 + 1e-8).all()

def test_daily_returns_start_after_rebalance_date():
    dates = pd.date_range(
        "2020-01-30",
        "2020-02-04",
        freq="B",
    )

    returns = pd.DataFrame(
        {
            "AAPL": [0.01, 0.02, 0.03, 0.04],
            "MSFT": [0.02, 0.03, 0.04, 0.05],
        },
        index=dates,
    )

    rebalance_dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-30"),
            pd.Timestamp("2020-02-03"),
        ]
    )

    target_weights = {
        pd.Timestamp("2020-01-30"): pd.Series(
            {"AAPL": 0.5, "MSFT": 0.5}
        ),
        pd.Timestamp("2020-02-03"): pd.Series(
            {"AAPL": 0.2, "MSFT": 0.8}
        ),
    }

    backtester = WalkForwardBacktester(BacktestConfig())

    portfolio_returns = backtester._calculate_daily_returns(
        returns=returns,
        rebalance_dates=rebalance_dates,
        target_weights=target_weights,
    )

    # January 30 is the rebalance day, so it must NOT have a return.
    assert pd.Timestamp("2020-01-30") not in portfolio_returns.index

    # January 31 uses the January 30 target weights.
    assert portfolio_returns.loc["2020-01-31"] == 0.025

    # February 3 is the second rebalance day, so it is excluded.
    assert pd.Timestamp("2020-02-03") not in portfolio_returns.index

    # February 4 uses the February 3 target weights.
    assert portfolio_returns.loc["2020-02-04"] == pytest.approx(0.048)

def test_rebalance_costs_accounts_for_weight_drift():
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2020-02-28"),
        ]
    )

    returns_dates = pd.date_range(
        "2020-02-03",
        "2020-02-27",
        freq="B",
    )

    returns = pd.DataFrame(
        {
            "AAPL": 0.10,
            "MSFT": 0.00,
        },
        index=returns_dates,
    )

    target_weights = {
        dates[0]: pd.Series(
            {
                "AAPL": 0.5,
                "MSFT": 0.5,
            }
        ),
        dates[1]: pd.Series(
            {
                "AAPL": 0.6,
                "MSFT": 0.4,
            }
        ),
    }

    backtester = WalkForwardBacktester(
        BacktestConfig()
    )

    turnover, transaction_costs, slippage = (
        backtester._calculate_rebalance_costs(
            returns=returns,
            rebalance_dates=dates,
            target_weights=target_weights,
        )
    )

    # Initial deployment = 100% traded.
    assert turnover.loc[dates[0]] == pytest.approx(1.0)

    # AAPL rises repeatedly between rebalances, so its
    # portfolio weight drifts above 50%.
    #
    # Therefore the actual turnover required to reach
    # the 60/40 target differs from simply comparing
    # 50/50 -> 60/40.
    assert turnover.loc[dates[1]] > 0.0

    assert transaction_costs.loc[dates[1]] == pytest.approx(
        turnover.loc[dates[1]] * 0.001
    )

    assert slippage.loc[dates[1]] == pytest.approx(
        turnover.loc[dates[1]] * 0.0005
    )

def test_run_returns_backtest_result():
    dates = pd.date_range(
        "2020-01-01",
        "2020-04-30",
        freq="B",
    )

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AVGO",
        "GOOGL",
        "AMZN",
        "META",
        "JPM",
        "V",
        "MA",
    ]

    returns = pd.DataFrame(
        0.001,
        index=dates,
        columns=tickers,
    )

    expected_returns = returns.groupby(
        returns.index.to_period("M")
    ).mean()

    covariance = pd.DataFrame(
        0.002,
        index=tickers,
        columns=tickers,
    )

    for ticker in tickers:
        covariance.loc[ticker, ticker] = 0.01

    covariance_matrices = {
        pd.Timestamp("2020-01-31"): covariance,
        pd.Timestamp("2020-02-28"): covariance,
        pd.Timestamp("2020-03-31"): covariance,
        pd.Timestamp("2020-04-30"): covariance,
    }

    # The expected-return index must contain the actual
    # rebalance dates.
    expected_returns = pd.DataFrame(
        0.001,
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2020-01-31"),
                pd.Timestamp("2020-02-28"),
                pd.Timestamp("2020-03-31"),
                pd.Timestamp("2020-04-30"),
            ]
        ),
        columns=tickers,
    )

    backtester = WalkForwardBacktester(BacktestConfig())

    result = backtester.run(
        returns=returns,
        expected_returns=expected_returns,
        covariance_matrices=covariance_matrices,
    )

    assert isinstance(result, BacktestResult)

    assert not result.returns.empty
    assert not result.weights.empty
    assert not result.turnover.empty
    assert not result.transaction_costs.empty
    assert not result.slippage.empty

    assert len(result.weights) == 4
    assert result.weights.shape[1] == 10

    # Every portfolio is fully invested.
    assert (result.weights.sum(axis=1) - 1.0).abs().max() < 1e-8

    # Initial deployment.
    assert result.turnover.iloc[0] == pytest.approx(1.0)

    # Initial trading costs reduce the first realized return.
    assert result.returns.iloc[0] == pytest.approx(
        0.001 - 0.001 - 0.0005
    )

def test_apply_rebalance_costs():
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-02-03"),
            pd.Timestamp("2020-02-04"),
            pd.Timestamp("2020-02-05"),
        ]
    )

    gross_returns = pd.Series(
        [0.01, 0.02, 0.03],
        index=dates,
    )

    rebalance_dates = pd.DatetimeIndex(
        [pd.Timestamp("2020-01-31")]
    )

    transaction_costs = pd.Series(
        [0.001],
        index=rebalance_dates,
    )

    slippage = pd.Series(
        [0.0005],
        index=rebalance_dates,
    )

    net_returns = apply_rebalance_costs(
        gross_returns=gross_returns,
        rebalance_dates=rebalance_dates,
        transaction_costs=transaction_costs,
        slippage=slippage,
    )

    # Costs from Jan 31 are charged against the first
    # realized portfolio return on Feb 3.
    assert net_returns.loc["2020-02-03"] == pytest.approx(0.0085)

    # Later returns remain unchanged.
    assert net_returns.loc["2020-02-04"] == pytest.approx(0.02)
    assert net_returns.loc["2020-02-05"] == pytest.approx(0.03)

def test_apply_rebalance_costs_multiple_rebalances():
    dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-02-03"),
            pd.Timestamp("2020-02-04"),
            pd.Timestamp("2020-03-02"),
            pd.Timestamp("2020-03-03"),
        ]
    )

    gross_returns = pd.Series(
        [0.01, 0.02, 0.03, 0.04],
        index=dates,
    )

    rebalance_dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-01-31"),
            pd.Timestamp("2020-02-28"),
        ]
    )

    transaction_costs = pd.Series(
        [0.001, 0.0002],
        index=rebalance_dates,
    )

    slippage = pd.Series(
        [0.0005, 0.0001],
        index=rebalance_dates,
    )

    net_returns = apply_rebalance_costs(
        gross_returns=gross_returns,
        rebalance_dates=rebalance_dates,
        transaction_costs=transaction_costs,
        slippage=slippage,
    )

    # Jan 31 costs are applied to first subsequent
    # realized return.
    assert net_returns.loc["2020-02-03"] == pytest.approx(0.0085)

    # Feb 28 costs are applied to first subsequent
    # realized return.
    assert net_returns.loc["2020-03-02"] == pytest.approx(0.0297)

    # Other days remain unchanged.
    assert net_returns.loc["2020-02-04"] == pytest.approx(0.02)
    assert net_returns.loc["2020-03-03"] == pytest.approx(0.04)

def test_equal_weight_can_use_custom_rebalance_dates():
    returns = make_returns()

    rebalance_dates = pd.DatetimeIndex(
        [
            pd.Timestamp("2020-02-28"),
            pd.Timestamp("2020-04-30"),
            pd.Timestamp("2020-06-30"),
        ]
    )

    result = EqualWeightBacktester(
        BacktestConfig()
    ).run(
        returns=returns,
        rebalance_dates=rebalance_dates,
    )

    assert list(result.weights.index) == list(rebalance_dates)