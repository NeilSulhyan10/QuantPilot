from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.goals.assets import build_selected_return_matrix
from src.goals.optimizer import optimize_for_goal, optimize_for_goal_with_cash
from src.goals.planner import (
    _evaluate_subset_for_risk_profile,
    _sort_subset_results,
    _sharpe_or_none,
    build_allocation_weights,
    evaluate_subsets_with_investment,
    RiskProfileResult,
    SubsetInvestmentResult,
)


def test_ranking_method_sorts_low_risk_and_highest_future_value():
    def subset(ticker, volatility, expected_return, future_value):
        profile = SimpleNamespace(
            feasible=True,
            expected_volatility=volatility,
            expected_return=expected_return,
        )
        return SubsetInvestmentResult(
            tickers=(ticker,),
            subset_size=1,
            max_weight=1.0,
            feasible=True,
            conservative=profile,
            best_future_value=future_value,
        )

    low_risk = subset("LOW_RISK", volatility=0.08, expected_return=0.12, future_value=1200)
    high_value = subset("HIGH_VALUE", volatility=0.16, expected_return=0.18, future_value=1800)

    results = [high_value, low_risk]
    _sort_subset_results(results, "low_risk")
    assert [result.tickers[0] for result in results] == ["LOW_RISK", "HIGH_VALUE"]

    _sort_subset_results(results, "max_future_value")
    assert [result.tickers[0] for result in results] == ["HIGH_VALUE", "LOW_RISK"]


def test_valid_multi_stock_allocation_includes_cash_row():
    weights = pd.Series({"AAA": 0.4, "BBB": 0.35, "CCC": 0.25})

    allocation = build_allocation_weights(
        ["AAA", "BBB", "CCC"], weights, cash_weight=0.0
    )

    assert allocation.to_dict() == {
        "AAA": 0.4,
        "BBB": 0.35,
        "CCC": 0.25,
        "Cash": 0.0,
    }
    assert allocation.sum() == pytest.approx(1.0)


def test_valid_zero_weight_stock_is_preserved_and_mapped_by_ticker():
    tickers = ["AAA", "BBB", "CCC"]
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.05, "CCC": 0.02})
    covariance = pd.DataFrame(
        np.diag([0.1, 0.0, 0.2]), index=tickers, columns=tickers
    )
    result = optimize_for_goal(
        expected_returns,
        covariance,
        target_return=0.01,
        risk_tolerance="aggressive",
        max_weight=1.0,
    )

    assert result.weights.index.tolist() == tickers
    assert result.weights["AAA"] == pytest.approx(0.0, abs=1e-5)
    allocation = build_allocation_weights(tickers, result.weights, 0.0)
    assert allocation.index.tolist() == [*tickers, "Cash"]
    assert allocation["AAA"] == pytest.approx(0.0, abs=1e-5)
    assert allocation.sum() == pytest.approx(1.0)


def test_infeasible_optimization_is_reported():
    tickers = ["AAA", "BBB"]
    mu = pd.Series([0.02, 0.03], index=tickers)
    covariance = pd.DataFrame(np.eye(2) * 0.001, index=tickers, columns=tickers)
    with pytest.raises(ValueError, match="infeasible"):
        optimize_for_goal(mu, covariance, 0.50, "moderate", max_weight=1.0)


@pytest.mark.parametrize("volatility", [0.0, 1e-10, 1e-8, 5e-5])
def test_zero_or_near_zero_volatility_has_no_sharpe(volatility):
    assert _sharpe_or_none(0.10, volatility) is None


def test_meaningful_volatility_has_finite_sharpe():
    assert _sharpe_or_none(0.10, 0.20) == pytest.approx(0.5)


def test_missing_price_data_is_rejected():
    prices = pd.DataFrame({"Close": [100.0, np.nan, 102.0]})
    with pytest.raises(ValueError, match="missing"):
        build_selected_return_matrix({"AAA": prices})


@pytest.mark.parametrize("bad_return", [np.nan, np.inf, -np.inf])
def test_nan_or_infinite_expected_returns_are_rejected(bad_return):
    mu = pd.Series({"AAA": 0.05, "BBB": bad_return})
    covariance = pd.DataFrame(
        np.eye(2) * 0.001, index=mu.index, columns=mu.index
    )
    with pytest.raises(ValueError, match="NaN|non-finite"):
        optimize_for_goal(mu, covariance, 0.01, "moderate", max_weight=1.0)


@pytest.mark.parametrize(
    "covariance, error",
    [
        (np.array([[1.0, 0.5], [0.1, 1.0]]), "symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "positive semidefinite"),
        (np.array([[1.0, np.inf], [np.inf, 1.0]]), "non-finite"),
    ],
)
def test_invalid_covariance_is_rejected(covariance, error):
    tickers = ["AAA", "BBB"]
    sigma = pd.DataFrame(covariance, index=tickers, columns=tickers)
    mu = pd.Series([0.05, 0.06], index=tickers)
    with pytest.raises(ValueError, match=error):
        optimize_for_goal(mu, sigma, 0.01, "moderate", max_weight=1.0)


def test_stock_cash_weight_sum_is_checked():
    with pytest.raises(ValueError, match="sum to 1"):
        build_allocation_weights(["AAA", "BBB"], pd.Series({"AAA": 0.5, "BBB": 0.3}), 0.1)


def test_weight_mapping_rejects_missing_or_unexpected_tickers():
    with pytest.raises(ValueError, match="do not match"):
        build_allocation_weights(["AAA", "BBB"], pd.Series({"AAA": 0.5, "CCC": 0.5}))


def test_cash_optimizer_failure_metrics_are_unavailable(monkeypatch):
    import src.goals.planner as planner

    dates = pd.bdate_range("2022-01-01", periods=260)
    returns = np.tile([0.0001, 0.0002, 0.0003], (len(dates), 1))
    assets = {
        ticker: pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + returns[:, i])}, index=dates
        )
        for i, ticker in enumerate(["AAA", "BBB", "CCC"])
    }
    monkeypatch.setattr(
        planner,
        "calculate_maximum_feasible_return_with_cash",
        lambda **kwargs: 0.10,
    )
    monkeypatch.setattr(
        "src.goals.optimizer.optimize_for_goal_with_cash",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("infeasible")),
    )

    result = _evaluate_subset_for_risk_profile(
        asset_data=assets,
        years=5,
        risk_tolerance="moderate",
        max_weight=1 / 3,
        minimum_observations=252,
        estimation_window=60,
        shrinkage_alpha=0.5,
        lump_sum=1000,
        monthly_sip=0,
        use_cash=True,
    )

    assert result.feasible is False
    assert result.expected_return is None
    assert result.expected_volatility is None
    assert result.sharpe_ratio is None
    assert result.future_value_lump_sum is None


def test_stock_only_profile_optimizes_when_cash_is_disabled():
    dates = pd.bdate_range("2022-01-01", periods=300)
    t = np.arange(len(dates))
    daily_returns = np.column_stack(
        [
            0.0012 + 0.004 * np.sin(t * 0.17),
            0.0009 + 0.003 * np.sin(t * 0.11 + 1),
            0.0007 + 0.0035 * np.sin(t * 0.07 + 2),
        ]
    )
    assets = {
        ticker: pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + daily_returns[:, i])}, index=dates
        )
        for i, ticker in enumerate(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    }

    result = _evaluate_subset_for_risk_profile(
        asset_data=assets,
        years=5,
        risk_tolerance="moderate",
        max_weight=1 / 3,
        minimum_observations=252,
        estimation_window=60,
        shrinkage_alpha=0.5,
        lump_sum=1000,
        monthly_sip=0,
        use_cash=False,
    )

    assert result.feasible is True
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert result.cash_weight == 0.0
    assert result.expected_return >= result.recommended_return - 1e-5


def test_subset_is_feasible_when_any_risk_profile_is_feasible(monkeypatch):
    import src.goals.planner as planner

    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    dates = pd.bdate_range("2022-01-01", periods=260)
    assets = {
        ticker: pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + 0.001 * np.sin(np.arange(260) / 8 + i))},
            index=dates,
        )
        for i, ticker in enumerate(tickers)
    }
    monkeypatch.setattr(planner, "load_selected_assets", lambda **kwargs: assets)

    def profile_result(*, risk_tolerance, **kwargs):
        feasible = risk_tolerance != "conservative"
        return RiskProfileResult(
            risk_profile=risk_tolerance,
            max_allowed_volatility=0.10,
            max_feasible_return=0.20,
            recommended_return=0.10,
            expected_return=0.10,
            expected_volatility=0.05,
            sharpe_ratio=2.0,
            future_value_lump_sum=1000.0 if feasible else None,
            future_value_monthly=0.0 if feasible else None,
            feasible=feasible,
            warning="Conservative risk limit not met" if not feasible else None,
            weights=pd.Series(1 / 3, index=tickers),
            cash_weight=0.0,
        )

    monkeypatch.setattr(planner, "_evaluate_subset_for_risk_profile", profile_result)
    results = evaluate_subsets_with_investment(
        all_tickers=tickers,
        market="india",
        lump_sum=1000,
        monthly_sip=0,
        years=5,
        minimum_observations=252,
        estimation_window=60,
        allow_download=False,
    )

    assert len(results) == 1
    assert results[0].feasible is True
    assert results[0].conservative.feasible is False
    assert results[0].moderate.feasible is True
    assert results[0].aggressive.feasible is True


def test_cash_and_stock_allocations_remain_consistent():
    allocation = build_allocation_weights(
        ["AAA", "BBB", "CCC"],
        pd.Series({"CCC": 0.0, "AAA": 0.3, "BBB": 0.2}),
        cash_weight=0.5,
    )
    assert allocation.to_dict() == {
        "AAA": 0.3,
        "BBB": 0.2,
        "CCC": 0.0,
        "Cash": 0.5,
    }
    assert allocation.sum() == pytest.approx(1.0)


def test_cash_optimizer_returns_consistent_stock_and_cash_weights():
    tickers = ["AAA", "BBB", "CCC"]
    expected_returns = pd.Series([0.01, 0.02, 0.03], index=tickers)
    covariance = pd.DataFrame(
        np.eye(3) * 0.01, index=tickers, columns=tickers
    )
    result = optimize_for_goal_with_cash(
        expected_returns,
        covariance,
        target_return=0.045,
        risk_tolerance="moderate",
        max_weight=1 / 3,
        risk_free_rate=0.045,
    )

    assert result.weights.index.tolist() == tickers
    assert result.cash_weight == pytest.approx(1.0, abs=1e-5)
    allocation = build_allocation_weights(tickers, result.weights, result.cash_weight)
    assert allocation.sum() == pytest.approx(1.0)
    assert list(allocation.index) == [*tickers, "Cash"]


def test_cash_meeting_target_returns_exact_all_cash_portfolio():
    tickers = ["SBIN.NS", "HINDUNILVR.NS", "ITC.NS"]
    expected_returns = pd.Series([0.08, 0.07, 0.06], index=tickers)
    covariance = pd.DataFrame(
        np.diag([0.0004, 0.0003, 0.00025]), index=tickers, columns=tickers
    )

    result = optimize_for_goal_with_cash(
        expected_returns,
        covariance,
        target_return=0.0367,
        risk_tolerance="conservative",
        max_weight=1 / 3,
        risk_free_rate=0.045,
    )

    assert result.weights.to_dict() == {ticker: 0.0 for ticker in tickers}
    assert result.cash_weight == 1.0
    assert result.expected_return == pytest.approx(0.045)
    assert result.expected_volatility == 0.0
    assert _sharpe_or_none(result.expected_return, result.expected_volatility) is None
