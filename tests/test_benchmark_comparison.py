import pandas as pd
import pytest

from src.evaluation.benchmark_comparison import (
    evaluate_strategy_comparison,
)


def test_strategy_comparison_aligns_to_common_dates():
    index_a = pd.date_range("2020-01-01", periods=5, freq="D")
    index_b = pd.date_range("2020-01-03", periods=5, freq="D")

    returns_a = pd.Series(
        [0.01, -0.005, 0.012, 0.003, -0.002],
        index=index_a,
    )

    returns_b = pd.Series(
        [0.02, -0.01, 0.015, 0.005, -0.003],
        index=index_b,
    )

    result = evaluate_strategy_comparison(
        {
            "A": returns_a,
            "B": returns_b,
        }
    )

    assert list(result.index) == ["A", "B"]
    assert result["cumulative_return"].notna().all()
    assert result["cagr"].notna().all()
    assert result["sharpe_ratio"].notna().all()

    # Common dates are 2020-01-03 through 2020-01-05.
    assert result.loc["A", "cumulative_return"] == pytest.approx(
        (1.012 * 1.003 * 0.998) - 1
    )

    assert result.loc["B", "cumulative_return"] == pytest.approx(
        (1.02 * 0.99 * 1.015) - 1
    )


def test_empty_strategy_raises():
    returns = pd.Series(
        [0.01, 0.02],
        index=pd.date_range("2020-01-01", periods=2),
    )

    with pytest.raises(ValueError):
        evaluate_strategy_comparison(
            {
                "A": returns,
                "B": pd.Series(dtype=float),
            }
        )


def test_no_common_dates_raises():
    returns_a = pd.Series(
        [0.01, 0.02],
        index=pd.date_range("2020-01-01", periods=2),
    )

    returns_b = pd.Series(
        [0.01, 0.02],
        index=pd.date_range("2021-01-01", periods=2),
    )

    with pytest.raises(ValueError):
        evaluate_strategy_comparison(
            {
                "A": returns_a,
                "B": returns_b,
            }
        )
