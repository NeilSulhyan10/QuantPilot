import numpy as np
import pandas as pd

from src.evaluation.portfolio_diagnostics import (
    active_positions,
    average_weights,
    cap_hit_frequency,
    clean_weights,
    effective_number_of_stocks,
    maximum_weights,
    turnover_statistics,
)


def sample_weights():
    return pd.DataFrame(
        {
            "A": [0.50, 0.40],
            "B": [0.50, 0.40],
            "C": [0.0, 0.20],
        },
        index=pd.date_range(
            "2020-01-01",
            periods=2,
        ),
    )


def test_clean_weights_removes_solver_noise():
    weights = pd.DataFrame(
        {
            "A": [1e-12],
            "B": [-1e-12],
            "C": [0.5],
        }
    )

    cleaned = clean_weights(weights)

    assert cleaned.loc[0, "A"] == 0.0
    assert cleaned.loc[0, "B"] == 0.0
    assert cleaned.loc[0, "C"] == 0.5


def test_average_weights():
    result = average_weights(sample_weights())

    assert np.isclose(result["A"], 0.45)
    assert np.isclose(result["B"], 0.45)
    assert np.isclose(result["C"], 0.10)


def test_maximum_weights():
    result = maximum_weights(sample_weights())

    assert np.isclose(result["A"], 0.50)
    assert np.isclose(result["B"], 0.50)
    assert np.isclose(result["C"], 0.20)


def test_active_positions():
    result = active_positions(sample_weights())

    assert result.tolist() == [2, 3]


def test_effective_number_of_stocks():
    result = effective_number_of_stocks(sample_weights())

    assert np.isclose(result.iloc[0], 2.0)


def test_cap_hit_frequency():
    weights = pd.DataFrame(
        {
            "A": [0.10, 0.10, 0.05],
            "B": [0.05, 0.10, 0.10],
        }
    )

    result = cap_hit_frequency(weights, cap=0.10)

    assert np.isclose(result["A"], 2 / 3)
    assert np.isclose(result["B"], 2 / 3)


def test_turnover_statistics():
    turnover = pd.Series([0.10, 0.20, 0.30])

    result = turnover_statistics(turnover)

    assert np.isclose(result["mean"], 0.20)
    assert np.isclose(result["median"], 0.20)
    assert np.isclose(result["minimum"], 0.10)
    assert np.isclose(result["maximum"], 0.30)
