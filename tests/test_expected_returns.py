import numpy as np
import pandas as pd
import pytest

from src.models.expected_returns import (
    historical_mean_expected_returns,
    shrink_expected_returns,
    calculate_expected_returns,
)


def test_historical_mean():

    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03],
            "B": [0.03, 0.02, 0.01],
        }
    )

    result = historical_mean_expected_returns(
        returns,
        window=2,
    )

    assert np.isnan(result.iloc[0]["A"])

    assert result.iloc[1]["A"] == pytest.approx(0.015)

    assert result.iloc[2]["A"] == pytest.approx(0.025)


def test_shrinkage():

    expected = pd.DataFrame(
        {
            "A": [0.04],
            "B": [0.02],
            "C": [0.00],
        }
    )

    result = shrink_expected_returns(
        expected,
        alpha=0.5,
    )

    # Cross-sectional mean = 0.02
    #
    # A: 0.02 + 0.5 * (0.04 - 0.02) = 0.03
    # B: 0.02 + 0.5 * (0.02 - 0.02) = 0.02
    # C: 0.02 + 0.5 * (0.00 - 0.02) = 0.01

    assert result.iloc[0]["A"] == pytest.approx(0.03)
    assert result.iloc[0]["B"] == pytest.approx(0.02)
    assert result.iloc[0]["C"] == pytest.approx(0.01)


def test_alpha_one_returns_original():

    expected = pd.DataFrame(
        {
            "A": [0.04],
            "B": [0.02],
            "C": [0.01],
        }
    )

    result = shrink_expected_returns(
        expected,
        alpha=1.0,
    )

    pd.testing.assert_frame_equal(
        result,
        expected,
    )


def test_alpha_zero_returns_cross_sectional_mean():

    expected = pd.DataFrame(
        {
            "A": [0.04],
            "B": [0.02],
            "C": [0.00],
        }
    )

    result = shrink_expected_returns(
        expected,
        alpha=0.0,
    )

    assert result.iloc[0]["A"] == pytest.approx(0.02)
    assert result.iloc[0]["B"] == pytest.approx(0.02)
    assert result.iloc[0]["C"] == pytest.approx(0.02)


def test_invalid_alpha():

    expected = pd.DataFrame(
        {
            "A": [0.01],
            "B": [0.02],
        }
    )

    with pytest.raises(ValueError):
        shrink_expected_returns(
            expected,
            alpha=1.5,
        )


def test_invalid_window():

    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02],
            "B": [0.02, 0.01],
        }
    )

    with pytest.raises(ValueError):
        historical_mean_expected_returns(
            returns,
            window=0,
        )