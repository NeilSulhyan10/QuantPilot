import numpy as np
import pandas as pd
import pytest

from src.models.ml_features import (
    calculate_ml_features,
    calculate_forward_return_target,
)


def make_returns():
    return pd.DataFrame(
        {
            "A": np.arange(1, 101) / 10000,
            "B": np.arange(101, 201) / 10000,
        },
        index=pd.date_range(
            "2020-01-01",
            periods=100,
            freq="D",
        ),
    )


def test_ml_features_shape():

    returns = make_returns()

    features = calculate_ml_features(
        returns
    )

    assert features.shape == (100, 10)


def test_ml_features_columns():

    returns = make_returns()

    features = calculate_ml_features(
        returns
    )

    expected = {
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volatility_60d",
        "mean_60d",
    }

    assert set(
        features.columns.get_level_values("feature")
    ) == expected


def test_ml_features_no_lookahead():

    returns = make_returns()

    features_1 = calculate_ml_features(
        returns
    )

    modified = returns.copy()

    # Change only future observations.
    modified.iloc[80:] *= 100

    features_2 = calculate_ml_features(
        modified
    )

    # Features before the modification must remain identical.
    pd.testing.assert_frame_equal(
        features_1.iloc[:80],
        features_2.iloc[:80],
    )


def test_forward_target_shape():

    returns = make_returns()

    target = calculate_forward_return_target(
        returns,
        horizon=21,
    )

    assert target.shape == returns.shape


def test_forward_target_uses_future_returns():

    returns = pd.DataFrame(
        {
            "A": [0.01] * 5
        },
        index=pd.date_range(
            "2020-01-01",
            periods=5,
            freq="D",
        ),
    )

    target = calculate_forward_return_target(
        returns,
        horizon=3,
    )

    expected = (
        (1.01 ** 3) - 1
    )

    assert target.iloc[0, 0] == pytest.approx(
        expected
    )


def test_invalid_horizon():

    returns = make_returns()

    with pytest.raises(ValueError):
        calculate_forward_return_target(
            returns,
            horizon=0,
        )