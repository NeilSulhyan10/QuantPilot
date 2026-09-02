import numpy as np
import pandas as pd
import pytest

from src.models.ml_expected_returns import (
    calculate_ml_expected_returns,
)


def make_returns(n=400):

    rng = np.random.default_rng(42)

    return pd.DataFrame(
        rng.normal(
            0.0005,
            0.01,
            size=(n, 3),
        ),
        columns=["A", "B", "C"],
        index=pd.date_range(
            "2020-01-01",
            periods=n,
            freq="D",
        ),
    )


def test_ml_expected_returns_shape():

    returns = make_returns()

    predictions = calculate_ml_expected_returns(
        returns,
        min_train_size=50,
        horizon=21,
    )

    assert predictions.shape == returns.shape


def test_ml_expected_returns_columns():

    returns = make_returns()

    predictions = calculate_ml_expected_returns(
        returns,
        min_train_size=50,
    )

    assert list(predictions.columns) == [
        "A",
        "B",
        "C",
    ]


def test_predictions_are_eventually_available():

    returns = make_returns()

    predictions = calculate_ml_expected_returns(
        returns,
        min_train_size=50,
    )

    assert predictions.notna().any().any()


def test_predictions_are_finite():

    returns = make_returns()

    predictions = calculate_ml_expected_returns(
        returns,
        min_train_size=50,
    )

    valid = predictions.stack().dropna()

    assert not valid.empty
    assert np.isfinite(valid).all()


def test_no_lookahead_from_future_returns():

    returns = make_returns()

    predictions_1 = calculate_ml_expected_returns(
        returns,
        min_train_size=50,
    )

    modified = returns.copy()

    # Modify only the final observation.
    modified.iloc[-1] += 10.0

    predictions_2 = calculate_ml_expected_returns(
        modified,
        min_train_size=50,
    )

    # Predictions sufficiently before the modified observation
    # must remain unchanged.
    comparison_index = returns.index[:-30]

    pd.testing.assert_frame_equal(
        predictions_1.loc[comparison_index],
        predictions_2.loc[comparison_index],
    )


def test_invalid_training_size():

    returns = make_returns()

    with pytest.raises(ValueError):

        calculate_ml_expected_returns(
            returns,
            min_train_size=0,
        )   