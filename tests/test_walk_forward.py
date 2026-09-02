import numpy as np
import pandas as pd
import pytest

from src.models.walk_forward import (
    walk_forward_predictions,
)


def make_dataset(n=30):

    index = pd.date_range(
        "2020-01-01",
        periods=n,
        freq="D",
    )

    features = pd.DataFrame(
        {
            "feature_1": np.arange(n, dtype=float),
            "feature_2": np.arange(n, dtype=float) ** 2,
        },
        index=index,
    )

    targets = pd.Series(
        0.01 * features["feature_1"],
        index=index,
        name="target",
    )

    return features, targets


def test_walk_forward_respects_min_train_size():

    features, targets = make_dataset()

    predictions = walk_forward_predictions(
        features,
        targets,
        min_train_size=10,
    )

    assert predictions.iloc[:10].isna().all()

    assert predictions.iloc[10:].notna().all()


def test_walk_forward_predictions_have_correct_index():

    features, targets = make_dataset()

    predictions = walk_forward_predictions(
        features,
        targets,
        min_train_size=10,
    )

    assert predictions.index.equals(
        features.index
    )


def test_walk_forward_predictions_are_finite():

    features, targets = make_dataset()

    predictions = walk_forward_predictions(
        features,
        targets,
        min_train_size=10,
    )

    valid = predictions.dropna()

    assert np.isfinite(valid).all()


def test_future_targets_are_not_used():

    features, targets = make_dataset()

    predictions_1 = walk_forward_predictions(
        features,
        targets,
        min_train_size=10,
    )

    modified_targets = targets.copy()

    # Change only a future target.
    modified_targets.iloc[-1] = 999.0

    predictions_2 = walk_forward_predictions(
        features,
        modified_targets,
        min_train_size=10,
    )

    # Earlier predictions must remain unchanged.
    pd.testing.assert_series_equal(
        predictions_1.iloc[:-1],
        predictions_2.iloc[:-1],
    )


def test_missing_training_targets_are_skipped():

    features, targets = make_dataset()

    targets.iloc[5:8] = np.nan

    predictions = walk_forward_predictions(
        features,
        targets,
        min_train_size=10,
    )

    assert predictions.notna().sum() > 0


def test_invalid_min_train_size():

    features, targets = make_dataset()

    with pytest.raises(ValueError):

        walk_forward_predictions(
            features,
            targets,
            min_train_size=0,
        )

def test_future_label_not_used_before_completion():
    features = pd.DataFrame(
        {
            "x1": range(40),
            "x2": range(40, 80),
        },
        dtype=float,
    )

    targets = pd.Series(
        [float(i) for i in range(40)],
        dtype=float,
    )

    predictions_1 = walk_forward_predictions(
        features,
        targets,
        min_train_size=5,
        horizon=3,
        alpha=1.0,
    )

    modified_targets = targets.copy()

    # Target at index 11 requires information through
    # index 14, so it must NOT affect prediction at index 13.
    modified_targets.iloc[11] = 999.0

    predictions_2 = walk_forward_predictions(
        features,
        modified_targets,
        min_train_size=5,
        horizon=3,
        alpha=1.0,
    )

    assert predictions_1.iloc[13] == predictions_2.iloc[13]