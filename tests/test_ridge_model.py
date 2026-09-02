import numpy as np
import pandas as pd
import pytest

from src.models.ridge_model import (
    RidgeExpectedReturnModel,
)


def make_dataset():

    rng = np.random.default_rng(42)

    features = pd.DataFrame(
        rng.normal(size=(100, 5)),
        columns=[
            "return_5d",
            "return_20d",
            "volatility_20d",
            "volatility_60d",
            "mean_60d",
        ],
        index=pd.date_range(
            "2020-01-01",
            periods=100,
            freq="D",
        ),
    )

    targets = pd.Series(
        (
            0.5 * features["return_5d"]
            - 0.2 * features["volatility_20d"]
            + rng.normal(0, 0.1, 100)
        ),
        index=features.index,
    )

    return features, targets


def test_ridge_model_fit():

    features, targets = make_dataset()

    model = RidgeExpectedReturnModel()

    model.fit(features, targets)

    assert "asset" in model.models


def test_ridge_model_predict():

    features, targets = make_dataset()

    model = RidgeExpectedReturnModel()

    model.fit(features, targets)

    predictions = model.predict(
        features.iloc[:10]
    )

    assert len(predictions) == 10
    assert predictions.index.equals(
        features.iloc[:10].index
    )
    assert predictions.notna().all()


def test_prediction_is_deterministic():

    features, targets = make_dataset()

    model = RidgeExpectedReturnModel()

    model.fit(features, targets)

    predictions_1 = model.predict(
        features.iloc[:10]
    )

    predictions_2 = model.predict(
        features.iloc[:10]
    )

    pd.testing.assert_series_equal(
        predictions_1,
        predictions_2,
    )


def test_predict_before_fit():

    features, _ = make_dataset()

    model = RidgeExpectedReturnModel()

    with pytest.raises(ValueError):
        model.predict(features)


def test_invalid_alpha():

    with pytest.raises(ValueError):
        RidgeExpectedReturnModel(alpha=-1)


def test_mismatched_lengths():

    features, targets = make_dataset()

    model = RidgeExpectedReturnModel()

    with pytest.raises(ValueError):
        model.fit(
            features,
            targets.iloc[:-1],
        )