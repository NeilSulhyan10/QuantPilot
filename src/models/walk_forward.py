import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def walk_forward_predictions(
    features,
    targets,
    min_train_size=252,
    horizon=1,
    alpha=1.0,
):
    """
    Generate strictly out-of-sample walk-forward predictions.

    At prediction date t, only training observations whose
    forward-return targets are completely known by date t
    are eligible for training.

    For a horizon of H:
        target at date s uses returns from s+1 through s+H.

    Therefore, when predicting at date t, the latest usable
    training target is at date t-H.
    """

    if features.empty:
        raise ValueError("Features dataframe is empty.")

    if targets.empty:
        raise ValueError("Targets series is empty.")

    if min_train_size <= 0:
        raise ValueError(
            "min_train_size must be positive."
        )

    if horizon <= 0:
        raise ValueError(
            "horizon must be positive."
        )

    if alpha < 0:
        raise ValueError(
            "alpha must be non-negative."
        )

    if len(features) != len(targets):
        raise ValueError(
            "Features and targets must have the same length."
        )

    if not features.index.equals(targets.index):
        raise ValueError(
            "Features and targets must have identical indices."
        )

    if features.isna().any().any():
        raise ValueError(
            "Features contain missing values."
        )

    predictions = pd.Series(
        index=features.index,
        dtype=float,
        name="prediction",
    )

    X = features.to_numpy()
    y = targets.to_numpy()

    for i in range(len(features)):

        # Target at training date s uses returns through s + horizon.
        #
        # Therefore, when predicting at i, only targets with:
        #
        #     s + horizon <= i
        #
        # are completely known.
        #
        # The largest usable s is:
        #
        #     i - horizon
        #
        # Python slicing is exclusive, hence:
        train_end = i - horizon + 1

        if train_end <= 0:
            continue

        train_X = X[:train_end]
        train_y = y[:train_end]

        # Ignore training observations whose target is NaN.
        valid = pd.notna(train_y)

        if valid.sum() < min_train_size:
            continue

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    Ridge(alpha=alpha),
                ),
            ]
        )

        model.fit(
            train_X[valid],
            train_y[valid],
        )

        predictions.iloc[i] = model.predict(
            X[i:i + 1]
        )[0]

    return predictions