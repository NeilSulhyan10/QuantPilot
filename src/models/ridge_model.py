import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RidgeExpectedReturnModel:
    """
    Ridge Regression model for predicting future asset returns.

    The model is trained separately for each asset.
    """

    def __init__(
        self,
        alpha: float = 1.0,
    ):
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        self.alpha = alpha
        self.models: dict[str, Pipeline] = {}
        self.feature_columns: list[str] | None = None

    def fit(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
    ) -> None:
        """
        Fit a Ridge model using historical observations.
        """

        if features.empty:
            raise ValueError("Features dataframe is empty.")

        if targets.empty:
            raise ValueError("Targets series is empty.")

        if len(features) != len(targets):
            raise ValueError(
                "Features and targets must have the same length."
            )

        if features.isna().any().any():
            raise ValueError(
                "Features contain missing values."
            )

        if targets.isna().any():
            raise ValueError(
                "Targets contain missing values."
            )

        self.feature_columns = list(features.columns)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    Ridge(alpha=self.alpha),
                ),
            ]
        )

        model.fit(features, targets)

        self.models["asset"] = model

    def predict(
        self,
        features: pd.DataFrame,
    ) -> pd.Series:
        """
        Generate predictions for new observations.
        """

        if not self.models:
            raise ValueError(
                "Model has not been fitted."
            )

        if features.empty:
            raise ValueError(
                "Features dataframe is empty."
            )

        if self.feature_columns is None:
            raise ValueError(
                "Feature columns are not available."
            )

        if list(features.columns) != self.feature_columns:
            raise ValueError(
                "Feature columns do not match training columns."
            )

        if features.isna().any().any():
            raise ValueError(
                "Features contain missing values."
            )

        predictions = self.models["asset"].predict(
            features
        )

        return pd.Series(
            predictions,
            index=features.index,
            name="prediction",
        )