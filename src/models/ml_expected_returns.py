import pandas as pd

from src.models.ml_features import (
    calculate_forward_return_target,
    calculate_ml_features,
)
from src.models.walk_forward import walk_forward_predictions


def calculate_ml_expected_returns(
    returns: pd.DataFrame,
    min_train_size: int = 252,
    horizon: int = 21,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Generate walk-forward ML expected returns for every asset.

    Each asset gets its own Ridge model. Predictions are strictly
    out-of-sample.

    The model predicts a cumulative forward return over `horizon`
    trading days. This is converted to a daily-equivalent expected
    return before being passed to the portfolio optimizer.
    """

    if returns.empty:
        raise ValueError("Returns dataframe is empty.")

    if returns.isna().any().any():
        raise ValueError(
            "Returns dataframe contains missing values."
        )

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

    if len(returns) < min_train_size:
        raise ValueError(
            "Returns dataframe is shorter than min_train_size."
        )

    predictions = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    for ticker in returns.columns:

        asset_returns = returns[[ticker]]

        # --------------------------------------------------
        # Build features
        # --------------------------------------------------

        features = calculate_ml_features(
            asset_returns
        )

        # Features are MultiIndex because the feature builder
        # supports multiple assets.
        features.columns = features.columns.droplevel(
            "ticker"
        )

        # --------------------------------------------------
        # Build forward-return target
        # --------------------------------------------------

        targets = calculate_forward_return_target(
            asset_returns,
            horizon=horizon,
        )[ticker]

        # Remove rows where features are unavailable.
        valid = features.notna().all(axis=1)

        features = features.loc[valid]
        targets = targets.loc[valid]

        # --------------------------------------------------
        # Strict walk-forward prediction
        # --------------------------------------------------

        asset_predictions = walk_forward_predictions(
            features,
            targets,
            min_train_size=min_train_size,
            horizon=horizon,
            alpha=alpha,
        )

        # --------------------------------------------------
        # Convert cumulative H-day prediction to
        # daily-equivalent expected return.
        # --------------------------------------------------

        valid_predictions = asset_predictions.notna()

        daily_predictions = pd.Series(
            index=asset_predictions.index,
            dtype=float,
        )

        # Ridge is unconstrained and can theoretically produce
        # an economically impossible prediction below -100%.
        # Treat such predictions as invalid rather than clipping
        # them or crashing the complete walk-forward process.
        valid_for_conversion = (
            valid_predictions
            & (asset_predictions > -1.0)
        )

        daily_predictions.loc[valid_for_conversion] = (
            (
                1.0
                + asset_predictions.loc[
                    valid_for_conversion
                ]
            )
            ** (1.0 / horizon)
            - 1.0
        )

        # --------------------------------------------------
        # Store predictions for this asset
        # --------------------------------------------------

        predictions.loc[
            daily_predictions.index,
            ticker,
        ] = daily_predictions

    return predictions