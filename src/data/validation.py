from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert yfinance MultiIndex columns into simple column names.

    Expected input:
        Price -> Ticker -> ...
    Expected output:
        Open, High, Low, Close, Volume
    """
    data = data.copy()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns.name = None

    return data


def validate_columns(data: pd.DataFrame) -> None:
    """Ensure all required market-data columns exist."""
    missing = set(REQUIRED_COLUMNS) - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def validate_dates(data: pd.DataFrame) -> None:
    """Validate the date index."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data index must be a pandas DatetimeIndex")

    if data.index.has_duplicates:
        raise ValueError("Duplicate dates detected")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Dates are not sorted in ascending order")


def validate_missing_values(data: pd.DataFrame) -> None:
    """Check for missing values in required columns."""
    missing = data[REQUIRED_COLUMNS].isna().sum()

    if missing.any():
        problems = missing[missing > 0].to_dict()
        raise ValueError(f"Missing values detected: {problems}")


def validate_prices(data: pd.DataFrame, tolerance: float = 1e-6) -> None:
    """Validate OHLC price relationships and positivity."""
    price_columns = ["Open", "High", "Low", "Close"]

    if (data[price_columns] <= 0).any().any():
        raise ValueError("Non-positive price detected")

    if (data["High"] + tolerance < data["Low"]).any():
        raise ValueError("High price is below Low price")

    if (data["Open"] > data["High"] + tolerance).any():
        raise ValueError("Open price is above High price")

    if (data["Open"] + tolerance < data["Low"]).any():
        raise ValueError("Open price is below Low price")

    if (data["Close"] > data["High"] + tolerance).any():
        raise ValueError("Close price is above High price")

    if (data["Close"] + tolerance < data["Low"]).any():
        raise ValueError("Close price is below Low price")


def validate_volume(data: pd.DataFrame) -> None:
    """Validate trading volume."""
    if (data["Volume"] < 0).any():
        raise ValueError("Negative trading volume detected")


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and validate a market-data DataFrame.

    Returns:
        A normalized copy of the validated data.
    """
    if data.empty:
        raise ValueError("Dataset is empty")

    data = normalize_columns(data)

    validate_columns(data)
    validate_dates(data)
    validate_missing_values(data)
    validate_prices(data)
    validate_volume(data)

    return data


def validate_file(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file and validate it."""
    path = Path(path)

    data = pd.read_parquet(path)

    return validate_data(data)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "AAPL.parquet"

    validated_data = validate_file(data_path)

    print("Validation successful.")
    print()
    print(validated_data.head())
    print()
    print(validated_data.dtypes)
