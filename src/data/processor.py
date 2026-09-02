from pathlib import Path

import pandas as pd

from src.data.validation import validate_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

COLUMN_ORDER = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


def process_file(input_path: Path, output_path: Path) -> Path:
    """
    Load, normalize, validate, and save one market-data file.
    """
    data = pd.read_parquet(input_path)

    # Normalize and validate the dataset.
    data = validate_data(data)

    # Standardize column order.
    data = data[COLUMN_ORDER]

    # Make sure the index is sorted.
    data = data.sort_index()

    # Create output directory if necessary.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed dataset.
    data.to_parquet(output_path)

    return output_path


def process_universe() -> None:
    """Process every raw Parquet file in the universe."""
    files = sorted(RAW_DATA_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {RAW_DATA_DIR}"
        )

    for input_path in files:
        ticker = input_path.stem
        output_path = PROCESSED_DATA_DIR / f"{ticker}.parquet"

        print(f"Processing {ticker}...")

        process_file(
            input_path=input_path,
            output_path=output_path,
        )

        print(f"Saved → {output_path}")


if __name__ == "__main__":
    process_universe()
