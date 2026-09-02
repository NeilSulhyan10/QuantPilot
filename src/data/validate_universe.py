from pathlib import Path

from validation import validate_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def validate_universe() -> None:
    """Validate every Parquet dataset in the raw-data directory."""
    files = sorted(RAW_DATA_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError("No Parquet files found in data/raw/")

    failures = []

    for path in files:
        ticker = path.stem

        try:
            data = validate_file(path)

            print(
                f"✓ {ticker:<6} "
                f"{len(data):>5} rows | "
                f"{data.index.min().date()} → "
                f"{data.index.max().date()}"
            )

        except Exception as error:
            failures.append((ticker, str(error)))
            print(f"✗ {ticker:<6} FAILED: {error}")

    print()
    print(f"Datasets checked: {len(files)}")
    print(f"Passed: {len(files) - len(failures)}")
    print(f"Failed: {len(failures)}")

    if failures:
        raise RuntimeError("One or more datasets failed validation.")


if __name__ == "__main__":
    validate_universe()
