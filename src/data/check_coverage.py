from pathlib import Path

import pandas as pd

from reader import load_stock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def check_coverage() -> None:
    """Check date coverage across all processed stock datasets."""
    files = sorted(PROCESSED_DATA_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError("No processed datasets found.")

    date_ranges = []

    for path in files:
        ticker = path.stem
        data = load_stock(ticker)

        date_ranges.append(
            {
                "ticker": ticker,
                "start": data.index.min(),
                "end": data.index.max(),
                "rows": len(data),
            }
        )

    coverage = pd.DataFrame(date_ranges)
    coverage = coverage.sort_values("ticker")

    print("\nIndividual coverage:\n")
    print(coverage.to_string(index=False))

    print("\nOverall coverage:")
    print(f"Earliest date: {coverage['start'].min().date()}")
    print(f"Latest date:   {coverage['end'].max().date()}")

    common_start = coverage["start"].max()
    common_end = coverage["end"].min()

    print("\nCommon date range:")
    print(f"Start: {common_start.date()}")
    print(f"End:   {common_end.date()}")


if __name__ == "__main__":
    check_coverage()
