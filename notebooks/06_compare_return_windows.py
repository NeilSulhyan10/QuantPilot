from pathlib import Path

import pandas as pd
import yaml

from src.data.reader import load_universe
from src.features.returns import calculate_return_matrix
from src.features.statistics import rolling_mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------

with open(PROJECT_ROOT / "configs" / "universe.yaml", "r") as f:
    config = yaml.safe_load(f)

tickers = config["universe"]["tickers"]


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

price_data = load_universe(tickers)

returns = calculate_return_matrix(
    price_data,
    price_column="Close",
    method="simple",
)


# ---------------------------------------------------------
# Compare estimation windows
# ---------------------------------------------------------

windows = [60, 120, 252]

results = []

for window in windows:

    expected_returns = rolling_mean(
        returns,
        window=window,
    )

    valid = expected_returns.dropna(how="all")

    flat = valid.stack()

    annualized = flat * 252

    rankings = valid.rank(
        axis=1,
        ascending=False,
        method="min",
    )

    rank_changes = rankings.diff().abs().stack()

    spread = (
        valid.max(axis=1)
        - valid.min(axis=1)
    )

    results.append(
        {
            "window": window,
            "mean_daily_expected_return": flat.mean(),
            "median_daily_expected_return": flat.median(),
            "mean_annualized_expected_return": annualized.mean(),
            "median_annualized_expected_return": annualized.median(),
            "expected_return_std": flat.std(),
            "rank_change_mean": rank_changes.mean(),
            "rank_change_median": rank_changes.median(),
            "rank_change_max": rank_changes.max(),
            "average_cross_sectional_spread": spread.mean(),
            "maximum_cross_sectional_spread": spread.max(),
        }
    )


# ---------------------------------------------------------
# Display comparison
# ---------------------------------------------------------

comparison = pd.DataFrame(results)

print("=" * 80)
print("EXPECTED RETURN WINDOW COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ---------------------------------------------------------
# More readable annualized summary
# ---------------------------------------------------------

print("\nAnnualized expected-return summary")
print("----------------------------------")

for _, row in comparison.iterrows():

    print(
        f"\n{int(row['window'])}-day window"
    )

    print(
        f"Mean annualized estimate:   "
        f"{row['mean_annualized_expected_return']:.2%}"
    )

    print(
        f"Median annualized estimate: "
        f"{row['median_annualized_expected_return']:.2%}"
    )

    print(
        f"Estimate std:               "
        f"{row['expected_return_std']:.6f}"
    )

    print(
        f"Average rank change:        "
        f"{row['rank_change_mean']:.3f}"
    )

    print(
        f"Median rank change:         "
        f"{row['rank_change_median']:.3f}"
    )

    print(
        f"Maximum rank change:        "
        f"{row['rank_change_max']:.0f}"
    )

    print(
        f"Average cross-sectional spread: "
        f"{row['average_cross_sectional_spread']:.6f}"
    )


print("\nDiagnostic complete.")