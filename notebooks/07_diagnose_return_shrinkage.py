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
# Baseline expected returns
# ---------------------------------------------------------

expected_returns = rolling_mean(
    returns,
    window=60,
)


# ---------------------------------------------------------
# Shrinkage experiment
# ---------------------------------------------------------

alphas = [1.00, 0.75, 0.50, 0.25]

results = []

for alpha in alphas:

    # Cross-sectional mean for each date
    cross_sectional_mean = expected_returns.mean(axis=1)

    # Broadcast the mean across all stocks
    shrunk = (
        cross_sectional_mean.to_numpy()[:, None]
        + alpha
        * (
            expected_returns.to_numpy()
            - cross_sectional_mean.to_numpy()[:, None]
        )
    )

    shrunk = pd.DataFrame(
        shrunk,
        index=expected_returns.index,
        columns=expected_returns.columns,
    )

    valid = shrunk.dropna(how="all")

    flat = valid.stack()

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
            "alpha": alpha,
            "mean_daily_return": flat.mean(),
            "median_daily_return": flat.median(),
            "mean_annualized_return": flat.mean() * 252,
            "median_annualized_return": flat.median() * 252,
            "estimate_std": flat.std(),
            "rank_change_mean": rank_changes.mean(),
            "rank_change_median": rank_changes.median(),
            "rank_change_max": rank_changes.max(),
            "average_spread": spread.mean(),
            "maximum_spread": spread.max(),
        }
    )

# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

comparison = pd.DataFrame(results)

print("=" * 80)
print("EXPECTED RETURN SHRINKAGE DIAGNOSTIC")
print("=" * 80)

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


print("\nInterpretation")
print("--------------")

for _, row in comparison.iterrows():

    print(
        f"\nAlpha = {row['alpha']:.2f}"
    )

    print(
        f"Mean annualized return: "
        f"{row['mean_annualized_return']:.2%}"
    )

    print(
        f"Median annualized return: "
        f"{row['median_annualized_return']:.2%}"
    )

    print(
        f"Estimate std: "
        f"{row['estimate_std']:.6f}"
    )

    print(
        f"Average rank change: "
        f"{row['rank_change_mean']:.3f}"
    )

    print(
        f"Average cross-sectional spread: "
        f"{row['average_spread']:.6f}"
    )


print("\nDiagnostic complete.")