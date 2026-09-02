from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _validate_returns(
    strategy_returns: dict[str, pd.Series],
) -> None:
    if not strategy_returns:
        raise ValueError("strategy_returns cannot be empty")

    for name, returns in strategy_returns.items():
        if returns.empty:
            raise ValueError(
                f"Returns for {name} cannot be empty"
            )

        if returns.isna().any():
            raise ValueError(
                f"Returns for {name} cannot contain NaN values"
            )


def plot_cumulative_returns(
    strategy_returns: dict[str, pd.Series],
    output_path: str | Path,
) -> None:
    """
    Plot cumulative growth of $1 for each strategy.
    """
    _validate_returns(strategy_returns)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 6))

    for name, returns in strategy_returns.items():
        cumulative = (1.0 + returns).cumprod()
        plt.plot(
            cumulative.index,
            cumulative.values,
            label=name,
        )

    plt.title("QuantPilot — Cumulative Portfolio Growth")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def calculate_drawdown(
    returns: pd.Series,
) -> pd.Series:
    """
    Calculate portfolio drawdown from cumulative wealth.
    """
    if returns.empty:
        raise ValueError("returns cannot be empty")

    if returns.isna().any():
        raise ValueError("returns cannot contain NaN values")

    wealth = (1.0 + returns).cumprod()
    running_max = wealth.cummax()

    return wealth / running_max - 1.0


def plot_drawdowns(
    strategy_returns: dict[str, pd.Series],
    output_path: str | Path,
) -> None:
    """
    Plot drawdown curves for each strategy.
    """
    _validate_returns(strategy_returns)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 6))

    for name, returns in strategy_returns.items():
        drawdown = calculate_drawdown(returns)

        plt.plot(
            drawdown.index,
            drawdown.values,
            label=name,
        )

    plt.title("QuantPilot — Portfolio Drawdowns")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_portfolio_weights(
    weights: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot portfolio weights across rebalance dates.
    """
    if weights.empty:
        raise ValueError("weights cannot be empty")

    if weights.isna().any().any():
        raise ValueError(
            "weights cannot contain NaN values"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(14, 8))

    plt.stackplot(
        weights.index,
        weights.T.values,
        labels=weights.columns,
    )

    plt.title("QuantPilot — Portfolio Weights")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Weight")
    plt.ylim(0, 1.0)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=8,
    )
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_turnover(
    turnover: pd.Series,
    output_path: str | Path,
) -> None:
    """
    Plot turnover at each rebalance.
    """
    if turnover.empty:
        raise ValueError("turnover cannot be empty")

    if turnover.isna().any():
        raise ValueError(
            "turnover cannot contain NaN values"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        turnover.index,
        turnover.values,
    )

    plt.title("QuantPilot — Portfolio Turnover")
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()
