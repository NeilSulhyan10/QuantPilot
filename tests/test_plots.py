import numpy as np
import pandas as pd

from src.evaluation.plots import (
    calculate_drawdown,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_portfolio_weights,
    plot_turnover,
)


def sample_returns():
    index = pd.date_range(
        "2020-01-01",
        periods=4,
    )

    return {
        "Strategy A": pd.Series(
            [0.01, -0.02, 0.03, 0.01],
            index=index,
        ),
        "Strategy B": pd.Series(
            [0.02, 0.01, -0.01, 0.02],
            index=index,
        ),
    }


def test_calculate_drawdown():
    returns = pd.Series(
        [0.10, -0.20, 0.05],
        index=pd.date_range(
            "2020-01-01",
            periods=3,
        ),
    )

    drawdown = calculate_drawdown(returns)

    assert np.isclose(drawdown.iloc[0], 0.0)
    assert drawdown.iloc[1] < 0.0
    assert drawdown.iloc[2] < 0.0


def test_plot_cumulative_returns(tmp_path):
    output = tmp_path / "cumulative.png"

    plot_cumulative_returns(
        sample_returns(),
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_drawdowns(tmp_path):
    output = tmp_path / "drawdown.png"

    plot_drawdowns(
        sample_returns(),
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_portfolio_weights(tmp_path):
    output = tmp_path / "weights.png"

    weights = pd.DataFrame(
        {
            "A": [0.5, 0.4],
            "B": [0.5, 0.6],
        },
        index=pd.date_range(
            "2020-01-01",
            periods=2,
        ),
    )

    plot_portfolio_weights(
        weights,
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_turnover(tmp_path):
    output = tmp_path / "turnover.png"

    turnover = pd.Series(
        [0.1, 0.2, 0.15],
        index=pd.date_range(
            "2020-01-01",
            periods=3,
        ),
    )

    plot_turnover(
        turnover,
        output,
    )

    assert output.exists()
    assert output.stat().st_size > 0
