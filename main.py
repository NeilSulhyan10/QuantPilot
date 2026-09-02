from pathlib import Path

from src.config import load_config
from src.backtesting.config import BacktestConfig
from src.backtesting.engine import WalkForwardBacktester
from src.backtesting.benchmark import EqualWeightBacktester
from src.backtesting.buy_and_hold import BuyAndHoldBacktester
from src.data.reader import load_universe
from src.data.benchmark_reader import load_benchmark
from src.features.returns import calculate_return_matrix
from src.features.covariance import rolling_covariance
from src.models.expected_returns import calculate_expected_returns
from src.evaluation.benchmark_comparison import (
    evaluate_strategy_comparison,
)
from src.evaluation.portfolio_diagnostics import (
    portfolio_diagnostics,
)
from src.evaluation.plots import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_portfolio_weights,
    plot_turnover,
)

def main() -> None:
    config = load_config()

    print("=" * 72)
    print(f"{config.project_name} v{config.project_version}")
    print("Canonical Portfolio Comparison")
    print("=" * 72)

    # ---------------------------------------------------------
    # 1. Load universe
    # ---------------------------------------------------------

    tickers = list(config.universe.tickers)

    stocks = load_universe(tickers)

    print(f"\nUniverse: {len(tickers)} stocks")

    # ---------------------------------------------------------
    # 2. Calculate asset returns
    # ---------------------------------------------------------

    returns = calculate_return_matrix(
        stocks,
        method=config.returns.method,
    )

    print(
        f"Return observations: {len(returns)} "
        f"({returns.index.min().date()} → "
        f"{returns.index.max().date()})"
    )

    # ---------------------------------------------------------
    # 3. QuantPilot expected returns
    # ---------------------------------------------------------

    expected_returns = calculate_expected_returns(
        returns,
        window=config.expected_returns.window,
        alpha=config.expected_returns.shrinkage_alpha,
    )

    expected_returns = expected_returns.dropna(how="any")

    # ---------------------------------------------------------
    # 4. Rolling covariance
    # ---------------------------------------------------------

    covariance_matrices = rolling_covariance(
        returns,
        window=config.covariance.window,
    )

    # ---------------------------------------------------------
    # 5. QuantPilot V2
    # ---------------------------------------------------------

    backtest_config = BacktestConfig.from_quantpilot_config(
        config
    )

    quantpilot = WalkForwardBacktester(
        config=backtest_config,
    )

    quantpilot_result = quantpilot.run(
        returns=returns,
        expected_returns=expected_returns,
        covariance_matrices=covariance_matrices,
    )

    # ---------------------------------------------------------
    # 6. Equal Weight
    # ---------------------------------------------------------

    equal_weight = EqualWeightBacktester(
        config=backtest_config
    )

    equal_weight_result = equal_weight.run(
        returns=returns,
    )

    # ---------------------------------------------------------
    # 7. Buy & Hold
    # ---------------------------------------------------------

    buy_and_hold = BuyAndHoldBacktester()

    buy_and_hold_result = buy_and_hold.run(
        returns=returns,
    )

    # ---------------------------------------------------------
    # 8. SPY
    # ---------------------------------------------------------

    spy_data = load_benchmark(
        config.benchmarks.market
    )

    spy_returns = spy_data["Close"].pct_change().dropna()

    # ---------------------------------------------------------
    # 9. Strategy comparison
    # ---------------------------------------------------------

    strategy_returns = {
        "QuantPilot V2": quantpilot_result.returns,
        "Equal Weight": equal_weight_result.returns,
        "Buy & Hold": buy_and_hold_result.returns,
        config.benchmarks.market: spy_returns,
    }

    comparison = evaluate_strategy_comparison(
        strategy_returns
    )

    # ---------------------------------------------------------
    # 10. Common evaluation period
    # ---------------------------------------------------------

    common_index = strategy_returns[
        "QuantPilot V2"
    ].index

    for series in strategy_returns.values():
        common_index = common_index.intersection(
            series.index
        )

    common_index = common_index.sort_values()

    common_start = common_index[0]
    common_end = common_index[-1]

    # ---------------------------------------------------------
    # 11. Persist final research outputs
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parent

    results_dir = project_root / "results"
    backtests_dir = results_dir / "backtests"
    comparisons_dir = results_dir / "comparisons"
    figures_dir = results_dir / "figures"

    backtests_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparisons_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # QuantPilot daily returns
    quantpilot_result.returns.rename(
        "return"
    ).to_csv(
        backtests_dir / "quantpilot_v2_returns.csv",
        index_label="date",
    )

    # QuantPilot portfolio weights
    quantpilot_result.weights.to_csv(
        backtests_dir / "quantpilot_v2_weights.csv",
        index_label="date",
    )

    # Turnover
    quantpilot_result.turnover.rename(
        "turnover"
    ).to_csv(
        backtests_dir / "quantpilot_v2_turnover.csv",
        index_label="date",
    )

    # Transaction costs
    quantpilot_result.transaction_costs.rename(
        "transaction_cost"
    ).to_csv(
        backtests_dir / "quantpilot_v2_transaction_costs.csv",
        index_label="date",
    )

    # Slippage
    quantpilot_result.slippage.rename(
        "slippage"
    ).to_csv(
        backtests_dir / "quantpilot_v2_slippage.csv",
        index_label="date",
    )

    # Strategy comparison table
    comparison.to_csv(
        comparisons_dir / "strategy_comparison.csv"
    )

    plot_cumulative_returns(
        strategy_returns,
        figures_dir / "cumulative_returns.png",
    )

    plot_drawdowns(
        strategy_returns,
        figures_dir / "drawdown.png",
    )

    plot_portfolio_weights(
        quantpilot_result.weights,
        figures_dir / "portfolio_weights.png",
    )

    plot_turnover(
        quantpilot_result.turnover,
        figures_dir / "turnover.png",
    )

    # ---------------------------------------------------------
    # 12. Portfolio diagnostics
    # ---------------------------------------------------------

    diagnostics = portfolio_diagnostics(
        weights=quantpilot_result.weights,
        turnover=quantpilot_result.turnover,
        max_weight=config.optimizer.max_weight,
    )

    # Average weights
    diagnostics["average_weights"].rename(
        "average_weight"
    ).to_csv(
        backtests_dir / "quantpilot_v2_average_weights.csv",
        index_label="ticker",
    )

    # Maximum weights
    diagnostics["maximum_weights"].rename(
        "maximum_weight"
    ).to_csv(
        backtests_dir / "quantpilot_v2_maximum_weights.csv",
        index_label="ticker",
    )

    # Active positions
    diagnostics["active_positions"].rename(
        "active_positions"
    ).to_csv(
        backtests_dir / "quantpilot_v2_active_positions.csv",
        index_label="date",
    )

    # Effective number of stocks
    diagnostics["effective_number_of_stocks"].rename(
        "effective_number_of_stocks"
    ).to_csv(
        backtests_dir / "quantpilot_v2_effective_stocks.csv",
        index_label="date",
    )

    # Cap hit frequency
    diagnostics["cap_hit_frequency"].rename(
        "cap_hit_frequency"
    ).to_csv(
        backtests_dir / "quantpilot_v2_cap_hit_frequency.csv",
        index_label="ticker",
    )

    # Turnover statistics
    diagnostics["turnover_statistics"].rename(
        "value"
    ).to_csv(
        backtests_dir / "quantpilot_v2_turnover_statistics.csv",
        index_label="statistic",
    )

    # ---------------------------------------------------------
    # 13. Performance report
    # ---------------------------------------------------------

    print("\nCommon Evaluation")
    print("-" * 72)

    print(
        f"Start: {common_start.date()}"
    )

    print(
        f"End:   {common_end.date()}"
    )

    print("\nPerformance Comparison")
    print("-" * 72)

    display_columns = [
        "cumulative_return",
        "cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
    ]

    display = comparison[display_columns].copy()

    display["cumulative_return"] *= 100
    display["cagr"] *= 100
    display["annualized_volatility"] *= 100
    display["maximum_drawdown"] *= 100

    display.columns = [
        "Cumulative %",
        "CAGR %",
        "Volatility %",
        "Sharpe",
        "Sortino",
        "Max Drawdown %",
    ]

    print(
        display.to_string(
            float_format=lambda x: f"{x:.2f}"
        )
    )

    # ---------------------------------------------------------
    # 14. QuantPilot trading statistics
    # ---------------------------------------------------------

    print("\nQuantPilot Trading Statistics")
    print("-" * 72)

    print(
        f"Rebalances:              "
        f"{len(quantpilot_result.turnover)}"
    )

    print(
        f"Average turnover:        "
        f"{quantpilot_result.turnover.mean():.2%}"
    )

    print(
        f"Transaction costs:       "
        f"{quantpilot_result.transaction_costs.sum():.4f}"
    )

    print(
        f"Slippage:                "
        f"{quantpilot_result.slippage.sum():.4f}"
    )

    # ---------------------------------------------------------
    # 15. Portfolio diagnostics report
    # ---------------------------------------------------------

    print("\nPortfolio Diagnostics")
    print("-" * 72)

    print(
        f"Average active positions: "
        f"{diagnostics['active_positions'].mean():.2f}"
    )

    print(
        f"Average effective stocks: "
        f"{diagnostics['effective_number_of_stocks'].mean():.2f}"
    )

    print(
        f"Positions at "
        f"{config.optimizer.max_weight:.0%} cap: "
        f"{(diagnostics['cap_hit_frequency'] > 0).sum()}"
    )

    print("\nTop Average Weights")
    print("-" * 72)

    print(
        diagnostics["average_weights"]
        .head(10)
        .mul(100)
        .to_string(
            float_format=lambda x: f"{x:.2f}%"
        )
    )

    # ---------------------------------------------------------
    # 16. Saved results
    # ---------------------------------------------------------

    print("\nSaved Results")
    print("-" * 72)

    print(
        f"Backtests:    {backtests_dir}"
    )

    print(
        f"Comparisons:  {comparisons_dir}"
    )

    print(
        f"Figures:      {figures_dir}"
    )


if __name__ == "__main__":
    main()