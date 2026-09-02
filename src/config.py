from dataclasses import dataclass
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "quantpilot.yaml"


@dataclass(frozen=True)
class UniverseConfig:
    name: str
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str | None
    interval: str
    auto_adjust: bool


@dataclass(frozen=True)
class ReturnsConfig:
    method: str


@dataclass(frozen=True)
class ExpectedReturnsConfig:
    model: str
    window: int
    shrinkage_alpha: float


@dataclass(frozen=True)
class CovarianceConfig:
    window: int


@dataclass(frozen=True)
class OptimizerConfig:
    method: str
    risk_aversion: float
    max_weight: float
    long_only: bool
    fully_invested: bool


@dataclass(frozen=True)
class RebalancingConfig:
    frequency: str


@dataclass(frozen=True)
class CostsConfig:
    transaction_cost_bps: float
    slippage_bps: float


@dataclass(frozen=True)
class BacktestConfigFile:
    walk_forward: bool
    drift_aware: bool


@dataclass(frozen=True)
class MLConfig:
    enabled: bool
    model: str
    min_train_size: int
    horizon: int
    alpha: float


@dataclass(frozen=True)
class BenchmarkConfig:
    equal_weight: bool
    buy_and_hold: bool
    market: str


@dataclass(frozen=True)
class QuantPilotConfig:
    project_name: str
    project_version: str
    universe: UniverseConfig
    data: DataConfig
    returns: ReturnsConfig
    expected_returns: ExpectedReturnsConfig
    covariance: CovarianceConfig
    optimizer: OptimizerConfig
    rebalancing: RebalancingConfig
    costs: CostsConfig
    backtest: BacktestConfigFile
    ml: MLConfig
    benchmarks: BenchmarkConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> QuantPilotConfig:
    """Load and validate the master QuantPilot configuration."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError("Configuration must contain a YAML mapping.")

    universe = raw["universe"]
    data = raw["data"]
    returns = raw["returns"]
    expected_returns = raw["expected_returns"]
    covariance = raw["covariance"]
    optimizer = raw["optimizer"]
    rebalancing = raw["rebalancing"]
    costs = raw["costs"]
    backtest = raw["backtest"]
    ml = raw["ml"]
    benchmarks = raw["benchmarks"]

    tickers = tuple(str(ticker).upper() for ticker in universe["tickers"])

    if not tickers:
        raise ValueError("Universe must contain at least one ticker.")

    if len(set(tickers)) != len(tickers):
        raise ValueError("Universe contains duplicate tickers.")

    if not 0.0 <= float(expected_returns["shrinkage_alpha"]) <= 1.0:
        raise ValueError("shrinkage_alpha must be between 0 and 1.")

    if not 0.0 < float(optimizer["max_weight"]) <= 1.0:
        raise ValueError("max_weight must be greater than 0 and at most 1.")

    if float(costs["transaction_cost_bps"]) < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")

    if float(costs["slippage_bps"]) < 0:
        raise ValueError("slippage_bps cannot be negative.")

    return QuantPilotConfig(
        project_name=str(raw["project"]["name"]),
        project_version=str(raw["project"]["version"]),
        universe=UniverseConfig(
            name=str(universe["name"]),
            tickers=tickers,
        ),
        data=DataConfig(
            start_date=str(data["start_date"]),
            end_date=data["end_date"],
            interval=str(data["interval"]),
            auto_adjust=bool(data["auto_adjust"]),
        ),
        returns=ReturnsConfig(
            method=str(returns["method"]),
        ),
        expected_returns=ExpectedReturnsConfig(
            model=str(expected_returns["model"]),
            window=int(expected_returns["window"]),
            shrinkage_alpha=float(expected_returns["shrinkage_alpha"]),
        ),
        covariance=CovarianceConfig(
            window=int(covariance["window"]),
        ),
        optimizer=OptimizerConfig(
            method=str(optimizer["method"]),
            risk_aversion=float(optimizer["risk_aversion"]),
            max_weight=float(optimizer["max_weight"]),
            long_only=bool(optimizer["long_only"]),
            fully_invested=bool(optimizer["fully_invested"]),
        ),
        rebalancing=RebalancingConfig(
            frequency=str(rebalancing["frequency"]),
        ),
        costs=CostsConfig(
            transaction_cost_bps=float(costs["transaction_cost_bps"]),
            slippage_bps=float(costs["slippage_bps"]),
        ),
        backtest=BacktestConfigFile(
            walk_forward=bool(backtest["walk_forward"]),
            drift_aware=bool(backtest["drift_aware"]),
        ),
        ml=MLConfig(
            enabled=bool(ml["enabled"]),
            model=str(ml["model"]),
            min_train_size=int(ml["min_train_size"]),
            horizon=int(ml["horizon"]),
            alpha=float(ml["alpha"]),
        ),
        benchmarks=BenchmarkConfig(
            equal_weight=bool(benchmarks["equal_weight"]),
            buy_and_hold=bool(benchmarks["buy_and_hold"]),
            market=str(benchmarks["market"]).upper(),
        ),
    )
