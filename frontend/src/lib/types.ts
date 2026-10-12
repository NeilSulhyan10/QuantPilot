export interface MarketConfig {
  market: string;
  name: string;
  currency: string;
  benchmark: string;
  ticker_suffix: string;
}

export interface MultiHorizonReturns {
  "30d": number | null;
  "90d": number | null;
  "180d": number | null;
  "1y": number | null;
}

export interface TechnicalIndicators {
  sma_50: number | null;
  sma_200: number | null;
  price_vs_sma50: "above" | "below" | null;
  price_vs_sma200: "above" | "below" | null;
}

export interface StockAnalysis {
  ticker: string;
  current_price: number;
  multi_horizon: MultiHorizonReturns;
  trend: "bearish" | "neutral" | "bullish";
  trend_score: number;
  technical: TechnicalIndicators;
}

export interface RiskProfileResult {
  risk_profile: string;
  max_allowed_volatility: number;
  max_feasible_return: number;
  recommended_return: number;
  expected_return: number;
  expected_volatility: number;
  sharpe_ratio: number;
  future_value_lump_sum: number;
  future_value_monthly: number;
  feasible: boolean;
  warning: string | null;
  weights: Record<string, number> | null;
  cash_weight: number;
}

export interface SubsetInvestmentResult {
  tickers: string[];
  subset_size: number;
  max_weight: number;
  feasible: boolean;
  error: string | null;
  conservative: RiskProfileResult | null;
  moderate: RiskProfileResult | null;
  aggressive: RiskProfileResult | null;
  best_future_value: number | null;
  extreme_return_warnings: string[] | null;
}

export interface ComparisonMetrics {
  cumulative_return: number;
  cagr: number;
  annualized_volatility: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  maximum_drawdown: number;
}

export interface BacktestResponse {
  goal_returns: Record<string, number>;
  goal_weights: Record<string, number>[];
  equal_weight_returns: Record<string, number>;
  benchmark_returns: Record<string, number>;
  comparison_returns: Record<string, number>[];
  equity_curve: {
    date: string;
    "QuantPilot Portfolio": number;
    "Equal Weight": number;
    Benchmark: number;
  }[];
  metrics: Record<string, ComparisonMetrics>;
}

export interface PortfolioBuilderData {
  comparison: Record<string, ComparisonMetrics>;
  returns: { date: string; return: number }[];
  weights: Record<string, number>[];
  turnover: { date: string; turnover: number }[];
  average_weights: Record<string, { average_weight: number }>;
  effective_stocks: { date: string; effective_stocks: number }[];
}

export interface HorizonParseResponse {
  years: number;
  formatted: string;
}

export interface EvaluateRequest {
  tickers: string[];
  market: string;
  lump_sum: number;
  monthly_sip: number;
  years: number;
  minimum_observations?: number;
  estimation_window?: number;
  shrinkage_alpha?: number;
  min_subset_size?: number;
  max_subset_size?: number;
  max_position_weight?: number;
  ranking_metric?: "low_risk" | "max_future_value";
}

export interface BacktestRequest {
  tickers: string[];
  market: string;
  weights: Record<string, number>;
  cash_weight: number;
  expected_return: number;
  expected_volatility: number;
  target_return: number;
  risk_tolerance: string;
  maximum_allowed_volatility: number;
  estimation_window?: number;
  shrinkage_alpha?: number;
}

export interface TrendRequest {
  tickers: string[];
  market: string;
}
