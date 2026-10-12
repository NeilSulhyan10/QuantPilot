import type {
  MarketConfig,
  SubsetInvestmentResult,
  StockAnalysis,
  BacktestResponse,
  PortfolioBuilderData,
  HorizonParseResponse,
  EvaluateRequest,
  BacktestRequest,
  TrendRequest,
} from "@/lib/types";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

async function fetchJson<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

export const api = {
  markets: () => fetchJson<MarketConfig[]>("/api/markets"),

  evaluate: (body: EvaluateRequest) =>
    fetchJson<SubsetInvestmentResult[]>("/api/evaluate", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  backtest: (body: BacktestRequest) =>
    fetchJson<BacktestResponse>("/api/backtest", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  trends: (body: TrendRequest) =>
    fetchJson<StockAnalysis[]>("/api/trend", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  parseHorizon: (horizon: string) =>
    fetchJson<HorizonParseResponse>(`/api/parse-horizon?horizon=${encodeURIComponent(horizon)}`),

  portfolioBuilder: {
    comparison: (market: string = "us") => fetchJson<Record<string, any>>(`/api/portfolio-builder/comparison?market=${market}`),
    returns: (market: string = "us") => fetchJson<{ date: string; return: number }[]>(`/api/portfolio-builder/returns?market=${market}`),
    weights: (market: string = "us") => fetchJson<Record<string, number>[]>(`/api/portfolio-builder/weights?market=${market}`),
    turnover: (market: string = "us") => fetchJson<{ date: string; turnover: number }[]>(`/api/portfolio-builder/turnover?market=${market}`),
    averageWeights: (market: string = "us") => fetchJson<Record<string, { average_weight: number }>>(`/api/portfolio-builder/average-weights?market=${market}`),
    effectiveStocks: (market: string = "us") => fetchJson<{ date: string; effective_stocks: number }[]>(`/api/portfolio-builder/effective-stocks?market=${market}`),
  },
};