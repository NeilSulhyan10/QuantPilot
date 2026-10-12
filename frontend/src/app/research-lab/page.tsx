"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Hero,
  Section,
  Chip,
  Card,
  CardHeader,
  CardTitle,
  CardValue,
  CardNote,
  Table,
  Badge,
  Select,
} from "@/components/ui";
import { Button } from "@/components/ui";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { api } from "@/lib/api";
import { formatPercent, formatNumber, formatSharpe, getCurrencySymbol } from "@/lib/formatters";
import { useMarket } from "@/context/MarketContext";
import type { ComparisonMetrics, MarketConfig } from "@/lib/types";

interface PortfolioBuilderData {
  comparison: Record<string, ComparisonMetrics>;
  returns: { date: string; return: number }[];
  weights: Record<string, number>[];
  turnover: { date: string; turnover: number }[];
  average_weights: Record<string, { average_weight: number }>;
  effective_stocks: { date: string; effective_stocks: number }[];
}

export default function PortfolioBuilderPage() {
  const { market, markets, setMarket } = useMarket();
  const currencySymbol = market ? getCurrencySymbol(market.currency) : "$";
  const [data, setData] = useState<PortfolioBuilderData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      if (!market) return;
      try {
        setLoading(true);
        const [comparison, returns, weights, turnover, averageWeights, effectiveStocks] = await Promise.all([
          api.portfolioBuilder.comparison(market.market),
          api.portfolioBuilder.returns(market.market),
          api.portfolioBuilder.weights(market.market),
          api.portfolioBuilder.turnover(market.market),
          api.portfolioBuilder.averageWeights(market.market),
          api.portfolioBuilder.effectiveStocks(market.market),
        ]);
        setData({
          comparison: comparison as Record<string, ComparisonMetrics>,
          returns: returns as { date: string; return: number }[],
          weights: weights as Record<string, number>[],
          turnover: turnover as { date: string; turnover: number }[],
          average_weights: averageWeights as Record<string, { average_weight: number }>,
          effective_stocks: effectiveStocks as { date: string; effective_stocks: number }[],
        });
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [market]);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={error} />;
  if (!data) return <ErrorState message="No data available" />;

  const qp = data.comparison["QuantPilot V2"];

  const cumulativeReturns = data.returns.map((r) => ({
    date: r.date.split("T")[0],
    value: data.returns
      .slice(0, data.returns.findIndex((x) => x.date === r.date) + 1)
      .reduce((acc, cur) => acc * (1 + cur.return), 1),
  }));

  const drawdownData = cumulativeReturns.map((r) => {
    const peak = Math.max(...cumulativeReturns.slice(0, cumulativeReturns.indexOf(r) + 1).map((x) => x.value));
    return { date: r.date, value: (r.value / peak - 1) * 100 };
  });

  const latestWeights = data.weights[data.weights.length - 1] || {};
  const sortedLatest = Object.entries(latestWeights)
    .filter(([, v]) => v > 1e-8)
    .sort(([, a], [, b]) => b - a);

  const sortedAverage = Object.entries(data.average_weights)
    .sort(([, a], [, b]) => b.average_weight - a.average_weight);

  const avgActivePositions = data.weights.reduce((sum, w) => {
    return sum + Object.values(w).filter((v) => v > 1e-8).length;
  }, 0) / data.weights.length;

  const avgEffectiveStocks = data.effective_stocks.reduce((sum, e) => sum + e.effective_stocks, 0) / data.effective_stocks.length;
  const maxPosition = Math.max(...Object.values(latestWeights));

  return (
    <div className="min-h-screen">
      <Hero
        eyebrow="Portfolio Analytics"
        title="Portfolio intelligence, backed by data."
        subtitle="Walk-forward quantitative research using rolling expected returns, covariance estimation, constrained mean-variance optimization, realistic turnover, transaction costs and benchmark analysis."
      />

      <Section title="QuantPilot V2" subtitle="Core out-of-sample performance">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>CAGR</CardTitle>
              <CardValue>{formatPercent(qp?.cagr)}</CardValue>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Sharpe</CardTitle>
              <CardValue>{formatSharpe(qp?.sharpe_ratio)}</CardValue>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Volatility</CardTitle>
              <CardValue>{formatPercent(qp?.annualized_volatility)}</CardValue>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Max Drawdown</CardTitle>
              <CardValue>{formatPercent(qp?.maximum_drawdown)}</CardValue>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Cumulative Return</CardTitle>
              <CardValue>{formatPercent(qp?.cumulative_return)}</CardValue>
            </CardHeader>
          </Card>
        </div>
      </Section>

      <Section title="Performance Comparison" subtitle="Risk-adjusted comparison across the research universe">
        <Table
          columns={[
            { key: "strategy", header: "Strategy" },
            { key: "cumulative_return", header: "Cumulative Return", render: (r) => formatPercent(r.cumulative_return) },
            { key: "cagr", header: "CAGR", render: (r) => formatPercent(r.cagr) },
            { key: "annualized_volatility", header: "Volatility", render: (r) => formatPercent(r.annualized_volatility) },
            { key: "sharpe_ratio", header: "Sharpe", render: (r) => formatSharpe(r.sharpe_ratio) },
            { key: "sortino_ratio", header: "Sortino", render: (r) => formatSharpe(r.sortino_ratio) },
            { key: "maximum_drawdown", header: "Max Drawdown", render: (r) => formatPercent(r.maximum_drawdown) },
          ]}
          data={Object.entries(data.comparison).map(([strategy, metrics]) => ({ strategy, ...metrics }))}
          keyExtractor={(r) => r.strategy}
        />
      </Section>

      <Section title="Cumulative Portfolio Growth">
        <div className="chart-container h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={cumulativeReturns} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => v.split("-").slice(0, 2).join("-")} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => `${((v - 1) * 100).toFixed(1)}%`} />
              <Tooltip
                formatter={(v: any) => v !== undefined && v !== null ? [`${((Number(v) - 1) * 100).toFixed(2)}%`, "Cumulative"] : ["—", "Cumulative"]}
                contentStyle={{ background: "white", border: "1px solid #dbe4ef", borderRadius: "8px", boxShadow: "var(--shadow-sm)" }}
              />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} dot={false} name="Cumulative" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="QuantPilot Drawdown">
        <div className="chart-container h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={drawdownData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => v.split("-").slice(0, 2).join("-")} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => `${v.toFixed(1)}%`} />
              <Tooltip
                formatter={(v: any) => v !== undefined && v !== null ? [`${Number(v).toFixed(2)}%`, "Drawdown"] : ["—", "Drawdown"]}
                contentStyle={{ background: "white", border: "1px solid #dbe4ef", borderRadius: "8px", boxShadow: "var(--shadow-sm)" }}
              />
              <Area type="monotone" dataKey="value" stroke="#dc2626" fill="#dc2626" fillOpacity={0.15} name="Drawdown" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Portfolio Allocation" subtitle="Current and average capital allocation">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold text-slate-700 mb-3">Latest Weights</h3>
            <Table
              columns={[
                { key: "ticker", header: "Ticker" },
                { key: "weight", header: "Weight", render: (r) => formatPercent(r.weight) },
              ]}
              data={sortedLatest.map(([ticker, weight]) => ({ ticker, weight }))}
              keyExtractor={(r) => r.ticker}
            />
          </div>
          <div>
            <h3 className="font-semibold text-slate-700 mb-3">Average Weights</h3>
            <Table
              columns={[
                { key: "ticker", header: "Ticker" },
                { key: "weight", header: "Avg Weight", render: (r) => formatPercent(r.weight) },
              ]}
              data={sortedAverage.map(([ticker, { average_weight }]) => ({ ticker, weight: average_weight }))}
              keyExtractor={(r) => r.ticker}
            />
          </div>
        </div>
      </Section>

      <Section title="Portfolio Characteristics">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Avg Active Positions</CardTitle>
              <CardValue>{formatNumber(avgActivePositions)}</CardValue>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Avg Effective Stocks</CardTitle>
              <CardValue>{formatNumber(avgEffectiveStocks)}</CardValue>
            </CardHeader>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Maximum Position</CardTitle>
              <CardValue>{formatPercent(maxPosition)}</CardValue>
            </CardHeader>
          </Card>
        </div>
      </Section>

      <Section title="Portfolio Turnover">
        <div className="chart-container h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data.turnover} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => v.split("-").slice(0, 2).join("-")} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Tooltip
                formatter={(v: any) => v !== undefined && v !== null ? [`${(Number(v) * 100).toFixed(2)}%`, "Turnover"] : ["—", "Turnover"]}
                contentStyle={{ background: "white", border: "1px solid #dbe4ef", borderRadius: "8px", boxShadow: "var(--shadow-sm)" }}
              />
              <Line type="monotone" dataKey="turnover" stroke="#2563eb" strokeWidth={2} dot={false} name="Turnover" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Methodology">
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-5 text-slate-700 text-sm leading-relaxed space-y-3">
          <p><strong>Universe</strong> — 20 fixed liquid US equities.</p>
          <p><strong>Expected Returns</strong> — 60-day historical mean returns with 50% cross-sectional shrinkage.</p>
          <p><strong>Risk Model</strong> — 60-day rolling covariance matrix.</p>
          <p><strong>Optimization</strong> — Long-only, fully invested mean-variance optimization with a 10% maximum position constraint.</p>
          <p><strong>Rebalancing</strong> — Monthly.</p>
          <p><strong>Costs</strong> — 10 bps transaction cost + 5 bps slippage.</p>
          <p><strong>Backtesting</strong> — Walk-forward, out-of-sample evaluation with drift-aware portfolio weights.</p>
          <p><strong>Benchmarks</strong> — Equal Weight, Buy & Hold, and SPY.</p>
        </div>
      </Section>

      <Section title="Research Limitations">
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-5 text-slate-700 text-sm leading-relaxed space-y-2">
          <p>• The MVP uses a fixed current-stock universe, creating potential survivorship bias.</p>
          <p>• Historical estimates are based on daily market data.</p>
          <p>• Transaction costs and slippage are modeled assumptions, not live execution measurements.</p>
          <p>• Backtest results are historical and do not guarantee future performance.</p>
          <p>• QuantPilot is not selected solely because it has the highest historical return.</p>
        </div>
      </Section>

      <div className="footer">
        QuantPilot · Portfolio Analytics
      </div>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-500">Loading Portfolio Analytics data...</p>
      </div>
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center p-8 max-w-md mx-auto">
        <div className="w-16 h-16 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-4">
          <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h2 className="text-slate-900 font-display font-bold text-xl mb-2">Failed to Load Data</h2>
        <p className="text-slate-500 text-sm mb-6">{message}</p>
        <Button onClick={() => window.location.reload()}>Retry</Button>
      </div>
    </div>
  );
}