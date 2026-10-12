"use client";

import { useState, useEffect, useCallback } from "react";
import {
  Hero,
  Section,
  Chip,
  Card,
  CardHeader,
  CardTitle,
  CardValue,
  Table,
  Badge,
  Button,
} from "@/components/ui";
import { Input, Select } from "@/components/ui";
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell } from "recharts";
import { api } from "@/lib/api";
import { formatPercent, formatCurrency, formatNumber, formatSharpe, getCurrencySymbol, getTrendBadge, getVsMABadge } from "@/lib/formatters";
import { useMarket } from "@/context/MarketContext";
import type { MarketConfig, SubsetInvestmentResult, RiskProfileResult, StockAnalysis, BacktestResponse, ComparisonMetrics } from "@/lib/types";

interface BacktestRun {
  portfolioKey: string;
  profileLabel: string;
  data: BacktestResponse;
}

const ESTIMATION_WINDOWS = [
  { value: 60, label: "60 days (~2 months)" },
  { value: 120, label: "120 days (~4 months)" },
  { value: 252, label: "252 days (~1 year)" },
  { value: 504, label: "504 days (~2 years)" },
  { value: 756, label: "756 days (~3 years)" },
];

const RANKING_METRICS = [
  { value: "max_future_value", label: "Highest Future Value / Expected Return" },
  { value: "low_risk", label: "Low Risk (lowest volatility)" },
];
const POSITION_WEIGHT_OPTIONS = [20, 25, 30, 35, 40, 50, 75, 100];

const DEFAULT_US_TICKERS = "AAPL, MSFT, NVDA, AVGO, GOOGL, AMZN, META, JPM, V, MA";
const DEFAULT_INDIA_TICKERS = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, SBIN, ITC, LT, HINDUNILVR, MARUTI";

function formatInvestmentAmount(amount: number, marketName: string | undefined): string {
  return new Intl.NumberFormat(marketName === "india" ? "en-IN" : "en-US", {
    maximumFractionDigits: 0,
  }).format(amount);
}

export default function ResearchLabPage() {
  const { market, markets, setMarket, currencySymbol } = useMarket();
  const [investmentMode, setInvestmentMode] = useState<"lump_sum" | "sip">("lump_sum");
  const [investmentAmount, setInvestmentAmount] = useState(1_000_000);
  const [investmentAmountInput, setInvestmentAmountInput] = useState("1000000");
  const [editingInvestmentAmount, setEditingInvestmentAmount] = useState(false);
  const [monthlySip, setMonthlySip] = useState(10_000);
  const [horizon, setHorizon] = useState("10 years");
  const [horizonYears, setHorizonYears] = useState<number | null>(null);
  const [horizonError, setHorizonError] = useState<string | null>(null);
  const [estimationWindow, setEstimationWindow] = useState(252);
  const [tickersInput, setTickersInput] = useState(DEFAULT_US_TICKERS);
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [tickerError, setTickerError] = useState<string | null>(null);
  const [companySelectionMode, setCompanySelectionMode] = useState<"all" | "group">("all");
  const [groupSize, setGroupSize] = useState(5);
  const [maxPositionWeight, setMaxPositionWeight] = useState(0.25);
  const [rankingMetric, setRankingMetric] = useState<"low_risk" | "max_future_value">("max_future_value");
  const [building, setBuilding] = useState(false);
  const [results, setResults] = useState<SubsetInvestmentResult[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trends, setTrends] = useState<StockAnalysis[] | null>(null);
  const [trendsLoading, setTrendsLoading] = useState(false);
  const [backtestData, setBacktestData] = useState<BacktestRun | null>(null);
  const [backtestLoading, setBacktestLoading] = useState(false);

  // Update tickers when market changes
  useEffect(() => {
    if (market) {
      setTickersInput(market.market === "us" ? DEFAULT_US_TICKERS : DEFAULT_INDIA_TICKERS);
    }
  }, [market]);

  // Parse horizon
  const parseHorizon = useCallback(async () => {
    if (!horizon.trim()) {
      setHorizonYears(null);
      setHorizonError("Enter a horizon");
      return;
    }
    try {
      const res = await api.parseHorizon(horizon);
      setHorizonYears(res.years);
      setHorizonError(null);
    } catch (e) {
      setHorizonYears(null);
      setHorizonError(e instanceof Error ? e.message : "Invalid horizon");
    }
  }, [horizon]);

  useEffect(() => {
    const timer = setTimeout(parseHorizon, 500);
    return () => clearTimeout(timer);
  }, [parseHorizon]);

  // Normalize tickers
  useEffect(() => {
    if (!market) return;
    const raw = tickersInput.split(",").map(t => t.trim().toUpperCase()).filter(Boolean);
    try {
      const normalized = raw.map(t => {
        if (market.market === "india" && !t.startsWith("^") && !t.endsWith(".NS")) return `${t}.NS`;
        return t.replace(/\.US$/, "");
      });
      const unique = [...new Set(normalized)];
      setSelectedTickers(unique);
      setTickerError(unique.length < 3 ? `Need at least 3 companies (have ${unique.length})` : 
        unique.length > 15 ? `Max 15 companies (have ${unique.length})` : null);
    } catch {
      setSelectedTickers([]);
      setTickerError("Invalid tickers");
    }
  }, [tickersInput, market]);

  const canBuild = selectedTickers.length >= 3 && selectedTickers.length <= 15 && horizonYears !== null && horizonYears > 0 && !building;
  const selectedGroupSize = Math.min(Math.max(3, groupSize), selectedTickers.length);
  const subsetSize = companySelectionMode === "all" ? selectedTickers.length : selectedGroupSize;
  const effectivePositionCap = Math.max(maxPositionWeight, 1 / Math.max(1, subsetSize));

  const handleBuild = async () => {
    if (!canBuild || !market) return;
    setBuilding(true);
    setError(null);
    setResults(null);
    setTrends(null);
    setBacktestData(null);
    try {
      const res = await api.evaluate({
        tickers: selectedTickers,
        market: market.market,
        lump_sum: investmentMode === "lump_sum" ? investmentAmount : 0,
        monthly_sip: investmentMode === "sip" ? monthlySip : 0,
        years: horizonYears!,
        estimation_window: estimationWindow,
        shrinkage_alpha: 0.5,
        min_subset_size: subsetSize,
        max_subset_size: subsetSize,
        max_position_weight: maxPositionWeight,
        ranking_metric: rankingMetric,
      });
      setResults(res);
      // Fetch trends for all selected tickers
      setTrendsLoading(true);
      try {
        const trendRes = await api.trends({ tickers: selectedTickers, market: market.market });
        setTrends(trendRes.filter(t => !("error" in t)) as StockAnalysis[]);
      } catch {
        setTrends([]);
      } finally {
        setTrendsLoading(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Evaluation failed");
    } finally {
      setBuilding(false);
    }
  };

  const runBacktest = async (weights: Record<string, number>, profile: RiskProfileResult, portfolioKey: string) => {
    if (!market) return;
    setBacktestLoading(true);
    try {
      const res = await api.backtest({
        tickers: Object.keys(weights),
        market: market.market,
        weights,
        cash_weight: profile.cash_weight,
        expected_return: profile.expected_return,
        expected_volatility: profile.expected_volatility,
        target_return: profile.recommended_return,
        risk_tolerance: profile.risk_profile,
        maximum_allowed_volatility: profile.max_allowed_volatility,
        estimation_window: estimationWindow,
        shrinkage_alpha: 0.5,
      });
      const profileLabel = profile.risk_profile.charAt(0).toUpperCase() + profile.risk_profile.slice(1);
      setBacktestData({ data: res, portfolioKey, profileLabel });
    } catch (e) {
      console.error("Backtest failed:", e);
    } finally {
      setBacktestLoading(false);
    }
  };

  return (
    <div className="min-h-screen">
      <Hero
        eyebrow="Research Lab"
        title="Plan your portfolio from an investment amount."
        subtitle="Choose a market, enter your investment amount and horizon, select companies, and QuantPilot finds the top portfolio combinations with risk/return projections."
      />

      <Section title="01 · Choose Market" subtitle="Market selection determines ticker conventions, currency and benchmark">
        <Select
          label="Market"
          value={market?.market || ""}
          onChange={(e) => {
            const m = markets.find(x => x.market === e.target.value);
            if (m) setMarket(m);
          }}
          options={markets.map(m => ({ value: m.market, label: m.name }))}
          placeholder="Select market"
        />
        {market && (
          <div className="mt-3 flex flex-wrap gap-2">
            <Chip label={market.name} variant="blue" />
            <Chip label={market.currency} variant="default" />
            <Chip label={`Benchmark: ${market.benchmark}`} variant="purple" />
          </div>
        )}
      </Section>

      <Section title="02 · Investment Plan" subtitle="How much and how long">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-slate-600 text-sm font-medium mb-2">Investment type</label>
            <div className="flex gap-3">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="investmentMode"
                  value="lump_sum"
                  checked={investmentMode === "lump_sum"}
                  onChange={() => setInvestmentMode("lump_sum")}
                  className="w-4 h-4 text-blue-600 border-slate-300 focus:ring-blue-500"
                />
                <span className="text-slate-700">Lump Sum</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  name="investmentMode"
                  value="sip"
                  checked={investmentMode === "sip"}
                  onChange={() => setInvestmentMode("sip")}
                  className="w-4 h-4 text-blue-600 border-slate-300 focus:ring-blue-500"
                />
                <span className="text-slate-700">Monthly SIP</span>
              </label>
            </div>
          </div>
          <div>
            {investmentMode === "lump_sum" ? (
              <Input
                label={`Initial Investment (${currencySymbol})`}
                type="text"
                inputMode="numeric"
                pattern="[0-9,]*"
                value={editingInvestmentAmount ? investmentAmountInput : formatInvestmentAmount(investmentAmount, market?.market)}
                onFocus={() => {
                  setInvestmentAmountInput(investmentAmount > 0 ? String(investmentAmount) : "");
                  setEditingInvestmentAmount(true);
                }}
                onChange={(e) => {
                  const digits = e.target.value.replace(/\D/g, "");
                  setInvestmentAmountInput(digits);
                  setInvestmentAmount(digits ? Number(digits) : 0);
                }}
                onBlur={() => setEditingInvestmentAmount(false)}
                helperText="Amount to invest upfront. Minimum 100."
              />
            ) : (
              <Input
                label={`Monthly SIP (${currencySymbol})`}
                type="number"
                min="100"
                step="100"
                value={monthlySip}
                onChange={(e) => setMonthlySip(Number(e.target.value))}
                helperText="Amount to invest each month. Minimum 100."
              />
            )}
          </div>
        </div>
        <Input
          label="Investment Horizon"
          value={horizon}
          onChange={(e) => setHorizon(e.target.value)}
          placeholder="e.g., 10 years, 5 years 6 months, 18 months"
          error={horizonError || undefined}
          helperText={horizonYears ? `Parsed as: ${horizonYears.toFixed(2)} years` : "Examples: '10 years', '5 years 6 months', '18 months', '2.5 years', '1y 6m', '6m'"}
        />
      </Section>

      <Section title="03 · Estimation Window" subtitle="Historical lookback for expected return estimation">
        <Select
          label="Estimation window"
          value={estimationWindow}
          onChange={(e) => setEstimationWindow(Number(e.target.value))}
          options={ESTIMATION_WINDOWS.map(w => ({ value: String(w.value), label: w.label }))}
          placeholder="Select estimation window"
        />
      </Section>

      <Section title="04 · Select Companies" subtitle="Enter the companies you want QuantPilot to consider">
        <Input
          label="Ticker symbols"
          value={tickersInput}
          onChange={(e) => setTickersInput(e.target.value)}
          placeholder="AAPL, MSFT, NVDA..."
          error={tickerError || undefined}
          helperText="Enter Yahoo Finance ticker symbols separated by commas. Indian symbols can be entered without the .NS suffix. Minimum 3, maximum 15 companies."
        />
        {selectedTickers.length > 0 && (
          <p className="mt-2 text-slate-500 text-sm">Normalized symbols: {selectedTickers.join(", ")}</p>
        )}
      </Section>

      <Section title="05 · Analysis Scope" subtitle="Analyze your full list or compare groups of a chosen size">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <label className={`flex gap-3 p-4 rounded-xl border cursor-pointer transition-colors ${companySelectionMode === "all" ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-white hover:border-slate-300"}`}>
            <input
              type="radio"
              name="companySelectionMode"
              value="all"
              checked={companySelectionMode === "all"}
              onChange={() => setCompanySelectionMode("all")}
              className="mt-1 w-4 h-4 text-blue-600 border-slate-300 focus:ring-blue-500"
            />
            <span>
              <span className="block font-semibold text-slate-800">Analyze all companies</span>
              <span className="block mt-1 text-sm text-slate-500">Build and review a portfolio using your entire list ({selectedTickers.length} companies).</span>
            </span>
          </label>
          <label className={`flex gap-3 p-4 rounded-xl border cursor-pointer transition-colors ${companySelectionMode === "group" ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-white hover:border-slate-300"}`}>
            <input
              type="radio"
              name="companySelectionMode"
              value="group"
              checked={companySelectionMode === "group"}
              onChange={() => setCompanySelectionMode("group")}
              className="mt-1 w-4 h-4 text-blue-600 border-slate-300 focus:ring-blue-500"
            />
            <span>
              <span className="block font-semibold text-slate-800">Choose a group size</span>
              <span className="block mt-1 text-sm text-slate-500">Compare groups from your list and find the strongest candidates for your ranking goal.</span>
            </span>
          </label>
        </div>
        {companySelectionMode === "group" && (
          <div className="mt-4 max-w-sm">
            <Select
              label="Companies in each group"
              value={selectedGroupSize}
              onChange={(e) => setGroupSize(Number(e.target.value))}
              options={Array.from({ length: Math.max(0, selectedTickers.length - 2) }, (_, index) => index + 3).map(size => ({
                value: String(size),
                label: `${size} companies`,
              }))}
              helperText={`Each result will contain exactly ${selectedGroupSize} companies.`}
            />
          </div>
        )}
        <div className="mt-4 max-w-sm">
          <Select
            label="Maximum weight per company"
            value={Math.round(maxPositionWeight * 100)}
            onChange={(e) => setMaxPositionWeight(Number(e.target.value) / 100)}
            options={POSITION_WEIGHT_OPTIONS.map(weight => ({
              value: String(weight),
              label: `${weight}%`,
            }))}
            helperText={`The optimizer may assign up to ${Math.round(effectivePositionCap * 100)}% to one company. The cap is raised only when needed to make the selected group feasible.`}
          />
        </div>
      </Section>

      <Section title="06 · Ranking Method" subtitle="How to rank the portfolio options">
        <Select
          label="Rank portfolios by"
          value={rankingMetric}
          onChange={(e) => {
            const value = e.target.value;
            if (value === "low_risk" || value === "max_future_value") setRankingMetric(value);
          }}
          options={RANKING_METRICS}
          placeholder="Select ranking method"
        />
      </Section>

      <div className="mt-8">
        <Button
          size="lg"
          className="w-full sm:w-auto ml-8"
          onClick={handleBuild}
          disabled={!canBuild}
          loading={building}
        >
          {companySelectionMode === "all" ? "Analyze Full Company List" : `Compare ${selectedGroupSize}-Company Groups`}
        </Button>
      </div>

      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700" role="alert">
          ⚠️ {error}
        </div>
      )}

      {results && results.length > 0 && (
        <ResultsSection
          results={results}
          analysisMode={companySelectionMode}
          companyCount={selectedTickers.length}
          groupSize={selectedGroupSize}
          rankingMetric={rankingMetric}
          currencySymbol={currencySymbol}
          investmentAmount={investmentMode === "lump_sum" ? investmentAmount : 0}
          monthlySip={investmentMode === "sip" ? monthlySip : 0}
          trends={trends}
          trendsLoading={trendsLoading}
          backtestData={backtestData}
          backtestLoading={backtestLoading}
          onRunBacktest={runBacktest}
          marketConfig={market!}
          estimationWindow={estimationWindow}
        />
      )}

      <div className="footer mt-12">
        QuantPilot · Research Lab · Model-based planning only — not financial advice
      </div>
    </div>
  );
}

interface ResultsSectionProps {
  results: SubsetInvestmentResult[];
  analysisMode: "all" | "group";
  companyCount: number;
  groupSize: number;
  rankingMetric: "low_risk" | "max_future_value";
  currencySymbol: string;
  investmentAmount: number;
  monthlySip: number;
  trends: StockAnalysis[] | null;
  trendsLoading: boolean;
  backtestData: BacktestRun | null;
  backtestLoading: boolean;
  onRunBacktest: (weights: Record<string, number>, profile: RiskProfileResult, portfolioKey: string) => void;
  marketConfig: MarketConfig;
  estimationWindow: number;
}

function ResultsSection({
  results,
  analysisMode,
  companyCount,
  groupSize,
  rankingMetric,
  currencySymbol,
  investmentAmount,
  monthlySip,
  trends,
  trendsLoading,
  backtestData,
  backtestLoading,
  onRunBacktest,
  marketConfig,
  estimationWindow }: ResultsSectionProps) {
  const feasibleResults = results.filter(r => r.feasible);
  const infeasibleResults = results.filter(r => !r.feasible);
  const top3 = feasibleResults.slice(0, 3);
  const rankingLabel = rankingMetric === "low_risk" ? "lowest volatility" : "projected future value";

  return (
    <div>
      <div className={`mb-6 p-4 rounded-xl border ${feasibleResults.length > 0 ? "bg-green-50 border-green-200 text-green-700" : "bg-amber-50 border-amber-200 text-amber-800"}`}>
        {analysisMode === "all"
          ? `Full-list analysis · ${companyCount} companies · ${feasibleResults.length ? "Portfolio is feasible under the selected risk profiles" : "No selected risk profile is feasible"}`
          : `Evaluated ${results.length} groups of ${groupSize} from ${companyCount} companies · ${feasibleResults.length} feasible · Showing up to 3 by ${rankingLabel}`}
      </div>

      <Section
        title={analysisMode === "all" ? "Full-list Portfolio" : "Top Portfolio Groups"}
        subtitle={analysisMode === "all" ? "Risk profiles and optimized allocation across all entered companies" : `Ranked by ${rankingLabel}`}
      >
        <div className="space-y-4">
          {top3.map((result, rank) => (
            <PortfolioResultCard
              key={`${result.tickers.join("-")}-${result.subset_size}`}
              rank={rank + 1}
              result={result}
              currencySymbol={currencySymbol}
              investmentAmount={investmentAmount}
              monthlySip={monthlySip}
              onRunBacktest={onRunBacktest}
              backtestLoading={backtestLoading}
              backtestData={backtestData?.portfolioKey === result.tickers.join("|") ? backtestData : null}
              marketConfig={marketConfig}
              estimationWindow={estimationWindow}
            />
          ))}
        </div>
      </Section>

      {infeasibleResults.length > 0 && (
        <Section title={`Infeasible Subsets (${infeasibleResults.length})`} subtitle="Subsets that could not meet risk/return constraints">
          <details className="group">
            <summary className="flex items-center gap-2 cursor-pointer font-semibold text-slate-700 p-3 bg-white border border-slate-200 rounded-xl">
              <span className="text-slate-500">Show {infeasibleResults.length} infeasible subsets</span>
            </summary>
            <div className="mt-3 space-y-2 max-h-64 overflow-y-auto">
              {infeasibleResults.slice(0, 20).map((r, i) => (
                <div key={i} className="p-3 bg-slate-50 border border-slate-200 rounded-lg text-sm">
                  <strong>{r.tickers.join(" + ")}</strong>
                  <span className="text-slate-500 ml-2">({r.subset_size} companies, cap {(r.max_weight * 100).toFixed(0)}%)</span>
                  <span className="text-red-600 ml-2">— {r.error || "Infeasible under risk/return constraints"}</span>
                </div>
              ))}
            </div>
          </details>
        </Section>
      )}

      {trends !== null && (
        <Section title="Stock Trend Analysis" subtitle="Multi-horizon returns, technicals, and trend classification">
          {trendsLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <span className="ml-3 text-slate-500">Analyzing stock trends...</span>
            </div>
          ) : trends && trends.length > 0 ? (
            <TrendTable trends={trends} currencySymbol={currencySymbol} />
          ) : (
            <p className="text-slate-500 text-center py-4">No trend data available</p>
          )}
        </Section>
      )}

    </div>
  );
}

function PortfolioResultCard({
  rank,
  result,
  currencySymbol,
  investmentAmount,
  monthlySip,
  onRunBacktest,
  backtestLoading,
  backtestData,
  marketConfig,
  estimationWindow,
}: {
  rank: number;
  result: SubsetInvestmentResult;
  currencySymbol: string;
  investmentAmount: number;
  monthlySip: number;
  onRunBacktest: (weights: Record<string, number>, profile: RiskProfileResult, portfolioKey: string) => void;
  backtestLoading: boolean;
  backtestData: BacktestRun | null;
  marketConfig: MarketConfig;
  estimationWindow: number;
}) {
  const profiles = [
    { name: "Conservative", data: result.conservative },
    { name: "Moderate", data: result.moderate },
    { name: "Aggressive", data: result.aggressive },
  ].filter(p => p.data) as { name: string; data: RiskProfileResult }[];

  const [expanded, setExpanded] = useState(rank === 1);
  const allocationChoice = [
    { label: "Moderate", profile: result.moderate },
    { label: "Conservative", profile: result.conservative },
    { label: "Aggressive", profile: result.aggressive },
  ].find(({ profile }) => profile?.feasible && profile.weights);
  const backtestChoice = [
    { label: "Moderate", profile: result.moderate },
    { label: "Conservative", profile: result.conservative },
    { label: "Aggressive", profile: result.aggressive },
  ].find(({ profile }) => profile?.feasible && profile.weights);

  return (
    <Card className="overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-4">
          <span className="w-8 h-8 rounded-lg bg-blue-100 text-blue-700 flex items-center justify-center font-bold font-display">#{rank}</span>
          <div>
            <p className="font-semibold text-slate-900">{result.tickers.join(" + ")}</p>
            <p className="text-slate-500 text-sm">
              {result.subset_size} companies · Cap: {(result.max_weight * 100).toFixed(0)}% · Best FV: {formatCurrency(result.best_future_value || 0, currencySymbol)}
            </p>
          </div>
        </div>
        <svg className={`w-5 h-5 text-slate-400 transition-transform ${expanded ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
      </button>

      {expanded && (
        <div className="border-t border-slate-200 p-4 space-y-6">
          {/* Risk Profile Comparison */}
          <h4 className="font-semibold text-slate-700">Risk Profile Comparison</h4>
          <Table
            columns={[
              { key: "Profile", header: "Profile" },
              { key: "Max Vol", header: "Max Vol", render: (r) => r["Max Vol"] },
              { key: "Rec. Return", header: "Rec. Return", render: (r) => r["Rec. Return"] },
              { key: "Exp. Return", header: "Exp. Return", render: (r) => r["Exp. Return"] },
              { key: "Exp. Vol", header: "Exp. Vol", render: (r) => r["Exp. Vol"] },
              { key: "Sharpe", header: "Sharpe", render: (r) => r["Sharpe"] },
              { key: "Future Value (Lump)", header: "Future Value (Lump)", render: (r) => r["Future Value (Lump)"] },
              { key: "Future Value (SIP)", header: "Future Value (SIP)", render: (r) => r["Future Value (SIP)"] },
              { key: "Total Future Value", header: "Total Future Value", render: (r) => r["Total Future Value"] },
            ]}
            data={profiles.map(p => {
              const d = p.data;
              const fvLump = investmentAmount > 0 ? formatCurrency(d.future_value_lump_sum, currencySymbol) : "—";
              const fvSip = monthlySip > 0 ? formatCurrency(d.future_value_monthly, currencySymbol) : "—";
              const totalFV = formatCurrency(d.future_value_lump_sum + d.future_value_monthly, currencySymbol);
              return {
                Profile: p.name,
                "Max Vol": formatPercent(d.max_allowed_volatility),
                "Rec. Return": formatPercent(d.recommended_return),
                "Exp. Return": formatPercent(d.expected_return),
                "Exp. Vol": formatPercent(d.expected_volatility),
                "Sharpe": formatSharpe(d.sharpe_ratio),
                "Future Value (Lump)": fvLump,
                "Future Value (SIP)": fvSip,
                "Total Future Value": totalFV,
              };
            })}
            keyExtractor={(r) => r.Profile}
          />

          {/* Show an allocation whenever at least one risk profile is feasible. */}
          {allocationChoice?.profile?.weights && (
            <div>
              <h4 className="font-semibold text-slate-700 mb-3">Allocation ({allocationChoice.label} Profile)</h4>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={Object.entries(allocationChoice.profile.weights).sort(([, a], [, b]) => b - a).map(([ticker, weight]) => ({ ticker, weight }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="ticker" tick={{ fill: "#64748b", fontSize: 11 }} />
                      <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
                      <Tooltip formatter={(v: any) => v !== undefined && v !== null ? [`${(Number(v) * 100).toFixed(2)}%`, "Weight"] : ["—", "Weight"]} contentStyle={{ background: "white", border: "1px solid #dbe4ef", borderRadius: "8px" }} />
                      <Bar dataKey="weight" fill="#2563eb" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <Table
                  columns={[
                    { key: "Ticker", header: "Ticker" },
                    { key: "Weight", header: "Weight", render: (r) => formatPercent(r.Weight) },
                    { key: "Amount", header: `Amount (${currencySymbol})`, render: (r) => formatCurrency(r.Amount, currencySymbol) },
                  ]}
                  data={Object.entries(allocationChoice.profile.weights).sort(([, a], [, b]) => b - a).map(([ticker, weight]) => ({
                    Ticker: ticker,
                    Weight: weight,
                    Amount: weight * (investmentAmount > 0 ? investmentAmount : monthlySip * 12),
                  }))}
                  keyExtractor={(r) => r.Ticker}
                />
              </div>
            </div>
          )}

          {/* Warnings */}
          {profiles.some(p => p.data.warning) && (
            <div className="space-y-2">
              {profiles.map(p => p.data.warning && (
                <div key={p.name} className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-800 text-sm">
                  ⚠️ {p.name}: {p.data.warning}
                </div>
              ))}
            </div>
          )}

          {backtestChoice?.profile?.weights && (
            <Button
              variant="secondary"
              onClick={() => onRunBacktest(backtestChoice.profile!.weights!, backtestChoice.profile!, result.tickers.join("|"))}
              loading={backtestLoading}
              className="w-full sm:w-auto"
            >
              Run Historical Backtest ({backtestChoice.label} Profile)
            </Button>
          )}
          {backtestData && (
            <BacktestSection
              data={backtestData.data}
              profileLabel={backtestData.profileLabel}
              marketConfig={marketConfig}
              currencySymbol={currencySymbol}
              estimationWindow={estimationWindow}
            />
          )}
        </div>
      )}
    </Card>
  );
}

function TrendTable({ trends, currencySymbol }: { trends: StockAnalysis[]; currencySymbol: string }) {
  const bullishCount = trends.filter(t => t.trend === "bullish").length;
  const bearishCount = trends.filter(t => t.trend === "bearish").length;
  const neutralCount = trends.filter(t => t.trend === "neutral").length;

  return (
    <div>
      <Table
        columns={[
          { key: "Ticker", header: "Ticker" },
          { key: "Price", header: "Price", render: (r) => formatCurrency(r.Price, currencySymbol, 2) },
          { key: "Trend", header: "Trend", render: (r) => {
            const badge = getTrendBadge(r.Trend as "bearish" | "neutral" | "bullish");
            return <Badge variant={r.Trend === "bullish" ? "success" : r.Trend === "bearish" ? "error" : "warning"}>{badge.label}</Badge>;
          }},
          { key: "Score", header: "Score", render: (r) => r.Score },
          { key: "30D", header: "30D", render: (r) => r["30D"] },
          { key: "90D", header: "90D", render: (r) => r["90D"] },
          { key: "180D", header: "180D", render: (r) => r["180D"] },
          { key: "1Y", header: "1Y", render: (r) => r["1Y"] },
          { key: "SMA50", header: "SMA 50", render: (r) => formatCurrency(r.SMA50, currencySymbol, 2) },
          { key: "SMA200", header: "SMA 200", render: (r) => formatCurrency(r.SMA200, currencySymbol, 2) },
          { key: "vs 50", header: "vs 50", render: (r) => {
            const badge = getVsMABadge(r["vs 50"] as "above" | "below" | null);
            return <span className={badge.color}>{badge.label}</span>;
          }},
          { key: "vs 200", header: "vs 200", render: (r) => {
            const badge = getVsMABadge(r["vs 200"] as "above" | "below" | null);
            return <span className={badge.color}>{badge.label}</span>;
          }},
        ]}
        data={trends.map(t => ({
          Ticker: t.ticker,
          Price: t.current_price,
          Trend: t.trend,
          Score: t.trend_score,
          "30D": formatPercent(t.multi_horizon["30d"]),
          "90D": formatPercent(t.multi_horizon["90d"]),
          "180D": formatPercent(t.multi_horizon["180d"]),
          "1Y": formatPercent(t.multi_horizon["1y"]),
          SMA50: t.technical.sma_50,
          SMA200: t.technical.sma_200,
          "vs 50": t.technical.price_vs_sma50,
          "vs 200": t.technical.price_vs_sma200,
        }))}
        keyExtractor={(r) => r.Ticker}
      />

      <div className="grid grid-cols-3 gap-4 mt-4">
        <Card>
          <CardHeader>
            <CardTitle>Bullish</CardTitle>
            <CardValue>{bullishCount}</CardValue>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Neutral</CardTitle>
            <CardValue>{neutralCount}</CardValue>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Bearish</CardTitle>
            <CardValue>{bearishCount}</CardValue>
          </CardHeader>
        </Card>
      </div>
    </div>
  );
}

function BacktestSection({
  data,
  profileLabel,
  marketConfig,
  currencySymbol,
  estimationWindow,
}: {
  data: BacktestResponse;
  profileLabel: string;
  marketConfig: MarketConfig;
  currencySymbol: string;
  estimationWindow: number;
}) {
  const equityData = (Array.isArray(data.equity_curve) ? data.equity_curve : [])
    .flatMap((point) => {
      if (typeof point.date !== "string" || !point.date) return [];
      return [{
        date: point.date.split("T")[0],
        quantpilot: point["QuantPilot Portfolio"],
        equalWeight: point["Equal Weight"],
        benchmark: point["Benchmark"],
      }];
    });

  const drawdownData = equityData.map((point: any) => {
    const peak = Math.max(...equityData.slice(0, equityData.indexOf(point) + 1).map((p: any) => Math.max(p.quantpilot, p.equalWeight, p.benchmark)));
    return {
      date: point.date,
      quantpilot: (point.quantpilot / peak - 1) * 100,
      equalWeight: (point.equalWeight / peak - 1) * 100,
      benchmark: (point.benchmark / peak - 1) * 100,
    };
  });

  const metrics = data.metrics;
  const comparisonColumns = [
    { key: "QuantPilot Portfolio", label: `QuantPilot (${profileLabel})` },
    { key: "Equal Weight", label: "Equal Weight" },
    { key: "Benchmark", label: marketConfig.benchmark },
  ] as const;
  const metricRows: { label: string; key: keyof ComparisonMetrics; format: (value: number) => string; lowerIsBetter?: boolean }[] = [
    { label: "CAGR", key: "cagr", format: formatPercent },
    { label: "Annualized Volatility", key: "annualized_volatility", format: formatPercent, lowerIsBetter: true },
    { label: "Sharpe Ratio", key: "sharpe_ratio", format: formatSharpe },
    { label: "Sortino Ratio", key: "sortino_ratio", format: formatSharpe },
    { label: "Maximum Drawdown", key: "maximum_drawdown", format: formatPercent },
  ];

  return (
    <Section title="Historical Performance Comparison" subtitle="QuantPilot and Equal Weight both rebalance monthly; the benchmark is shown for context.">
      <div className="overflow-x-auto mb-6">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-slate-50 border-b border-slate-200">
              <th className="px-4 py-3 text-left font-semibold text-slate-600 uppercase tracking-wider text-xs">Metric</th>
              {comparisonColumns.map(column => (
                <th key={column.key} className="px-4 py-3 text-left font-semibold text-slate-600 uppercase tracking-wider text-xs">{column.label}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {metricRows.map(row => {
              const values = comparisonColumns.map(column => metrics?.[column.key]?.[row.key]);
              const finiteValues = values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));
              const bestValue = finiteValues.length
                ? (row.lowerIsBetter ? Math.min(...finiteValues) : Math.max(...finiteValues))
                : null;
              return (
                <tr key={row.key}>
                  <td className="px-4 py-3 text-slate-700">{row.label}</td>
                  {comparisonColumns.map((column, index) => {
                    const value = values[index];
                    const isWinner = typeof value === "number" && Number.isFinite(value) && bestValue !== null && Math.abs(value - bestValue) <= 1e-12;
                    return (
                      <td key={column.key} className={`px-4 py-3 text-slate-700${isWinner ? " font-bold text-slate-950" : ""}`}>
                        {typeof value === "number" && Number.isFinite(value) ? row.format(value) : "—"}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="chart-container h-[380px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={equityData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => v.split("-").slice(0, 2).join("-")} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => `${((v - 1) * 100).toFixed(1)}%`} />
              <Tooltip
                formatter={(v: any) => v !== undefined && v !== null ? [`${((Number(v) - 1) * 100).toFixed(2)}%`, "Growth"] : ["—", "Growth"]}
                contentStyle={{ background: "white", border: "1px solid #dbe4ef", borderRadius: "8px", boxShadow: "var(--shadow-sm)" }}
              />
              <Legend />
              <Line type="monotone" dataKey="quantpilot" stroke="#2563eb" strokeWidth={2} dot={false} name="QuantPilot" />
              <Line type="monotone" dataKey="equalWeight" stroke="#16a34a" strokeWidth={2} dot={false} name="Equal Weight" />
              <Line type="monotone" dataKey="benchmark" stroke="#dc2626" strokeWidth={2} dot={false} name={marketConfig.benchmark} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container h-[380px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={drawdownData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => v.split("-").slice(0, 2).join("-")} />
              <YAxis tick={{ fill: "#64748b", fontSize: 11 }} tickFormatter={(v) => `${v.toFixed(1)}%`} />
              <Tooltip
                formatter={(v: any) => v !== undefined && v !== null ? [`${Number(v).toFixed(2)}%`, "Drawdown"] : ["—", "Drawdown"]}
                contentStyle={{ background: "white", border: "1px solid #dbe4ef", borderRadius: "8px", boxShadow: "var(--shadow-sm)" }}
              />
              <Area type="monotone" dataKey="quantpilot" stroke="#2563eb" fill="#2563eb" fillOpacity={0.15} name="QuantPilot" />
              <Area type="monotone" dataKey="equalWeight" stroke="#16a34a" fill="#16a34a" fillOpacity={0.1} name="Equal Weight" />
              <Area type="monotone" dataKey="benchmark" stroke="#dc2626" fill="#dc2626" fillOpacity={0.1} name={marketConfig.benchmark} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <Section title="Model Notes">
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-5 text-slate-700 text-sm leading-relaxed space-y-2">
          <p><strong>Market:</strong> {marketConfig.name} ({marketConfig.currency}).</p>
          <p><strong>Benchmark:</strong> {marketConfig.benchmark}.</p>
          <p><strong>Estimation window:</strong> latest {estimationWindow} trading days.</p>
          <p><strong>Position constraint:</strong> maximum 1/N per company.</p>
          <p><strong>Portfolio:</strong> long-only and fully invested.</p>
          <p><strong>Historical comparison:</strong> best subset vs Equal Weight and the market-specific benchmark.</p>
        </div>
      </Section>
    </Section>
  );
}
