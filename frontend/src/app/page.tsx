"use client";

import { useState } from "react";
import Link from "next/link";
import { Hero, Section, Chip } from "@/components/ui";
import { Button } from "@/components/ui";

export default function Home() {
  const [mode, setMode] = useState<"portfolio-analytics" | "research-lab">("portfolio-analytics");

  return (
    <div className="min-h-screen flex flex-col">
      {/* Mode Selector */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="flex items-center gap-3 p-2 bg-white border border-slate-200 rounded-xl shadow-sm w-fit">
          <button
            onClick={() => setMode("portfolio-analytics")}
            className={`px-5 py-2 rounded-lg font-semibold text-sm transition-all ${
              mode === "portfolio-analytics"
                ? "bg-blue-50 text-blue-700 shadow-sm border border-blue-200"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
Portfolio Analytics
          </button>
          <button
            onClick={() => setMode("research-lab")}
            className={`px-5 py-2 rounded-lg font-semibold text-sm transition-all ${
              mode === "research-lab"
                ? "bg-blue-50 text-blue-700 shadow-sm border border-blue-200"
                : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
            }`}
          >
            Research Lab
          </button>
        </div>
      </div>

      {/* Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
        {mode === "portfolio-analytics" && (
          <Link href="/portfolio-analytics">
            <PortfolioBuilderPreview />
          </Link>
        )}
        {mode === "research-lab" && (
          <Link href="/research-lab">
            <ResearchLabPreview />
          </Link>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        QuantPilot · Portfolio Analytics · Model-based planning only — not financial advice
      </footer>
    </div>
  );
}

function PortfolioBuilderPreview() {
  return (
    <div>
      <Hero
        eyebrow="Portfolio Terminal"
        title="Plan your portfolio from an investment amount."
        subtitle="Choose a market, enter your investment amount and horizon, select companies, and QuantPilot finds the top portfolio combinations with risk/return projections."
      />
      
      <Section title="Key Features" subtitle="What you can analyze in Portfolio Analytics">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <FeatureCard title="Performance Metrics" items={["CAGR, Sharpe, Sortino", "Volatility & Max Drawdown", "Cumulative Returns"]} />
          <FeatureCard title="Portfolio Allocation" items={["Latest & Average Weights", "Active Positions", "Effective Stocks Count"]} />
          <FeatureCard title="Risk Analysis" items={["Turnover Tracking", "Transaction Costs", "Benchmark Comparison"]} />
          <FeatureCard title="Methodology" items={["20 Liquid US Equities", "60-Day Rolling Window", "10% Max Position Cap"]} />
          <FeatureCard title="Benchmarks" items={["Equal Weight", "Buy & Hold", "SPY"]} />
          <FeatureCard title="Limitations" items={["Survivorship Bias", "Modeled Costs", "Historical Only"]} />
        </div>
      </Section>
      
      <div className="text-center mt-8">
        <Button size="lg" className="w-full sm:w-auto">
          Launch Portfolio Analytics →
        </Button>
      </div>
    </div>
  );
}

function ResearchLabPreview() {
  return (
    <div>
      <Hero
        eyebrow="Research Lab"
        title="Portfolio intelligence, backed by data."
        subtitle="Walk-forward quantitative research using rolling expected returns, covariance estimation, constrained mean-variance optimization, realistic turnover, transaction costs and benchmark analysis."
      />
      
      <Section title="Workflow" subtitle="Step-by-step portfolio construction">
        <div className="space-y-3">
          <StepCard step="01" title="Choose Market" desc="US or India — determines tickers, currency & benchmark" />
          <StepCard step="02" title="Investment Plan" desc="Lump sum or Monthly SIP + time horizon" />
          <StepCard step="03" title="Risk Profile" desc="Conservative / Moderate / Aggressive" />
          <StepCard step="04" title="Estimation Window" desc="60–756 days lookback for returns" />
          <StepCard step="05" title="Select Companies" desc="3–15 tickers, auto 1/N position cap" />
          <StepCard step="06" title="Ranking Method" desc="By Recommended Return, Sharpe, Vol, etc." />
        </div>
      </Section>

      <Section title="Outputs" subtitle="What you get after clicking Build">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <FeatureCard title="Top 3 Portfolios" items={["Risk Profile Comparison", "Allocation Charts", "Capital Tables"]} />
          <FeatureCard title="Stock Trends" items={["Bullish/Neutral/Bearish", "Multi-horizon Returns", "SMA 50/200 Signals"]} />
          <FeatureCard title="Backtest Analysis" items={["Growth & Drawdown Charts", "vs Equal Weight", "vs Benchmark"]} />
        </div>
      </Section>
      
      <div className="text-center mt-8">
        <Button size="lg" className="w-full sm:w-auto">
          Launch Research Lab →
        </Button>
      </div>
    </div>
  );
}

function FeatureCard({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm hover:border-slate-300 hover:shadow-md transition-all">
      <h3 className="font-display font-bold text-slate-900 text-lg mb-3">{title}</h3>
      <ul className="space-y-2 text-sm text-slate-600">
        {items.map((item, i) => (
          <li key={i} className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function StepCard({ step, title, desc }: { step: string; title: string; desc: string }) {
  return (
    <div className="flex items-start gap-4 p-4 bg-white border border-slate-200 rounded-xl shadow-sm">
      <span className="w-8 h-8 rounded-lg bg-blue-50 text-blue-700 flex items-center justify-center font-bold text-sm font-display flex-shrink-0">
        {step}
      </span>
      <div>
        <h4 className="font-semibold text-slate-900">{title}</h4>
        <p className="text-slate-500 text-sm mt-0.5">{desc}</p>
      </div>
    </div>
  );
}