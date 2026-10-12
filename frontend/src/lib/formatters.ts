export function formatPercent(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || isNaN(value)) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatWeight(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) return "—";
  const pct = value * 100;
  if (pct >= 1) return `${pct.toFixed(2)}%`;
  if (pct >= 0.01) return `${pct.toFixed(4)}%`;
  if (pct > 0) return `${pct.toExponential(2)}%`;
  return "0.00%";
}

export function formatCurrency(value: number | null | undefined, symbol = "$", decimals = 0): string {
  if (value === null || value === undefined || isNaN(value)) return "—";
  const locale = symbol === "₹" ? "en-IN" : symbol === "$" ? "en-US" : undefined;
  return `${symbol}${value.toLocaleString(locale, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })}`;
}

export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || isNaN(value)) return "—";
  return value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export function formatSharpe(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) return "—";
  return value.toFixed(2);
}

export const CURRENCY_SYMBOLS: Record<string, string> = {
  USD: "$",
  INR: "₹",
};

export function getCurrencySymbol(currency: string): string {
  return CURRENCY_SYMBOLS[currency] || currency;
}

export function getTrendBadge(trend: "bearish" | "neutral" | "bullish"): { label: string; color: string } {
  switch (trend) {
    case "bullish":
      return { label: "🟢 BULLISH", color: "bg-green-100 text-green-800" };
    case "bearish":
      return { label: "🔴 BEARISH", color: "bg-red-100 text-red-800" };
    default:
      return { label: "🟡 NEUTRAL", color: "bg-yellow-100 text-yellow-800" };
  }
}

export function getVsMABadge(val: "above" | "below" | null): { label: string; color: string } {
  if (val === "above") return { label: "Above", color: "text-green-600 font-semibold" };
  if (val === "below") return { label: "Below", color: "text-red-600 font-semibold" };
  return { label: "—", color: "text-slate-400" };
}
