"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { api } from "@/lib/api";
import { getCurrencySymbol } from "@/lib/formatters";
import type { MarketConfig } from "@/lib/types";

interface MarketContextType {
  market: MarketConfig | null;
  markets: MarketConfig[];
  setMarket: (market: MarketConfig) => void;
  currencySymbol: string;
  loading: boolean;
  error: string | null;
}

const MarketContext = createContext<MarketContextType | undefined>(undefined);

export function MarketProvider({ children }: { children: ReactNode }) {
  const [markets, setMarkets] = useState<MarketConfig[]>([]);
  const [market, setMarketState] = useState<MarketConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadMarkets() {
      try {
        setLoading(true);
        const data = await api.markets();
        setMarkets(data);
        if (data.length > 0) {
          const savedMarket = localStorage.getItem("quantpilot-market");
          const initialMarket = data.find(m => m.market === savedMarket) || data[0];
          setMarketState(initialMarket);
          localStorage.setItem("quantpilot-market", initialMarket.market);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load markets");
        // Fallback markets
        const fallbackMarkets: MarketConfig[] = [
          { market: "us", name: "United States", currency: "USD", benchmark: "SPY", ticker_suffix: "" },
          { market: "india", name: "India", currency: "INR", benchmark: "^NSEI", ticker_suffix: ".NS" },
        ];
        setMarkets(fallbackMarkets);
        setMarketState(fallbackMarkets[0]);
      } finally {
        setLoading(false);
      }
    }
    loadMarkets();
  }, []);

  const setMarket = (newMarket: MarketConfig) => {
    setMarketState(newMarket);
    localStorage.setItem("quantpilot-market", newMarket.market);
  };

  const currencySymbol = market ? getCurrencySymbol(market.currency) : "$";

  return (
    <MarketContext.Provider value={{ market, markets, setMarket, currencySymbol, loading, error }}>
      {children}
    </MarketContext.Provider>
  );
}

export function useMarket() {
  const context = useContext(MarketContext);
  if (!context) {
    throw new Error("useMarket must be used within a MarketProvider");
  }
  return context;
}