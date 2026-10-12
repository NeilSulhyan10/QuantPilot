import type { Metadata } from "next";
import { Navigation } from "@/components/Navigation";
import { MarketProvider } from "@/context/MarketContext";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuantPilot",
  description: "Adaptive Quantitative Portfolio Manager",
  other: {
    'font-preconnect': 'https://fonts.googleapis.com https://fonts.gstatic.com',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full antialiased">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-full flex flex-col">
        <MarketProvider>
          <Navigation />
          <main className="flex-1">{children}</main>
        </MarketProvider>
      </body>
    </html>
  );
}