"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Home", href: "/" },
  { name: "Portfolio Analytics", href: "/portfolio-analytics" },
  { name: "Research Lab", href: "/research-lab" },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="border-b border-slate-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-3" aria-label="QuantPilot Home">
              <div className="w-9 h-9 rounded-xl flex items-center justify-center bg-gradient-to-br from-blue-400 to-blue-600 text-white font-display font-bold text-lg shadow-lg">
                Q
              </div>
              <span className="hidden sm:block font-display font-bold text-xl text-slate-900 tracking-tight">QuantPilot</span>
            </Link>

            <div className="hidden md:flex items-center gap-1">
              {navigation.map((item) => {
                const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
                return (
                  <Link
                    key={item.name}
                    href={item.href}
                    className={cn(
                      "px-4 py-2 rounded-lg text-sm font-semibold transition-all",
                      isActive
                        ? "bg-blue-50 text-blue-700 shadow-sm border border-blue-200"
                        : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                    )}
                    aria-current={isActive ? "page" : undefined}
                  >
                    {item.name}
                  </Link>
                );
              })}
            </div>
          </div>

          <div className="hidden md:block">
            <Link
              href="/"
              className="px-4 py-2 rounded-lg text-sm font-semibold text-slate-500 hover:text-slate-700 hover:bg-slate-50 transition-colors"
            >
              Overview
            </Link>
          </div>
        </div>

        {/* Mobile menu */}
        <div className="md:hidden py-4 border-t border-slate-200">
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            {navigation.map((item) => {
              const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "px-3 py-2 rounded-lg text-sm font-semibold whitespace-nowrap transition-all flex-shrink-0",
                    isActive
                      ? "bg-blue-50 text-blue-700 shadow-sm border border-blue-200"
                      : "text-slate-500 hover:text-slate-700 hover:bg-slate-50"
                  )}
                  aria-current={isActive ? "page" : undefined}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}