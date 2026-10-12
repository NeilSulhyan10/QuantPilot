"use client";

import { cn } from "@/lib/utils";

export interface SectionProps {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  className?: string;
}

export function Section({ title, subtitle, children, className }: SectionProps) {
  return (
    <section className={cn("mt-10 mb-4", className)}>
      <div className="flex items-center gap-3 mb-3">
        <div className="w-1 h-6 rounded-full bg-gradient-to-b from-blue-400 to-blue-600 shadow-sm" />
        <h2 className="text-slate-900 font-bold text-xl font-['Space_Grotesk'] tracking-tight">{title}</h2>
      </div>
      {subtitle && <p className="text-slate-500 text-sm mb-4 ml-7">{subtitle}</p>}
      <div className="ml-7">{children}</div>
    </section>
  );
}

export interface ChipProps {
  label: string;
  variant?: "default" | "blue" | "purple" | "amber";
  className?: string;
}

export function Chip({ label, variant = "default", className }: ChipProps) {
  const variants = {
    default: "border-slate-200 bg-slate-50 text-slate-700",
    blue: "border-blue-200 bg-blue-50 text-blue-700",
    purple: "border-purple-200 bg-purple-50 text-purple-700",
    amber: "border-amber-200 bg-amber-50 text-amber-700",
  };

  return (
    <span className={cn(
      "inline-flex items-center px-3 py-1 rounded-full border text-xs font-semibold",
      variants[variant],
      className
    )}>
      {label}
    </span>
  );
}

export function Hero({ title, subtitle, eyebrow, children, className }: {
  title: string;
  subtitle?: string;
  eyebrow?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn(
      "relative overflow-hidden border border-slate-200 rounded-2xl p-8 bg-gradient-to-br from-white to-slate-50 shadow-sm",
      className
    )}>
      {eyebrow && (
        <div className="text-blue-600 uppercase tracking-widest text-[10px] font-bold mb-3">
          {eyebrow}
        </div>
      )}
      <h1 className="text-slate-900 font-bold text-3xl font-['Space_Grotesk'] tracking-tight mb-3">
        {title}
      </h1>
      {subtitle && (
        <p className="text-slate-500 text-base leading-relaxed max-w-2xl mb-6">
          {subtitle}
        </p>
      )}
      {children}
    </div>
  );
}