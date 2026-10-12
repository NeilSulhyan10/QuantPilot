"use client";

import { cn } from "@/lib/utils";

export function Card({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "bg-white border border-slate-200 rounded-xl p-5 shadow-sm transition-all hover:border-slate-300 hover:shadow-md",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("mb-3", className)} {...props}>
      {children}
    </div>
  );
}

export function CardTitle({ children, className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3 className={cn("text-slate-500 text-xs font-semibold uppercase tracking-wider mb-2", className)} {...props}>
      {children}
    </h3>
  );
}

export function CardValue({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("text-slate-900 font-bold text-3xl font-['Space_Grotesk']", className)} {...props}>
      {children}
    </div>
  );
}

export function CardNote({ children, className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("text-slate-500 text-xs mt-2", className)} {...props}>
      {children}
    </div>
  );
}