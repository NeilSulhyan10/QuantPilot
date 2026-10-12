"use client";

import { cn } from "@/lib/utils";
import { HTMLAttributes, forwardRef } from "react";

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "error" | "info";
  size?: "sm" | "md";
}

export const Badge = forwardRef<HTMLSpanElement, BadgeProps>(
  ({ className, variant = "default", size = "md", children, ...props }, ref) => {
    const variants = {
      default: "bg-blue-50 text-blue-700 border-blue-200",
      success: "bg-green-50 text-green-700 border-green-200",
      warning: "bg-amber-50 text-amber-700 border-amber-200",
      error: "bg-red-50 text-red-700 border-red-200",
      info: "bg-purple-50 text-purple-700 border-purple-200",
    };

    const sizes = {
      sm: "px-2 py-0.5 text-xs",
      md: "px-2.5 py-1 text-sm",
    };

    return (
      <span
        ref={ref}
        className={cn(
          "inline-flex items-center font-semibold rounded-full border",
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      >
        {children}
      </span>
    );
  }
);

Badge.displayName = "Badge";