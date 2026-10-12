"use client";

import { cn } from "@/lib/utils";
import { InputHTMLAttributes, forwardRef } from "react";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, error, helperText, id, ...props }, ref) => {
    const inputId = id || label?.toLowerCase().replace(/\s+/g, "-");
    
    return (
      <div className="w-full">
        {label && (
          <label htmlFor={inputId} className="block text-slate-600 text-sm font-medium mb-1.5">
            {label}
          </label>
        )}
        <input
          ref={ref}
          id={inputId}
          className={cn(
            "w-full px-4 py-2.5 rounded-lg border bg-white text-slate-900 placeholder-slate-400",
            "border-slate-300 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 focus:outline-none",
            "transition-colors",
            error && "border-red-500 focus:border-red-500 focus:ring-red-500/20",
            className
          )}
          aria-invalid={error ? "true" : "false"}
          aria-describedby={error ? `${inputId}-error` : helperText ? `${inputId}-helper` : undefined}
          {...props}
        />
        {error && (
          <p id={`${inputId}-error`} className="mt-1.5 text-red-600 text-sm" role="alert">
            {error}
          </p>
        )}
        {helperText && !error && (
          <p id={`${inputId}-helper`} className="mt-1.5 text-slate-500 text-sm">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";