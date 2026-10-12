"use client";

import { cn } from "@/lib/utils";
import { TableHTMLAttributes, TdHTMLAttributes, ThHTMLAttributes } from "react";

export interface Column<T> {
  key: string;
  header: string;
  render?: (row: T, index: number) => React.ReactNode;
  className?: string;
  headerClassName?: string;
}

export interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (row: T) => string;
  className?: string;
  striped?: boolean;
  hoverable?: boolean;
  emptyMessage?: string;
}

export function Table<T>({ 
  columns, 
  data, 
  keyExtractor, 
  className, 
  striped = true, 
  hoverable = true,
  emptyMessage = "No data available"
}: TableProps<T>) {
  if (data.length === 0) {
    return (
      <div className={cn("bg-white border border-slate-200 rounded-xl overflow-hidden", className)}>
        <div className="p-8 text-center text-slate-500">{emptyMessage}</div>
      </div>
    );
  }

  return (
    <div className={cn("bg-white border border-slate-200 rounded-xl overflow-hidden", className)}>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" role="table">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={cn(
                    "px-4 py-3 text-left font-semibold text-slate-600 uppercase tracking-wider text-xs",
                    col.headerClassName
                  )}
                  scope="col"
                >
                  {col.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.map((row, rowIndex) => (
              <tr
                key={keyExtractor(row)}
                className={cn(
                  striped && rowIndex % 2 === 1 && "bg-slate-50/50",
                  hoverable && "hover:bg-blue-50/50 transition-colors"
                )}
              >
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className={cn(
                      "px-4 py-3 text-slate-700",
                      col.className
                    )}
                  >
                    {col.render ? col.render(row, rowIndex) : String((row as Record<string, unknown>)[col.key] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}