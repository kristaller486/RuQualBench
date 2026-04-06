import type { ErrorTypes } from "@/types";
import { ERROR_TYPE_COLORS, ERROR_TYPE_LABELS } from "@/lib/utils";
import { useState } from "react";

interface Props {
  errorTypes: ErrorTypes;
  width?: number;
  height?: number;
}

export function ErrorDistributionBar({ errorTypes, width = 140, height = 22 }: Props) {
  const [hovered, setHovered] = useState<string | null>(null);

  const entries = Object.entries(errorTypes).filter(([, v]) => v > 0);
  const total = entries.reduce((sum, [, v]) => sum + v, 0);

  if (total === 0) {
    return (
      <div
        style={{ width, height }}
        className="rounded bg-muted flex items-center justify-center text-xs text-muted-foreground"
      >
        0
      </div>
    );
  }

  return (
    <div className="relative" style={{ width }}>
      <div
        className="flex"
        style={{ height }}
      >
        {(() => {
          const visible = entries.filter(([, count]) => (count / total) * 100 >= 1);
          return visible.map(([type, count], i) => {
            const pct = (count / total) * 100;
            const isFirst = i === 0;
            const isLast = i === visible.length - 1;
            const borderRadius = `${isFirst ? "4px" : "0"} ${isLast ? "4px" : "0"} ${isLast ? "4px" : "0"} ${isFirst ? "4px" : "0"}`;
            return (
              <div
                key={type}
                style={{
                  width: `${pct}%`,
                  backgroundColor: ERROR_TYPE_COLORS[type] || "#6b7280",
                  opacity: hovered && hovered !== type ? 0.4 : 1,
                  transition: "opacity 0.15s",
                  borderRadius,
                }}
                className="cursor-default"
                onMouseEnter={() => setHovered(type)}
                onMouseLeave={() => setHovered(null)}
              />
            );
          });
        })()}
      </div>
      {hovered && (
        <div className="absolute z-50 bottom-full mb-1 left-1/2 -translate-x-1/2 bg-card border border-border rounded-md px-2 py-1 text-xs whitespace-nowrap shadow-lg">
          {ERROR_TYPE_LABELS[hovered] || hovered}:{" "}
          {errorTypes[hovered as keyof ErrorTypes]} ({((errorTypes[hovered as keyof ErrorTypes] / total) * 100).toFixed(0)}%)
        </div>
      )}
    </div>
  );
}
