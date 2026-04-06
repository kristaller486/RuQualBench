import { useState, useRef, useEffect, useCallback } from "react";
import { createPortal } from "react-dom";
import type { LeaderboardEntry } from "@/types";

interface Props {
  entry: LeaderboardEntry;
  children: React.ReactNode;
}

export function ModelTooltip({ entry, children }: Props) {
  const [show, setShow] = useState(false);
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);

  const updatePosition = useCallback(() => {
    if (!triggerRef.current || !tooltipRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    const tip = tooltipRef.current.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const goUp = spaceBelow < tip.height + 8;

    setPos({
      top: goUp ? rect.top - tip.height - 4 : rect.bottom + 4,
      left: Math.max(8, Math.min(rect.left, window.innerWidth - tip.width - 8)),
    });
  }, []);

  useEffect(() => {
    if (!show) return;
    requestAnimationFrame(updatePosition);
  }, [show, updatePosition]);

  const tags = entry.tags || {};
  const hasInfo =
    Object.values(tags).some((v) => v) ||
    entry.original_model ||
    entry.judge_model;

  return (
    <div
      className="relative inline-block"
      ref={triggerRef}
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show &&
        hasInfo &&
        createPortal(
          <div
            ref={(el) => {
              (tooltipRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
              if (el) requestAnimationFrame(updatePosition);
            }}
            className="fixed z-[9999] bg-card border border-border rounded-lg px-3 py-2 text-xs shadow-xl min-w-[220px] max-w-[360px]"
            style={pos ? { top: pos.top, left: pos.left } : { visibility: "hidden", top: 0, left: 0 }}
          >
            <div className="space-y-1">
              <div className="text-muted-foreground">
                Модель: <span className="text-foreground font-mono">{entry.original_model}</span>
              </div>
              <div className="text-muted-foreground">
                Судья: <span className="text-foreground font-mono">{entry.judge_model}</span>
              </div>
              <div className="text-muted-foreground">
                Датасет: <span className="text-foreground">{entry.dataset}</span> · Прогонов: {entry.num_runs}
              </div>
              {Object.entries(tags).map(([k, v]) =>
                v ? (
                  <div key={k} className="text-muted-foreground">
                    {k}:{" "}
                    {v.startsWith("http") ? (
                      <a href={v} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                        {v}
                      </a>
                    ) : (
                      <span className="text-foreground">{v}</span>
                    )}
                  </div>
                ) : null
              )}
            </div>
          </div>,
          document.body
        )}
    </div>
  );
}
