import { useState, useMemo, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { toPng } from "html-to-image";
import { ArrowUpDown, ArrowUp, ArrowDown, Search, Camera, X, Download, Sparkles } from "lucide-react";
import type { LeaderboardEntry } from "@/types";
import { cn, formatWithSE } from "@/lib/utils";
import { ErrorDistributionBar } from "./ErrorDistributionBar";
import { ModelTooltip } from "./ModelTooltip";

type SortKey =
  | "rank"
  | "display_name"
  | "l3_per_1k"
  | "l2_per_1k"
  | "l1_per_1k"
  | "weighted_per_1k"
  | "weighted_per_avg";

type SortDir = "asc" | "desc";

interface Props {
  data: LeaderboardEntry[];
}

function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = Math.min(100, Math.max(0, (1 - value / max) * 100));
  return (
    <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all"
        style={{
          width: `${pct}%`,
          background: `linear-gradient(90deg, #22c55e, #f59e0b, #ef4444)`,
          backgroundSize: `${(100 / pct) * 100}% 100%`,
        }}
      />
    </div>
  );
}

export function LeaderboardTable({ data }: Props) {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("weighted_per_avg");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const [screenshotMode, setScreenshotMode] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [highlighted, setHighlighted] = useState<Set<string>>(new Set());
  const tableRef = useRef<HTMLDivElement>(null);

  const maxWeighted = useMemo(
    () => Math.max(...data.map((d) => d.weighted_per_1k), 1),
    [data]
  );

  const filtered = useMemo(() => {
    let items = data;
    if (search) {
      const q = search.toLowerCase();
      items = items.filter(
        (d) =>
          d.display_name.toLowerCase().includes(q) ||
          d.original_model.toLowerCase().includes(q)
      );
    }
    const sorted = [...items].sort((a, b) => {
      const dir = sortDir === "asc" ? 1 : -1;
      if (sortKey === "display_name") {
        return a.display_name.localeCompare(b.display_name, "ru") * dir;
      }
      return ((a[sortKey] ?? 0) - (b[sortKey] ?? 0)) * dir;
    });
    return sorted;
  }, [data, search, sortKey, sortDir]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return <ArrowUpDown className="w-3 h-3 opacity-30" />;
    return sortDir === "asc" ? (
      <ArrowUp className="w-3 h-3" />
    ) : (
      <ArrowDown className="w-3 h-3" />
    );
  };

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleHighlight = (id: string) => {
    setHighlighted((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleScreenshot = useCallback(async () => {
    if (!tableRef.current) return;
    try {
      const rows = tableRef.current.querySelectorAll<HTMLElement>("tbody tr[data-model-id]");
      const hidden: HTMLElement[] = [];
      if (selected.size > 0) {
        rows.forEach((row) => {
          if (!selected.has(row.dataset.modelId || "")) {
            row.style.display = "none";
            hidden.push(row);
          }
        });
      }

      const controlCells = tableRef.current.querySelectorAll<HTMLElement>("[data-screenshot-hide]");
      controlCells.forEach((el) => (el.style.display = "none"));

      const wrapper = tableRef.current;
      const table = wrapper.querySelector<HTMLElement>("table");
      const prevWrapperWidth = wrapper.style.width;
      const prevTableWidth = table?.style.width ?? "";
      wrapper.style.width = "fit-content";
      if (table) table.style.width = "auto";

      const isDark = document.documentElement.classList.contains("dark");
      const dataUrl = await toPng(wrapper, {
        pixelRatio: 3,
        backgroundColor: isDark ? "#0a0a0a" : "#ffffff",
      });

      wrapper.style.width = prevWrapperWidth;
      if (table) table.style.width = prevTableWidth;
      hidden.forEach((row) => (row.style.display = ""));
      controlCells.forEach((el) => (el.style.display = ""));

      const link = document.createElement("a");
      link.download = `ruqualbench-leaderboard-${Date.now()}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error("Screenshot error:", err);
    }
  }, [selected]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 flex-wrap">
        <div className="relative flex-1 min-w-[200px] max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Поиск модели..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-3 py-2 bg-card border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
        <button
          onClick={() => {
            setScreenshotMode((v) => !v);
            if (screenshotMode) {
              setSelected(new Set());
              setHighlighted(new Set());
            }
          }}
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
            screenshotMode
              ? "bg-primary text-primary-foreground"
              : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
          )}
        >
          {screenshotMode ? <X className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
          {screenshotMode ? "Выйти" : "Скриншот"}
        </button>
        {screenshotMode && (
          <button
            onClick={handleScreenshot}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-success text-white hover:bg-success/90 transition-colors"
          >
            <Download className="w-4 h-4" />
            Скачать{selected.size > 0 ? ` (${selected.size})` : ""}
          </button>
        )}
      </div>

      <div ref={tableRef} className="rounded-xl border border-border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-separate" style={{ borderSpacing: 0 }}>
            <thead>
              <tr className="bg-card border-b border-border">
                {screenshotMode && (
                  <th className="px-2 py-3 text-left font-medium text-muted-foreground w-16" data-screenshot-hide />
                )}
                <th
                  className="px-3 py-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground w-12"
                  onClick={() => handleSort("rank")}
                >
                  <div className="flex items-center gap-1">
                    # <SortIcon col="rank" />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-left font-medium text-muted-foreground cursor-pointer hover:text-foreground min-w-[200px]"
                  onClick={() => handleSort("display_name")}
                >
                  <div className="flex items-center gap-1">
                    Модель <SortIcon col="display_name" />
                  </div>
                </th>
                <th className="px-3 py-3 text-left font-medium text-muted-foreground w-[160px]">
                  Типы ошибок
                </th>
                <th
                  className="px-3 py-3 text-right font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("l3_per_1k")}
                >
                  <div className="flex items-center justify-end gap-1">
                    L3/1K <SortIcon col="l3_per_1k" />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-right font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("l2_per_1k")}
                >
                  <div className="flex items-center justify-end gap-1">
                    L2/1K <SortIcon col="l2_per_1k" />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-right font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("l1_per_1k")}
                >
                  <div className="flex items-center justify-end gap-1">
                    L1/1K <SortIcon col="l1_per_1k" />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-right font-medium text-muted-foreground cursor-pointer hover:text-foreground min-w-[180px]"
                  onClick={() => handleSort("weighted_per_1k")}
                >
                  <div className="flex items-center justify-end gap-1">
                    Взвеш./1K <SortIcon col="weighted_per_1k" />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-right font-medium text-muted-foreground cursor-pointer hover:text-foreground"
                  onClick={() => handleSort("weighted_per_avg")}
                >
                  <div className="flex items-center justify-end gap-1">
                    Взвеш./ответ <SortIcon col="weighted_per_avg" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((entry) => {
                const isHighlighted = highlighted.has(entry.model_id);
                return (
                  <tr
                    key={entry.model_id}
                    data-model-id={entry.model_id}
                    className={cn(
                      "border-b border-border/50 hover:bg-secondary/50 transition-colors",
                    )}
                    style={
                      isHighlighted
                        ? { boxShadow: "inset 0 0 0 2px #22c55e", borderRadius: "8px" }
                        : undefined
                    }
                  >
                    {screenshotMode && (
                      <td className="px-2 py-2.5" data-screenshot-hide>
                        <div className="flex items-center gap-1">
                          <input
                            type="checkbox"
                            checked={selected.has(entry.model_id)}
                            onChange={() => toggleSelect(entry.model_id)}
                            className="rounded"
                          />
                          <button
                            onClick={() => toggleHighlight(entry.model_id)}
                            title="Выделить рамкой"
                            className={cn(
                              "p-0.5 rounded transition-colors",
                              isHighlighted
                                ? "text-green-600"
                                : "text-muted-foreground/40 hover:text-green-600"
                            )}
                          >
                            <Sparkles className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                    )}
                    <td className="px-3 py-2.5 font-mono text-muted-foreground">
                      {entry.rank}
                    </td>
                    <td className="px-3 py-2.5">
                      <ModelTooltip entry={entry}>
                        <button
                          onClick={() => navigate(`/results/${entry.model_id}`)}
                          className="text-left font-medium text-primary hover:underline cursor-pointer"
                        >
                          {entry.display_name}
                        </button>
                      </ModelTooltip>
                    </td>
                    <td className="px-3 py-2.5">
                      <ErrorDistributionBar errorTypes={entry.error_types} />
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-level-3">
                      {formatWithSE(entry.l3_per_1k, entry.l3_per_1k_se)}
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-level-2">
                      {formatWithSE(entry.l2_per_1k, entry.l2_per_1k_se)}
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-level-1">
                      {formatWithSE(entry.l1_per_1k, entry.l1_per_1k_se)}
                    </td>
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <ProgressBar value={entry.weighted_per_1k} max={maxWeighted} />
                        <span className="font-mono text-right min-w-[80px]">
                          {formatWithSE(entry.weighted_per_1k, entry.weighted_per_1k_se)}
                        </span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono font-bold">
                      {formatWithSE(entry.weighted_per_avg, entry.weighted_per_avg_se)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="text-xs text-muted-foreground space-y-1 px-1">
        <p>L3/L2/L1 — ошибки уровня 3 (критичные), 2 (обычные), 1 (незначительные) на 1000 токенов o200k_base.</p>
        <p>Взвеш./1K = L3×2 + L2×1 + L1×0.5 на 1000 токенов. Взвеш./ответ — та же метрика на средний ответ.</p>
        <p>± — стандартная ошибка (SE) по серии прогонов.</p>
      </div>
    </div>
  );
}
