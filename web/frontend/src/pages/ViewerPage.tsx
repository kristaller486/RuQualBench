import { useEffect, useState, useCallback } from "react";
import { useParams, Link, useSearchParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { api } from "@/lib/api";
import type { ModelResults, RunResults, DialogResult } from "@/types";
import { cn } from "@/lib/utils";
import { Dashboard } from "@/components/viewer/Dashboard";
import { AnswerViewer } from "@/components/viewer/AnswerViewer";

type Tab = "dashboard" | "answers";

export function ViewerPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const [searchParams, setSearchParams] = useSearchParams();

  const tab = (searchParams.get("tab") as Tab) || "dashboard";
  const selectedRun = parseInt(searchParams.get("run") || "0", 10);
  const selectedDialog = parseInt(searchParams.get("dialog") || "0", 10);

  const [data, setData] = useState<ModelResults | null>(null);
  const [runData, setRunData] = useState<RunResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);

  const updateParams = useCallback(
    (updates: Record<string, string | undefined>) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        for (const [k, v] of Object.entries(updates)) {
          if (v === undefined || v === "0" && k !== "tab") {
            next.delete(k);
          } else {
            next.set(k, v);
          }
        }
        return next;
      }, { replace: true });
    },
    [setSearchParams]
  );

  const setTab = useCallback(
    (t: Tab) => {
      if (t === "dashboard") {
        updateParams({ tab: undefined, run: undefined, dialog: undefined });
      } else {
        updateParams({ tab: t });
      }
    },
    [updateParams]
  );

  const setSelectedRun = useCallback(
    (r: number) => updateParams({ run: r === 0 ? undefined : String(r), dialog: undefined }),
    [updateParams]
  );

  const setSelectedDialog = useCallback(
    (d: number) => updateParams({ dialog: d === 0 ? undefined : String(d) }),
    [updateParams]
  );

  useEffect(() => {
    if (!modelId) return;
    api
      .getModelResults(modelId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [modelId]);

  useEffect(() => {
    if (!modelId || tab !== "answers") return;
    setLoadingRun(true);
    api
      .getRunResults(modelId, selectedRun)
      .then(setRunData)
      .catch((e) => setError(e.message))
      .finally(() => setLoadingRun(false));
  }, [modelId, selectedRun, tab]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        Загрузка...
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center justify-center h-64 text-destructive">
        Ошибка: {error || "Данные не найдены"}
      </div>
    );
  }

  const currentResult: DialogResult | null =
    runData?.results?.[selectedDialog] ?? null;

  return (
    <div className="max-w-[1600px] mx-auto px-6 py-6 flex flex-col h-[calc(100vh-57px)]">
      <div className="flex items-center gap-4 mb-4 flex-shrink-0">
        <Link
          to="/"
          className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="w-4 h-4" /> Назад
        </Link>
        <h1 className="text-xl font-bold">{data.display_name}</h1>
        {data.tags &&
          Object.entries(data.tags).map(
            ([k, v]) =>
              v && (
                <span
                  key={k}
                  className="text-xs px-2 py-0.5 rounded bg-secondary text-secondary-foreground"
                >
                  {k}: {v}
                </span>
              )
          )}
      </div>

      <div className="flex items-center gap-2 mb-4 flex-shrink-0">
        <button
          onClick={() => setTab("dashboard")}
          className={cn(
            "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
            tab === "dashboard"
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground hover:bg-secondary"
          )}
        >
          Дашборд
        </button>
        <button
          onClick={() => setTab("answers")}
          className={cn(
            "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
            tab === "answers"
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground hover:bg-secondary"
          )}
        >
          Ответы
        </button>

        {tab === "answers" && data.runs.length > 1 && (
          <div className="flex items-center gap-1 ml-4">
            <span className="text-xs text-muted-foreground">Прогон:</span>
            {data.runs.map((_, i) => (
              <button
                key={i}
                onClick={() => setSelectedRun(i)}
                className={cn(
                  "px-2 py-0.5 rounded text-xs font-medium transition-colors",
                  selectedRun === i
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                #{i + 1}
              </button>
            ))}
          </div>
        )}
      </div>

      {tab === "dashboard" && <Dashboard data={data} />}

      {tab === "answers" && (
        <div className="flex-1 flex gap-4 overflow-hidden min-h-0">
          <div
            className="w-[200px] flex-shrink-0 bg-card border border-border rounded-lg overflow-auto outline-none focus:ring-2 focus:ring-primary/30"
            tabIndex={0}
            onKeyDown={(e) => {
              const count = runData?.results.length ?? 0;
              if (count === 0) return;
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setSelectedDialog(Math.min(selectedDialog + 1, count - 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setSelectedDialog(Math.max(selectedDialog - 1, 0));
              }
            }}
          >
            <div className="p-2 text-xs font-bold uppercase text-muted-foreground border-b border-border">
              Промпты
            </div>
            <div className="p-1">
              {loadingRun ? (
                <div className="p-4 text-xs text-muted-foreground">Загрузка...</div>
              ) : (
                runData?.results.map((r, i) => {
                  const total =
                    (r.mistakes_count["3"] || 0) +
                    (r.mistakes_count["2"] || 0) +
                    (r.mistakes_count["1"] || 0);
                  const hasL3 = (r.mistakes_count["3"] || 0) > 0;
                  const hasL2 = (r.mistakes_count["2"] || 0) > 0;

                  return (
                    <button
                      key={i}
                      onClick={() => setSelectedDialog(i)}
                      className={cn(
                        "w-full text-left px-2 py-1.5 rounded text-xs transition-colors mb-0.5",
                        selectedDialog === i
                          ? "bg-primary/20 text-primary"
                          : "hover:bg-secondary text-muted-foreground hover:text-foreground",
                        r.error && "text-destructive"
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <span>#{r.dialog_id + 1}</span>
                        {total > 0 && (
                          <span
                            className={cn(
                              "font-bold",
                              hasL3
                                ? "text-red-600"
                                : hasL2
                                ? "text-amber-600"
                                : "text-yellow-600"
                            )}
                          >
                            {total}
                          </span>
                        )}
                      </div>
                    </button>
                  );
                })
              )}
            </div>
          </div>

          <div className="flex-1 overflow-auto">
            {currentResult ? (
              <AnswerViewer result={currentResult} />
            ) : (
              <div className="text-muted-foreground text-center py-16">
                Выберите промпт слева
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
