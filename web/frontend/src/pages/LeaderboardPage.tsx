import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { LeaderboardEntry } from "@/types";
import { LeaderboardTable } from "@/components/leaderboard/LeaderboardTable";

export function LeaderboardPage() {
  const [data, setData] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getLeaderboard()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        Загрузка...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64 text-destructive">
        Ошибка: {error}
      </div>
    );
  }

  return (
    <div className="max-w-[1600px] mx-auto px-6 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">RuQualBench V2 Leaderboard</h1>
        <p className="text-muted-foreground text-sm mt-1">
          Бенчмарк качества русского языка для LLM · {data.length} моделей
        </p>
      </div>
      <LeaderboardTable data={data} />
    </div>
  );
}
