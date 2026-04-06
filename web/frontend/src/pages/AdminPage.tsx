import { useEffect, useState, useCallback } from "react";
import { RefreshCw, Save, Plus, Trash2, Eye, EyeOff, LogIn } from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { LeaderboardEntry, TagType } from "@/types";

function LoginForm({ onLogin }: { onLogin: () => void }) {
  const [password, setPassword] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    localStorage.setItem("admin_token", password);
    onLogin();
  };

  return (
    <div className="flex items-center justify-center h-64">
      <form onSubmit={handleSubmit} className="bg-card border border-border rounded-lg p-6 space-y-4 w-80">
        <h2 className="text-lg font-bold text-center">Вход в админку</h2>
        <input
          type="password"
          placeholder="Пароль"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="w-full px-3 py-2 bg-background border border-border rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
          autoFocus
        />
        <button
          type="submit"
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          <LogIn className="w-4 h-4" /> Войти
        </button>
      </form>
    </div>
  );
}

export function AdminPage() {
  const [authed, setAuthed] = useState(!!localStorage.getItem("admin_token"));
  const [models, setModels] = useState<LeaderboardEntry[]>([]);
  const [tagTypes, setTagTypes] = useState<TagType[]>([]);
  const [syncOutput, setSyncOutput] = useState("");
  const [syncing, setSyncing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingTags, setEditingTags] = useState<string | null>(null);
  const [tagValues, setTagValues] = useState<Record<string, string>>({});
  const [newTagKey, setNewTagKey] = useState("");
  const [newTagLabel, setNewTagLabel] = useState("");

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [m, t] = await Promise.all([
        api.getLeaderboardAll(),
        api.getTagTypes(),
      ]);
      setModels(m);
      setTagTypes(t);
      setError(null);
    } catch (e: any) {
      setError(e.message);
      if (e.message.includes("401")) {
        localStorage.removeItem("admin_token");
        setAuthed(false);
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (authed) loadData();
  }, [authed, loadData]);

  const handleToggle = async (modelId: string, enabled: boolean) => {
    try {
      await api.toggleModel(modelId, enabled);
      setModels((prev) =>
        prev.map((m) => (m.model_id === modelId ? { ...m, enabled } : m))
      );
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleSync = async () => {
    setSyncing(true);
    setSyncOutput("");
    try {
      const res = await api.sync();
      setSyncOutput(res.output);
      await loadData();
    } catch (e: any) {
      setSyncOutput(`Ошибка: ${e.message}`);
    } finally {
      setSyncing(false);
    }
  };

  const handleSaveTags = async (modelId: string) => {
    try {
      await api.updateModelTags(modelId, tagValues);
      setEditingTags(null);
      await loadData();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleAddTagType = async () => {
    if (!newTagKey || !newTagLabel) return;
    try {
      await api.addTagType(newTagKey, newTagLabel, "");
      setNewTagKey("");
      setNewTagLabel("");
      const t = await api.getTagTypes();
      setTagTypes(t);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const handleDeleteTagType = async (key: string) => {
    try {
      await api.deleteTagType(key);
      const t = await api.getTagTypes();
      setTagTypes(t);
    } catch (e: any) {
      setError(e.message);
    }
  };

  if (!authed) {
    return <LoginForm onLogin={() => setAuthed(true)} />;
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        Загрузка...
      </div>
    );
  }

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Админка</h1>
        <div className="flex gap-2">
          <button
            onClick={() => {
              localStorage.removeItem("admin_token");
              setAuthed(false);
            }}
            className="px-3 py-1.5 rounded-lg text-xs text-muted-foreground hover:text-foreground bg-secondary"
          >
            Выйти
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-destructive/10 border border-destructive/30 rounded-lg px-4 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      {/* Синхронизация */}
      <div className="bg-card border border-border rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="font-medium">Синхронизация с GitHub</h2>
          <button
            onClick={handleSync}
            disabled={syncing}
            className={cn(
              "flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors",
              syncing
                ? "bg-secondary text-muted-foreground"
                : "bg-primary text-primary-foreground hover:bg-primary/90"
            )}
          >
            <RefreshCw className={cn("w-4 h-4", syncing && "animate-spin")} />
            {syncing ? "Синхронизация..." : "git pull"}
          </button>
        </div>
        {syncOutput && (
          <pre className="text-xs font-mono bg-background rounded p-3 overflow-auto max-h-40">
            {syncOutput}
          </pre>
        )}
      </div>

      {/* Типы тегов */}
      <div className="bg-card border border-border rounded-lg p-4 space-y-3">
        <h2 className="font-medium">Типы тегов (метаданные моделей)</h2>
        <div className="flex flex-wrap gap-2">
          {tagTypes.map((t) => (
            <div key={t.key} className="flex items-center gap-1 bg-secondary rounded-lg px-2 py-1 text-xs">
              <span className="font-medium">{t.label}</span>
              <span className="text-muted-foreground">({t.key})</span>
              <button
                onClick={() => handleDeleteTagType(t.key)}
                className="text-muted-foreground hover:text-destructive ml-1"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            placeholder="Ключ (напр. hf_url)"
            value={newTagKey}
            onChange={(e) => setNewTagKey(e.target.value)}
            className="px-2 py-1 bg-background border border-border rounded text-xs w-32"
          />
          <input
            placeholder="Название"
            value={newTagLabel}
            onChange={(e) => setNewTagLabel(e.target.value)}
            className="px-2 py-1 bg-background border border-border rounded text-xs w-40"
          />
          <button
            onClick={handleAddTagType}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-secondary hover:bg-secondary/80 transition-colors"
          >
            <Plus className="w-3 h-3" /> Добавить
          </button>
        </div>
      </div>

      {/* Модели */}
      <div className="bg-card border border-border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-border">
          <h2 className="font-medium">Модели ({models.length})</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left">
                <th className="px-4 py-2 font-medium text-muted-foreground w-10">Вкл</th>
                <th className="px-4 py-2 font-medium text-muted-foreground">#</th>
                <th className="px-4 py-2 font-medium text-muted-foreground">Модель</th>
                <th className="px-4 py-2 font-medium text-muted-foreground">Взвеш./ответ</th>
                <th className="px-4 py-2 font-medium text-muted-foreground">Теги</th>
                <th className="px-4 py-2 font-medium text-muted-foreground w-20">Действия</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr
                  key={m.model_id}
                  className={cn(
                    "border-b border-border/50",
                    !m.enabled && "opacity-50"
                  )}
                >
                  <td className="px-4 py-2">
                    <button
                      onClick={() => handleToggle(m.model_id, !m.enabled)}
                      className="text-muted-foreground hover:text-foreground"
                    >
                      {m.enabled ? (
                        <Eye className="w-4 h-4 text-success" />
                      ) : (
                        <EyeOff className="w-4 h-4" />
                      )}
                    </button>
                  </td>
                  <td className="px-4 py-2 font-mono text-muted-foreground">
                    {m.rank ?? "—"}
                  </td>
                  <td className="px-4 py-2 font-medium">{m.display_name}</td>
                  <td className="px-4 py-2 font-mono">
                    {m.weighted_per_avg.toFixed(2)}
                  </td>
                  <td className="px-4 py-2">
                    {editingTags === m.model_id ? (
                      <div className="space-y-1">
                        {tagTypes.map((t) => (
                          <div key={t.key} className="flex items-center gap-1">
                            <span className="text-xs text-muted-foreground w-20 shrink-0">
                              {t.label}:
                            </span>
                            <input
                              value={tagValues[t.key] || ""}
                              onChange={(e) =>
                                setTagValues((prev) => ({
                                  ...prev,
                                  [t.key]: e.target.value,
                                }))
                              }
                              className="px-1.5 py-0.5 bg-background border border-border rounded text-xs flex-1"
                            />
                          </div>
                        ))}
                        <div className="flex gap-1 mt-1">
                          <button
                            onClick={() => handleSaveTags(m.model_id)}
                            className="flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-primary text-primary-foreground"
                          >
                            <Save className="w-3 h-3" /> Сохранить
                          </button>
                          <button
                            onClick={() => setEditingTags(null)}
                            className="px-2 py-0.5 rounded text-xs bg-secondary"
                          >
                            Отмена
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="flex flex-wrap gap-1">
                        {Object.entries(m.tags || {}).map(
                          ([k, v]) =>
                            v && (
                              <span
                                key={k}
                                className="text-[10px] px-1.5 py-0.5 rounded bg-secondary text-secondary-foreground"
                              >
                                {k}: {v.length > 30 ? v.slice(0, 30) + "..." : v}
                              </span>
                            )
                        )}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-2">
                    <button
                      onClick={() => {
                        setEditingTags(m.model_id);
                        setTagValues(m.tags || {});
                      }}
                      className="text-xs text-primary hover:underline"
                    >
                      Теги
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
