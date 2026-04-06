import type {
  LeaderboardEntry,
  ModelResults,
  RunResults,
  CombinedResults,
  TagType,
} from "@/types";

const BASE = "";

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

function adminHeaders(): HeadersInit {
  const token = localStorage.getItem("admin_token") || "";
  return {
    "Content-Type": "application/json",
    "X-Admin-Token": token,
  };
}

export const api = {
  getLeaderboard: () => fetchJSON<LeaderboardEntry[]>("/api/leaderboard"),

  getLeaderboardAll: () => fetchJSON<LeaderboardEntry[]>("/api/leaderboard/all"),

  getModelResults: (modelId: string) =>
    fetchJSON<ModelResults>(`/api/results/${modelId}`),

  getRunResults: (modelId: string, runIndex: number) =>
    fetchJSON<RunResults>(`/api/results/${modelId}/run/${runIndex}`),

  getCombinedResults: (modelId: string) =>
    fetchJSON<CombinedResults>(`/api/results/${modelId}/combined`),

  toggleModel: (modelId: string, enabled: boolean) =>
    fetchJSON<{ ok: boolean }>(`/api/admin/models/${modelId}/toggle`, {
      method: "POST",
      headers: adminHeaders(),
      body: JSON.stringify({ enabled }),
    }),

  updateModelTags: (modelId: string, tags: Record<string, string>) =>
    fetchJSON<{ ok: boolean }>(`/api/admin/models/${modelId}/tags`, {
      method: "POST",
      headers: adminHeaders(),
      body: JSON.stringify({ tags }),
    }),

  getTagTypes: () =>
    fetchJSON<TagType[]>("/api/admin/tag-types", {
      headers: adminHeaders(),
    }),

  addTagType: (key: string, label: string, description: string) =>
    fetchJSON<{ ok: boolean }>("/api/admin/tag-types", {
      method: "POST",
      headers: adminHeaders(),
      body: JSON.stringify({ key, label, description }),
    }),

  deleteTagType: (key: string) =>
    fetchJSON<{ ok: boolean }>(`/api/admin/tag-types/${key}`, {
      method: "DELETE",
      headers: adminHeaders(),
    }),

  sync: () =>
    fetchJSON<{ ok: boolean; output: string }>("/api/admin/sync", {
      method: "POST",
      headers: adminHeaders(),
    }),

  getAdminStatus: () =>
    fetchJSON<{
      sync: { last_sync: string | null; last_output: string; is_syncing: boolean };
      total_models: number;
      leaderboard_entries: number;
    }>("/api/admin/status", {
      headers: adminHeaders(),
    }),

  health: () => fetchJSON<{ status: string }>("/api/health"),
};
