import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number, decimals = 2): string {
  return n.toFixed(decimals);
}

export function formatWithSE(value: number, se: number | null, decimals = 2): string {
  const v = value.toFixed(decimals);
  if (se !== null && se !== undefined && se > 0) {
    return `${v} ± ${se.toFixed(decimals)}`;
  }
  return v;
}

export const ERROR_TYPE_LABELS: Record<string, string> = {
  incorrect_agreement: "Согласование",
  other_language_insert: "Иноязычные вставки",
  syntax: "Синтаксис",
  calque: "Кальки",
  made_up_words: "Выдуманные слова",
  wrong_capitalization: "Капитализация",
  tautology: "Тавтология",
  grammatical_gender_change: "Смена рода",
  other: "Другое",
};

export const ERROR_TYPE_COLORS: Record<string, string> = {
  incorrect_agreement: "#ef4444",
  other_language_insert: "#f97316",
  syntax: "#f59e0b",
  calque: "#84cc16",
  made_up_words: "#22c55e",
  wrong_capitalization: "#06b6d4",
  tautology: "#3b82f6",
  grammatical_gender_change: "#8b5cf6",
  other: "#6b7280",
};

export const LEVEL_COLORS = {
  3: "#dc2626",
  2: "#d97706",
  1: "#ca8a04",
} as const;

export const LEVEL_BG_COLORS = {
  3: "rgba(220, 38, 38, 0.12)",
  2: "rgba(217, 119, 6, 0.12)",
  1: "rgba(202, 138, 4, 0.10)",
} as const;
