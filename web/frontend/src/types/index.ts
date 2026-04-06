export interface ErrorTypes {
  incorrect_agreement: number;
  other_language_insert: number;
  syntax: number;
  calque: number;
  made_up_words: number;
  wrong_capitalization: number;
  tautology: number;
  grammatical_gender_change: number;
  other: number;
}

export interface ErrorTypesByLevel {
  [type: string]: { "1": number; "2": number; "3": number };
}

export interface LeaderboardEntry {
  rank: number;
  model_id: string;
  display_name: string;
  original_model: string;
  judge_model: string;
  dataset: string;
  num_runs: number;
  l3_per_1k: number;
  l3_per_1k_se: number | null;
  l2_per_1k: number;
  l2_per_1k_se: number | null;
  l1_per_1k: number;
  l1_per_1k_se: number | null;
  weighted_per_1k: number;
  weighted_per_1k_se: number | null;
  weighted_per_avg: number;
  weighted_per_avg_se: number | null;
  avg_tokens: number;
  total_tokens: number;
  error_types: ErrorTypes;
  error_types_by_level: ErrorTypesByLevel;
  files: string[];
  tags: Record<string, string>;
  enabled?: boolean;
}

export interface Mistake {
  position: number[];
  level: 1 | 2 | 3;
  type: string;
  explanation: string;
}

export interface DialogResult {
  dialog_id: number;
  dialog: { role: string; content: string }[];
  answer: string;
  tokens: number;
  error: string | null;
  mistakes: Mistake[];
  mistakes_count: { "1": number; "2": number; "3": number };
  splitted_answer: string;
}

export interface RunSummary {
  run_index: number;
  file: string;
  timestamp: string;
  run_number: number;
  dataset: string;
  judge_model: string;
  total_tokens: number;
  avg_tokens: number;
  total_dialogs: number;
  successful_dialogs: number;
  level_3: number;
  level_2: number;
  level_1: number;
  l3_per_1k: number;
  l2_per_1k: number;
  l1_per_1k: number;
  weighted_per_1k: number;
  weighted_per_avg: number;
  error_types: ErrorTypes;
  error_types_by_level: ErrorTypesByLevel;
}

export interface ModelResults {
  model_id: string;
  display_name: string;
  tags: Record<string, string>;
  aggregated: LeaderboardEntry;
  runs: RunSummary[];
}

export interface RunResults {
  model_id: string;
  display_name: string;
  run_index: number;
  file: string;
  timestamp: string;
  dataset: string;
  judge_model: string;
  results: DialogResult[];
}

export interface CombinedResults {
  model_id: string;
  display_name: string;
  num_runs: number;
  results: {
    dialog_id: number;
    dialog: { role: string; content: string }[];
    runs: {
      run_index: number;
      answer: string;
      tokens: number;
      error: string | null;
      mistakes: Mistake[];
      mistakes_count: { "1": number; "2": number; "3": number };
      splitted_answer: string;
    }[];
  }[];
}

export interface TagType {
  key: string;
  label: string;
  description: string;
}
