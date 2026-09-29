/** Per-browser watch progress, used for resuming and the "Continue watching" row. */
export type HistoryEntry = {
  episodeId: string;
  seriesId: string;
  seriesTitle: string;
  number: number;
  title: string;
  hasThumb: boolean;
  hasCover: boolean;
  time: number;
  duration: number;
  updatedAt: number;
};

const KEY = "uzanidub:history";
const LIMIT = 100;

/** The raw stored string: a stable snapshot for useSyncExternalStore. */
export function historySnapshot(): string {
  try {
    return localStorage.getItem(KEY) ?? "";
  } catch {
    return "";
  }
}

export function subscribeHistory(onChange: () => void): () => void {
  const listener = (event: StorageEvent) => { if (event.key === KEY) onChange(); };
  window.addEventListener("storage", listener);
  return () => window.removeEventListener("storage", listener);
}

export function parseHistory(raw: string): HistoryEntry[] {
  try {
    const value = JSON.parse(raw || "[]") as unknown;
    return Array.isArray(value) ? value as HistoryEntry[] : [];
  } catch {
    return [];
  }
}

export function readHistory(): HistoryEntry[] {
  return parseHistory(historySnapshot());
}

export function findProgress(episodeId: string): HistoryEntry | undefined {
  return readHistory().find((entry) => entry.episodeId === episodeId);
}

export function saveProgress(entry: HistoryEntry): void {
  try {
    const rest = readHistory().filter((item) => item.episodeId !== entry.episodeId);
    localStorage.setItem(KEY, JSON.stringify([entry, ...rest].slice(0, LIMIT)));
  } catch {
    // Storage can be unavailable (private mode, blocked site data); progress is a convenience.
  }
}

export function isFinished(entry: Pick<HistoryEntry, "time" | "duration">): boolean {
  return entry.duration > 0 && entry.time / entry.duration > 0.95;
}
