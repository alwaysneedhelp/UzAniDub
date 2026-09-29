export function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export function formatDuration(seconds: number | null | undefined): string {
  if (!seconds || seconds < 1) return "";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function formatCount(value: number): string {
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0).replace(/\.0$/, "")}K`;
  return `${(value / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`;
}

export function episodeLabel(count: number): string {
  return `${count} ${count === 1 ? "episode" : "episodes"}`;
}

/** D1 stores CURRENT_TIMESTAMP as "YYYY-MM-DD HH:MM:SS" in UTC. */
export function parseTimestamp(value: string): Date {
  return new Date(value.includes("T") ? value : `${value.replace(" ", "T")}Z`);
}

export function timeAgo(value: string, now = Date.now()): string {
  const seconds = Math.max(0, (now - parseTimestamp(value).getTime()) / 1000);
  const units: [number, string][] = [[31_536_000, "year"], [2_592_000, "month"], [604_800, "week"], [86_400, "day"], [3600, "hour"], [60, "minute"]];
  for (const [size, name] of units) {
    const amount = Math.floor(seconds / size);
    if (amount >= 1) return `${amount} ${name}${amount === 1 ? "" : "s"} ago`;
  }
  return "Just now";
}

export function formatDate(value: string): string {
  return parseTimestamp(value).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}

/** A stable hue per series so cover-less posters still look distinct. */
export function posterHue(seed: string): number {
  let hash = 0;
  for (const char of seed) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return hash % 360;
}
