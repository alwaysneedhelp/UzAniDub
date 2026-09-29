import { env } from "cloudflare:workers";

export type Series = {
  id: string; title: string; synopsis: string; genre: string; cover_key: string | null; created_at: string;
  episode_count: number; views: number; first_episode_id: string | null; latest_episode_at: string | null;
};
export type Episode = {
  id: string; series_id: string; number: number; title: string; object_key: string; content_type: string;
  bytes: number; thumb_key: string | null; duration: number | null; views: number; created_at: string;
};
export type LatestEpisode = Episode & { series_title: string; series_cover_key: string | null };
export type Upload = {
  id: string; owner_email: string; series_id: string; number: number; title: string; object_key: string;
  upload_id: string; content_type: string; bytes: number; thumb_key: string | null; duration: number | null;
};
export type SeriesSort = "popular" | "newest" | "alphabetical";

type Bindings = { DB: D1Database; MEDIA: R2Bucket };

export function bindings(): Bindings {
  const value = env as unknown as Bindings;
  if (!value.DB || !value.MEDIA) throw new Error("Site storage is not configured");
  return value;
}

const SERIES_SELECT = `SELECT s.*, COUNT(e.id) AS episode_count, COALESCE(SUM(e.views), 0) AS views,
  (SELECT id FROM episodes f WHERE f.series_id = s.id ORDER BY f.number ASC LIMIT 1) AS first_episode_id,
  MAX(e.created_at) AS latest_episode_at
  FROM series s LEFT JOIN episodes e ON e.series_id = s.id`;

const ORDER: Record<SeriesSort, string> = {
  popular: "views DESC, latest_episode_at DESC, s.title COLLATE NOCASE ASC",
  newest: "COALESCE(latest_episode_at, s.created_at) DESC, s.title COLLATE NOCASE ASC",
  alphabetical: "s.title COLLATE NOCASE ASC",
};

export async function listSeries(options: { sort?: SeriesSort; genre?: string; query?: string; limit?: number } = {}): Promise<Series[]> {
  const where: string[] = [];
  const params: unknown[] = [];
  if (options.genre) {
    where.push("(',' || REPLACE(LOWER(s.genre), ', ', ',') || ',') LIKE ? ESCAPE '!'");
    params.push(`%,${escapeLike(options.genre.toLowerCase())},%`);
  }
  if (options.query) {
    where.push("(s.title LIKE ? ESCAPE '!' OR s.synopsis LIKE ? ESCAPE '!' OR s.genre LIKE ? ESCAPE '!')");
    const like = `%${escapeLike(options.query)}%`;
    params.push(like, like, like);
  }
  const sql = `${SERIES_SELECT} ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
    GROUP BY s.id ORDER BY ${ORDER[options.sort ?? "newest"]} ${options.limit ? `LIMIT ${Math.floor(options.limit)}` : ""}`;
  const { results } = await bindings().DB.prepare(sql).bind(...params).all<Series>();
  return results;
}

export async function findSeries(id: string): Promise<Series | null> {
  return bindings().DB.prepare(`${SERIES_SELECT} WHERE s.id = ? GROUP BY s.id`).bind(id).first<Series>();
}

export async function listEpisodes(seriesId: string): Promise<Episode[]> {
  const { results } = await bindings().DB.prepare(
    "SELECT * FROM episodes WHERE series_id = ? ORDER BY number ASC",
  ).bind(seriesId).all<Episode>();
  return results;
}

export async function listLatestEpisodes(limit: number): Promise<LatestEpisode[]> {
  const { results } = await bindings().DB.prepare(
    `SELECT e.*, s.title AS series_title, s.cover_key AS series_cover_key FROM episodes e
     JOIN series s ON s.id = e.series_id ORDER BY e.created_at DESC, e.number DESC LIMIT ?`,
  ).bind(limit).all<LatestEpisode>();
  return results;
}

export async function listGenres(): Promise<string[]> {
  const { results } = await bindings().DB.prepare("SELECT genre FROM series").all<{ genre: string }>();
  const seen = new Map<string, string>();
  for (const row of results) for (const genre of splitGenres(row.genre)) {
    if (!seen.has(genre.toLowerCase())) seen.set(genre.toLowerCase(), genre);
  }
  return [...seen.values()].sort((a, b) => a.localeCompare(b));
}

export async function lastEpisodeNumbers(): Promise<Record<string, number>> {
  const { results } = await bindings().DB.prepare(
    "SELECT series_id, MAX(number) AS last FROM episodes GROUP BY series_id",
  ).all<{ series_id: string; last: number }>();
  return Object.fromEntries(results.map((row) => [row.series_id, row.last]));
}

export async function findEpisode(id: string): Promise<Episode | null> {
  return bindings().DB.prepare("SELECT * FROM episodes WHERE id = ?").bind(id).first<Episode>();
}

export async function findUpload(id: string): Promise<Upload | null> {
  return bindings().DB.prepare("SELECT * FROM uploads WHERE id = ?").bind(id).first<Upload>();
}

function escapeLike(value: string): string {
  return value.replace(/[!%_]/g, (char) => `!${char}`);
}

export function normalizeGenres(value: unknown): string {
  const genres = typeof value === "string" ? splitGenres(value).slice(0, 6).map((genre) => genre.slice(0, 30)) : [];
  return genres.length ? genres.join(", ") : "Anime";
}

export function splitGenres(genre: string): string[] {
  return genre.split(",").map((part) => part.trim()).filter(Boolean);
}

export function validText(value: unknown, max: number): string | null {
  if (typeof value !== "string") return null;
  const text = value.trim();
  return text.length > 0 && text.length <= max ? text : null;
}

export function sameOrigin(request: Request): boolean {
  return request.headers.get("origin") === new URL(request.url).origin;
}
