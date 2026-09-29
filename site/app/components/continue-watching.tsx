"use client";

import Link from "next/link";
import { useMemo, useSyncExternalStore } from "react";
import { formatDuration } from "../../lib/format";
import { historySnapshot, isFinished, parseHistory, subscribeHistory } from "../../lib/history";
import Poster from "./poster";
import Rail from "./rail";

export default function ContinueWatching() {
  const raw = useSyncExternalStore(subscribeHistory, historySnapshot, () => "");
  const entries = useMemo(() => {
    // History is newest first, so the first entry per series is the one last watched.
    const seen = new Set<string>();
    const latest = parseHistory(raw).filter((entry) => !seen.has(entry.seriesId) && seen.add(entry.seriesId));
    return latest.filter((entry) => !isFinished(entry)).slice(0, 12);
  }, [raw]);
  if (!entries.length) return null;
  return <Rail title="Continue watching" variant="episode">
    {entries.map((entry) => {
      const left = formatDuration(entry.duration - entry.time);
      return <Link className="episode-card" href={`/watch/${entry.episodeId}`} key={entry.episodeId}>
        <div className="episode-thumb">
          {entry.hasThumb
            ? <img src={`/api/thumb/${entry.episodeId}`} alt="" loading="lazy" />
            : <Poster id={entry.seriesId} title={entry.seriesTitle} hasCover={entry.hasCover} />}
          <span className="thumb-progress"><span style={{ width: `${Math.min(100, entry.time / entry.duration * 100)}%` }} /></span>
        </div>
        <span className="episode-card-series">{entry.seriesTitle}</span>
        <h3>{entry.title}</h3>
        <span className="card-meta">Episode {entry.number}{left ? ` · ${left} left` : ""}</span>
      </Link>;
    })}
  </Rail>;
}
