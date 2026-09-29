"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { formatDuration } from "../../../lib/format";
import { findProgress, isFinished, saveProgress } from "../../../lib/history";
import { Play } from "@phosphor-icons/react/ssr";

type Props = {
  episode: { id: string; number: number; title: string; hasThumb: boolean };
  series: { id: string; title: string; hasCover: boolean };
  next: { id: string; number: number; title: string } | null;
};

const UP_NEXT_SECONDS = 10;

export default function Player({ episode, series, next }: Props) {
  const router = useRouter();
  const video = useRef<HTMLVideoElement>(null);
  const lastSaved = useRef(0);
  const counted = useRef(false);
  const [resumedAt, setResumedAt] = useState<number | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);

  function save() {
    const el = video.current;
    if (!el || !el.duration || !Number.isFinite(el.duration)) return;
    lastSaved.current = Date.now();
    saveProgress({
      episodeId: episode.id, seriesId: series.id, seriesTitle: series.title, number: episode.number, title: episode.title,
      hasThumb: episode.hasThumb, hasCover: series.hasCover, time: el.currentTime, duration: el.duration, updatedAt: Date.now(),
    });
  }

  function resume() {
    const el = video.current;
    const saved = findProgress(episode.id);
    if (el && saved && saved.time > 10 && !isFinished(saved)) {
      el.currentTime = saved.time;
      setResumedAt(saved.time);
    }
  }

  function countView() {
    if (counted.current) return;
    counted.current = true;
    const key = `uzanidub:viewed:${episode.id}`;
    try {
      if (sessionStorage.getItem(key)) return;
      sessionStorage.setItem(key, "1");
    } catch { /* storage unavailable: still count once per page view */ }
    fetch(`/api/episodes/${episode.id}/view`, { method: "POST" }).catch(() => {});
  }

  useEffect(() => {
    const el = video.current;
    // The metadata may already be loaded before hydration attached our handler.
    if (el && el.readyState >= 1) resume();
    const onLeave = () => save();
    window.addEventListener("pagehide", onLeave);
    return () => { onLeave(); window.removeEventListener("pagehide", onLeave); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [episode.id]);

  useEffect(() => {
    if (resumedAt === null) return;
    const timer = window.setTimeout(() => setResumedAt(null), 6000);
    return () => window.clearTimeout(timer);
  }, [resumedAt]);

  useEffect(() => {
    if (countdown === null || !next) return;
    if (countdown <= 0) { router.push(`/watch/${next.id}`); return; }
    const timer = window.setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => window.clearTimeout(timer);
  }, [countdown, next, router]);

  return <div className="player">
    <video
      ref={video}
      src={`/api/video/${episode.id}`}
      poster={episode.hasThumb ? `/api/thumb/${episode.id}` : undefined}
      controls
      autoPlay
      playsInline
      preload="metadata"
      aria-label={`${series.title} episode ${episode.number}: ${episode.title}`}
      onLoadedMetadata={resume}
      onPlay={() => { countView(); setCountdown(null); }}
      onPause={save}
      onTimeUpdate={() => { if (Date.now() - lastSaved.current > 5000) save(); }}
      onEnded={() => { save(); if (next) setCountdown(UP_NEXT_SECONDS); }}
    />
    {resumedAt !== null && <div className="player-toast" role="status">
      Resumed at {formatDuration(resumedAt)}
      <button type="button" onClick={() => { if (video.current) video.current.currentTime = 0; setResumedAt(null); }}>Start over</button>
    </div>}
    {countdown !== null && next && <div className="up-next" role="dialog" aria-label="Up next">
      <span className="up-next-label">Episode {next.number} starts in {countdown}s</span>
      <strong>{next.title}</strong>
      <div className="up-next-actions">
        <button type="button" className="btn btn-primary" onClick={() => router.push(`/watch/${next.id}`)}><Play size={16} weight="fill" /> Play now</button>
        <button type="button" className="btn btn-secondary" onClick={() => setCountdown(null)}>Cancel</button>
      </div>
    </div>}
  </div>;
}
