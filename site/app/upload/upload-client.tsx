"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState, type DragEvent, type FormEvent } from "react";
import { formatDuration } from "../../lib/format";
import { Check, FilmStrip, Image as ImageIcon, UploadSimple } from "@phosphor-icons/react/ssr";
import Poster from "../components/poster";

type ShowOption = { id: string; title: string; hasCover: boolean; lastEpisode: number };
type Tab = "episode" | "series";
type Picked = { file: File; duration: number | null; thumb: Blob | null; thumbUrl: string | null };

const VIDEO_TYPES = ["video/mp4", "video/webm"];

async function jsonRequest<T>(path: string, data: unknown): Promise<T> {
  const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const result = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}

function once(target: EventTarget, event: string, timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    const done = (ok: boolean) => { clearTimeout(timer); target.removeEventListener(event, onEvent); target.removeEventListener("error", onError); resolve(ok); };
    const onEvent = () => done(true);
    const onError = () => done(false);
    const timer = setTimeout(() => done(false), timeoutMs);
    target.addEventListener(event, onEvent);
    target.addEventListener("error", onError);
  });
}

/** Reads the duration and grabs a frame for the episode thumbnail. Either may be null if the browser can't decode the file. */
async function inspectVideo(file: File): Promise<Omit<Picked, "file">> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = url;
  try {
    if (!await once(video, "loadedmetadata", 15000)) return { duration: null, thumb: null, thumbUrl: null };
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : null;
    if (!video.videoWidth || !duration) return { duration, thumb: null, thumbUrl: null };
    video.currentTime = Math.min(Math.max(duration * 0.2, 1), Math.max(duration - 1, 0));
    if (!await once(video, "seeked", 15000)) return { duration, thumb: null, thumbUrl: null };
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = Math.round(640 * video.videoHeight / video.videoWidth);
    canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
    const thumb = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.82));
    return { duration, thumb, thumbUrl: thumb ? URL.createObjectURL(thumb) : null };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

export default function UploadClient({ initialTab, initialSeriesId, initialSeries, genres }: {
  initialTab: Tab; initialSeriesId?: string; initialSeries: ShowOption[]; genres: string[];
}) {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>(initialSeries.length ? initialTab : "series");
  const [shows, setShows] = useState(initialSeries);
  const [seriesId, setSeriesId] = useState(initialSeries.find((show) => show.id === initialSeriesId)?.id ?? initialSeries[0]?.id ?? "");
  const selected = shows.find((show) => show.id === seriesId);

  // Episode form
  const [picked, setPicked] = useState<Picked | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [dragging, setDragging] = useState(false);
  // The episode number defaults to the one after the selected anime's last episode until the user edits it.
  const [numberEdit, setNumberEdit] = useState<{ seriesId: string; value: string } | null>(null);
  const number = numberEdit?.seriesId === seriesId ? numberEdit.value : String((selected?.lastEpisode ?? 0) + 1);
  const setNumber = (value: string) => setNumberEdit({ seriesId, value });
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState<{ tone: "info" | "error" | "success"; text: string } | null>(null);
  const cancelled = useRef(false);
  // Set once the episode is published, so leaving for the watch page doesn't trigger the unsaved-upload warning.
  const finished = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);

  // Series form
  const [coverUrl, setCoverUrl] = useState<string | null>(null);
  const [genreText, setGenreText] = useState("");
  const [savingSeries, setSavingSeries] = useState(false);
  const [seriesMessage, setSeriesMessage] = useState<{ tone: "error" | "success"; text: string } | null>(null);

  useEffect(() => () => { if (picked?.thumbUrl) URL.revokeObjectURL(picked.thumbUrl); }, [picked]);
  useEffect(() => () => { if (coverUrl) URL.revokeObjectURL(coverUrl); }, [coverUrl]);
  useEffect(() => {
    if (!busy) return;
    const warn = (event: BeforeUnloadEvent) => { if (!finished.current) event.preventDefault(); };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [busy]);

  async function pickFile(file: File | undefined) {
    if (!file) return;
    const name = file.name.toLowerCase();
    if (!VIDEO_TYPES.includes(file.type) && !name.endsWith(".mp4") && !name.endsWith(".webm")) {
      setMessage({ tone: "error", text: "Choose an MP4 or WebM video." });
      return;
    }
    if (file.size > 20 * 1024 ** 3) { setMessage({ tone: "error", text: "Videos can be up to 20 GB." }); return; }
    setMessage(null);
    setInspecting(true);
    setPicked({ file, duration: null, thumb: null, thumbUrl: null });
    const details = await inspectVideo(file).catch(() => ({ duration: null, thumb: null, thumbUrl: null }));
    setPicked({ file, ...details });
    setInspecting(false);
  }

  function onDrop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    if (!busy) void pickFile(event.dataTransfer.files[0]);
  }

  async function uploadEpisode(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    if (!picked) { setMessage({ tone: "error", text: "Choose the dubbed video first." }); return; }
    if (!data.get("rights")) { setMessage({ tone: "error", text: "Confirm that you have permission to share this video." }); return; }
    setBusy(true); setProgress(0); cancelled.current = false;
    setMessage({ tone: "info", text: "Preparing upload…" });
    let uploadId = "";
    try {
      const file = picked.file;
      const start = await jsonRequest<{ id: string; partSize: number }>("/api/uploads/start", {
        seriesId, title: data.get("title"), number: Number(number), filename: file.name, bytes: file.size, duration: picked.duration,
      });
      uploadId = start.id;
      if (picked.thumb) {
        await fetch(`/api/uploads/thumbnail?id=${encodeURIComponent(uploadId)}`, { method: "PUT", headers: { "Content-Type": "image/jpeg" }, body: picked.thumb })
          .catch(() => undefined);
      }
      const parts: { partNumber: number; etag: string }[] = [];
      const count = Math.ceil(file.size / start.partSize);
      const began = Date.now();
      for (let index = 0; index < count; index++) {
        if (cancelled.current) throw new Error("Upload cancelled.");
        const partNumber = index + 1;
        const chunk = file.slice(index * start.partSize, Math.min(file.size, partNumber * start.partSize));
        const response = await fetch(`/api/uploads/part?id=${encodeURIComponent(uploadId)}&part=${partNumber}`, { method: "PUT", body: chunk });
        const result = await response.json() as { error?: string; partNumber: number; etag: string };
        if (!response.ok) throw new Error(result.error || `Part ${partNumber} failed`);
        parts.push(result);
        const sent = Math.min(file.size, partNumber * start.partSize);
        const speed = sent / Math.max(1, (Date.now() - began) / 1000);
        setProgress(Math.round(sent / file.size * 100));
        setMessage({ tone: "info", text: `${formatBytes(sent)} of ${formatBytes(file.size)} uploaded (${formatBytes(speed)}/s)` });
      }
      setMessage({ tone: "info", text: "Finishing up…" });
      const completed = await jsonRequest<{ episodeId: string }>("/api/uploads/complete", { id: uploadId, parts });
      setProgress(100);
      setMessage({ tone: "success", text: "Published. Opening the episode…" });
      finished.current = true;
      router.push(`/watch/${completed.episodeId}`);
    } catch (error) {
      setMessage({ tone: "error", text: error instanceof Error ? error.message : "Upload failed" });
      if (uploadId) await jsonRequest("/api/uploads/abort", { id: uploadId }).catch(() => {});
      setBusy(false);
    }
  }

  async function addSeries(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    const title = String(data.get("title") || "").trim();
    setSavingSeries(true); setSeriesMessage(null);
    try {
      const result = await jsonRequest<{ id: string }>("/api/series", { title, genre: genreText, synopsis: data.get("synopsis") });
      const cover = data.get("cover");
      let coverSaved = false;
      let coverProblem = "";
      if (cover instanceof File && cover.size) {
        const uploaded = await fetch(`/api/series/${result.id}/cover`, { method: "PUT", headers: { "Content-Type": cover.type }, body: cover });
        coverSaved = uploaded.ok;
        if (!uploaded.ok) coverProblem = (await uploaded.json().catch(() => ({})) as { error?: string }).error || "upload failed";
      }
      setShows((old) => [...old, { id: result.id, title, hasCover: coverSaved, lastEpisode: 0 }].sort((a, b) => a.title.localeCompare(b.title)));
      setSeriesId(result.id);
      form.reset(); setGenreText(""); setCoverUrl(null);
      if (coverProblem) {
        setSeriesMessage({ tone: "error", text: `${title} was added, but its cover was not saved: ${coverProblem}` });
      } else {
        setTab("episode");
        setMessage({ tone: "success", text: `${title} added. Now upload its first episode.` });
      }
    } catch (error) {
      setSeriesMessage({ tone: "error", text: error instanceof Error ? error.message : "Could not add this anime" });
    } finally { setSavingSeries(false); }
  }

  const currentGenres = genreText.split(",").map((genre) => genre.trim().toLowerCase()).filter(Boolean);
  const addGenre = (genre: string) => {
    if (currentGenres.includes(genre.toLowerCase())) return;
    setGenreText((text) => [...text.split(",").map((part) => part.trim()).filter(Boolean), genre].join(", "));
  };

  return <div className="upload-layout">
    <div className="segmented upload-tabs" role="tablist">
      <button type="button" role="tab" aria-selected={tab === "episode"} className={tab === "episode" ? "active" : ""} onClick={() => setTab("episode")} disabled={busy}>
        Episode
      </button>
      <button type="button" role="tab" aria-selected={tab === "series"} className={tab === "series" ? "active" : ""} onClick={() => setTab("series")} disabled={busy}>
        New anime
      </button>
    </div>

    {tab === "episode" && <form className="upload-card" onSubmit={uploadEpisode}>
      {!shows.length
        ? <div className="empty-state"><h2>Add an anime first</h2><p>Episodes belong to an anime. <button type="button" className="link-button" onClick={() => setTab("series")}>Add a new anime</button>.</p></div>
        : <div className="upload-grid">
          <div className="upload-fields">
            <div className="field">
              <label htmlFor="upload-series">Anime</label>
              <div className="series-select">
                {selected?.hasCover && <span className="series-select-poster"><Poster id={selected.id} title={selected.title} hasCover={selected.hasCover} /></span>}
                <select id="upload-series" value={seriesId} onChange={(event) => setSeriesId(event.target.value)} disabled={busy} required>
                  {shows.map((show) => <option key={show.id} value={show.id}>{show.title}</option>)}
                </select>
              </div>
              <small>Not in the list? <button type="button" className="link-button" onClick={() => setTab("series")} disabled={busy}>Add a new anime</button></small>
            </div>
            <div className="field-row">
              <div className="field narrow">
                <label htmlFor="episode-number">Episode</label>
                <input id="episode-number" type="number" min="1" max="10000" required value={number} onChange={(event) => setNumber(event.target.value)} disabled={busy} />
              </div>
              <div className="field">
                <label htmlFor="episode-title">Episode title</label>
                <input id="episode-title" name="title" maxLength={100} required placeholder="The Beginning" disabled={busy} />
              </div>
            </div>
            {selected && selected.lastEpisode > 0 && <p className="field-hint">Latest uploaded episode: {selected.lastEpisode}</p>}
            <label className="check-row"><input type="checkbox" name="rights" required disabled={busy} /><span>I have the rights to share this video and its audio.</span></label>
          </div>

          <div className={`dropzone${dragging ? " dragging" : ""}${picked ? " filled" : ""}`}
            onDragOver={(event) => { event.preventDefault(); if (!busy) setDragging(true); }}
            onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <input ref={fileInput} id="episode-video" type="file" accept=".mp4,.webm,video/mp4,video/webm" hidden
              onChange={(event) => { void pickFile(event.target.files?.[0]); event.target.value = ""; }} />
            {picked
              ? <div className="picked">
                <div className="picked-thumb">
                  {picked.thumbUrl ? <img src={picked.thumbUrl} alt="Episode thumbnail taken from the video" /> : <FilmStrip size={36} />}
                  {inspecting && <span className="picked-status">Reading the video…</span>}
                  {picked.duration && <span className="thumb-badge">{formatDuration(picked.duration)}</span>}
                </div>
                <div className="picked-info">
                  <strong title={picked.file.name}>{picked.file.name}</strong>
                  <span>{formatBytes(picked.file.size)}{!inspecting && !picked.thumb ? ", no preview in this browser" : ""}</span>
                  {!busy && <button type="button" className="link-button" onClick={() => fileInput.current?.click()}>Choose another file</button>}
                </div>
              </div>
              : <button type="button" className="dropzone-empty" onClick={() => fileInput.current?.click()}>
                <UploadSimple size={32} />
                <strong>Drop the dubbed video here, or choose a file</strong>
                <span>MP4 or WebM, up to 20 GB</span>
              </button>}
          </div>
        </div>}

      {shows.length > 0 && <div className="upload-footer">
        {busy && <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress} aria-label="Upload progress"><span style={{ width: `${progress}%` }} /></div>}
        {message && <p className={`form-message ${message.tone}`} role="status">{message.tone === "success" && <Check size={16} weight="bold" />}{message.text}</p>}
        <div className="upload-buttons">
          {busy && <button type="button" className="btn btn-secondary" onClick={() => { cancelled.current = true; }}>Cancel</button>}
          <button className="btn btn-primary" type="submit" disabled={busy || inspecting || !picked}>
            <UploadSimple size={18} weight="bold" /> {busy ? `Uploading ${progress}%` : "Publish episode"}
          </button>
        </div>
      </div>}
    </form>}

    {tab === "series" && <form className="upload-card" onSubmit={addSeries}>
      <div className="upload-grid series-form">
        <div className="upload-fields">
          <div className="field">
            <label htmlFor="series-title">Title</label>
            <input id="series-title" name="title" required maxLength={100} placeholder="Naruto" disabled={savingSeries} />
          </div>
          <div className="field">
            <label htmlFor="series-genre">Genres</label>
            <input id="series-genre" value={genreText} onChange={(event) => setGenreText(event.target.value)} maxLength={200}
              placeholder="Action, Adventure, Shounen" disabled={savingSeries} />
            <small>Separate genres with commas.</small>
            {genres.length > 0 && <div className="genre-suggest">{genres.map((genre) => <button key={genre} type="button"
              className={currentGenres.includes(genre.toLowerCase()) ? "active" : ""} onClick={() => addGenre(genre)}>{genre}</button>)}</div>}
          </div>
          <div className="field">
            <label htmlFor="series-synopsis">Description</label>
            <textarea id="series-synopsis" name="synopsis" maxLength={1000} rows={5} placeholder="What is this anime about?" disabled={savingSeries} />
          </div>
        </div>
        <label className="cover-picker">
          <input name="cover" type="file" accept="image/jpeg,image/png,image/webp" disabled={savingSeries}
            onChange={(event) => { const file = event.target.files?.[0]; setCoverUrl(file ? URL.createObjectURL(file) : null); }} />
          {coverUrl ? <img src={coverUrl} alt="Cover preview" /> : <span className="cover-empty"><ImageIcon size={32} /><strong>Add a cover</strong><small>Portrait JPEG, PNG or WebP, up to 5 MB</small></span>}
        </label>
      </div>
      <div className="upload-footer">
        {seriesMessage && <p className={`form-message ${seriesMessage.tone}`} role="status">{seriesMessage.text}</p>}
        <div className="upload-buttons"><button className="btn btn-primary" type="submit" disabled={savingSeries}>{savingSeries ? "Saving…" : "Add anime"}</button></div>
      </div>
    </form>}
  </div>;
}
