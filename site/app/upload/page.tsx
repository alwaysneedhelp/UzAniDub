import type { Metadata } from "next";
import Link from "next/link";
import { chatGPTSignInPath, getChatGPTUser } from "../chatgpt-auth";
import { lastEpisodeNumbers, listGenres, listSeries } from "../../lib/library";
import UploadClient from "./upload-client";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Upload" };

export default async function UploadPage({ searchParams }: { searchParams: Promise<{ tab?: string; series?: string }> }) {
  const query = await searchParams;
  const user = await getChatGPTUser();
  const returnTo = `/upload${query.tab ? `?tab=${encodeURIComponent(query.tab)}` : query.series ? `?series=${encodeURIComponent(query.series)}` : ""}`;

  if (!user) return <div className="page page-shell">
    <div className="page-head"><h1>Upload</h1></div>
    <div className="signin-card">
      <h2>Sign in to upload</h2>
      <p>Watching doesn’t need an account. Uploading does, so every episode has an owner.</p>
      <Link className="btn btn-primary" href={chatGPTSignInPath(returnTo)}>Sign in</Link>
    </div>
  </div>;

  const [shows, lastNumbers, genres] = await Promise.all([listSeries({ sort: "alphabetical" }), lastEpisodeNumbers(), listGenres()]);
  return <div className="page page-shell">
    <div className="page-head"><h1>Upload</h1><p className="page-note">Signed in as {user.displayName}</p></div>
    <UploadClient
      initialTab={query.tab === "series" ? "series" : "episode"}
      initialSeriesId={query.series}
      genres={genres}
      initialSeries={shows.map((show) => ({ id: show.id, title: show.title, hasCover: !!show.cover_key, lastEpisode: lastNumbers[show.id] ?? 0 }))}
    />
  </div>;
}
