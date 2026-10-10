import type { Metadata } from "next";
import { lastEpisodeNumbers, listGenres, listSeries } from "../../lib/library";
import UploadClient from "./upload-client";

export const dynamic = "force-dynamic";
export const metadata: Metadata = { title: "Upload" };

export default async function UploadPage({ searchParams }: { searchParams: Promise<{ tab?: string; series?: string }> }) {
  const query = await searchParams;
  const [shows, lastNumbers, genres] = await Promise.all([listSeries({ sort: "alphabetical" }), lastEpisodeNumbers(), listGenres()]);
  return <div className="page page-shell">
    <div className="page-head"><h1>Upload</h1><p className="page-note">No account needed</p></div>
    <UploadClient
      initialTab={query.tab === "series" ? "series" : "episode"}
      initialSeriesId={query.series}
      genres={genres}
      initialSeries={shows.map((show) => ({ id: show.id, title: show.title, hasCover: !!show.cover_key, lastEpisode: lastNumbers[show.id] ?? 0 }))}
    />
  </div>;
}
