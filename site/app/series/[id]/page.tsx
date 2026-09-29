import { Play, UploadSimple } from "@phosphor-icons/react/ssr";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { findSeries, listEpisodes, listSeries, splitGenres } from "../../../lib/library";
import { episodeLabel, formatCount } from "../../../lib/format";
import EpisodeCard from "../../components/episode-card";
import Poster from "../../components/poster";
import Rail from "../../components/rail";
import SeriesCard from "../../components/series-card";

export const dynamic = "force-dynamic";

type Props = { params: Promise<{ id: string }>; searchParams: Promise<{ episode?: string; order?: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const show = await findSeries((await params).id);
  return show ? { title: show.title, description: show.synopsis || undefined } : {};
}

export default async function SeriesPage({ params, searchParams }: Props) {
  const { id } = await params;
  const query = await searchParams;
  // Links from the previous design pointed at ?episode=<id>; episodes now play on their own page.
  if (query.episode) redirect(`/watch/${encodeURIComponent(query.episode)}`);
  const show = await findSeries(id);
  if (!show) notFound();
  const genres = splitGenres(show.genre);
  const [episodes, related] = await Promise.all([
    listEpisodes(id),
    genres[0] ? listSeries({ genre: genres[0], sort: "popular", limit: 21 }) : Promise.resolve([]),
  ]);
  const descending = query.order === "desc";
  const ordered = descending ? [...episodes].reverse() : episodes;
  const more = related.filter((item) => item.id !== id).slice(0, 20);

  return <>
    <section className="series-head page-shell">
      <div className="series-cover"><Poster id={show.id} title={show.title} hasCover={!!show.cover_key} eager /></div>
      <div className="series-info">
        <h1>{show.title}</h1>
        <p className="series-meta">
          <span>{episodeLabel(episodes.length)}</span>
          <span>{formatCount(show.views)} {show.views === 1 ? "view" : "views"}</span>
        </p>
        {genres.length > 0 && <p className="series-genres">
          {genres.map((genre, index) => <span key={genre}>{index > 0 && ", "}<Link href={`/browse?genre=${encodeURIComponent(genre)}`}>{genre}</Link></span>)}
        </p>}
        {show.synopsis && <p className="series-synopsis">{show.synopsis}</p>}
        <div className="actions">
          {episodes[0]
            ? <Link className="btn btn-primary" href={`/watch/${episodes[0].id}`}><Play size={18} weight="fill" /> Watch episode {episodes[0].number}</Link>
            : <Link className="btn btn-primary" href={`/upload?series=${show.id}`}><UploadSimple size={18} weight="bold" /> Upload episode 1</Link>}
        </div>
      </div>
    </section>

    <section className="page-shell episodes">
      <div className="section-head">
        <h2>Episodes <span className="count">{episodes.length}</span></h2>
        {episodes.length > 1 && <nav className="segmented" aria-label="Episode order">
          <Link className={descending ? "" : "active"} href={`/series/${id}`}>Oldest first</Link>
          <Link className={descending ? "active" : ""} href={`/series/${id}?order=desc`}>Newest first</Link>
        </nav>}
      </div>
      {ordered.length
        ? <div className="episode-grid">{ordered.map((episode) => <EpisodeCard key={episode.id} episode={episode}
          seriesId={show.id} seriesTitle={show.title} seriesHasCover={!!show.cover_key} />)}</div>
        : <div className="empty-note">No episodes yet.</div>}
    </section>

    {more.length > 0 && <Rail title="More like this">{more.map((item) => <SeriesCard show={item} key={item.id} />)}</Rail>}
  </>;
}
