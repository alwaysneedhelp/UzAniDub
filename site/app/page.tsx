import { Play, UploadSimple } from "@phosphor-icons/react/ssr";
import Link from "next/link";
import { listLatestEpisodes, listSeries, splitGenres } from "../lib/library";
import { episodeLabel } from "../lib/format";
import ContinueWatching from "./components/continue-watching";
import EpisodeCard from "./components/episode-card";
import Poster from "./components/poster";
import Rail from "./components/rail";
import SeriesCard from "./components/series-card";

export const dynamic = "force-dynamic";

export default async function Home() {
  const [popular, latest] = await Promise.all([listSeries({ sort: "popular" }), listLatestEpisodes(16)]);

  if (!popular.length) return <div className="empty-page page-shell">
    <h1>No anime yet</h1>
    <p>Anime you upload will show up here for everyone to watch.</p>
    <Link className="btn btn-primary" href="/upload?tab=series"><UploadSimple size={18} weight="bold" /> Upload</Link>
  </div>;

  const watchable = popular.filter((show) => show.episode_count > 0);
  const featured = watchable.find((show) => show.cover_key) ?? watchable[0];

  return <>
    {featured && <div className="feature-band"><section className="feature page-shell">
      <Link className="feature-cover" href={`/series/${featured.id}`} tabIndex={-1} aria-hidden="true">
        <Poster id={featured.id} title={featured.title} hasCover={!!featured.cover_key} eager />
      </Link>
      <div className="feature-info">
        <p className="feature-kicker">Uzbek dub</p>
        <h1><Link href={`/series/${featured.id}`}>{featured.title}</Link></h1>
        <p className="feature-meta">{splitGenres(featured.genre).join(", ")}<span>{episodeLabel(featured.episode_count)}</span></p>
        {featured.synopsis && <p className="feature-synopsis">{featured.synopsis}</p>}
        <div className="actions">
          {featured.first_episode_id && <Link className="btn btn-primary" href={`/watch/${featured.first_episode_id}`}><Play size={18} weight="fill" /> Watch episode 1</Link>}
          <Link className="btn btn-secondary" href={`/series/${featured.id}`}>All episodes</Link>
        </div>
      </div>
    </section></div>}

    <ContinueWatching />

    {latest.length > 0 && <Rail title="New episodes" variant="episode">
      {latest.map((episode) => <EpisodeCard key={episode.id} episode={episode} seriesId={episode.series_id}
        seriesTitle={episode.series_title} seriesHasCover={!!episode.series_cover_key} showSeries />)}
    </Rail>}

    <section className="page-shell home-grid">
      <div className="section-head">
        <h2>Popular</h2>
        {popular.length > 12 && <Link className="text-link" href="/browse">See all</Link>}
      </div>
      <div className="poster-grid">{popular.slice(0, 12).map((show) => <SeriesCard show={show} key={show.id} />)}</div>
    </section>
  </>;
}
