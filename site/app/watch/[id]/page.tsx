import { CaretLeft, CaretRight } from "@phosphor-icons/react/ssr";
import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { findEpisode, findSeries, listEpisodes } from "../../../lib/library";
import { formatCount, formatDate, formatDuration } from "../../../lib/format";
import Poster from "../../components/poster";
import Player from "./player";

export const dynamic = "force-dynamic";

type Props = { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const episode = await findEpisode((await params).id);
  const show = episode && await findSeries(episode.series_id);
  return episode && show ? { title: `${show.title}, episode ${episode.number}` } : {};
}

export default async function WatchPage({ params }: Props) {
  const episode = await findEpisode((await params).id);
  if (!episode) notFound();
  const [show, episodes] = await Promise.all([findSeries(episode.series_id), listEpisodes(episode.series_id)]);
  if (!show) notFound();
  const index = episodes.findIndex((item) => item.id === episode.id);
  const previous = episodes[index - 1];
  const next = episodes[index + 1];
  const hasCover = !!show.cover_key;

  return <div className="watch">
    <div className="player-stage">
      <Player
        key={episode.id}
        episode={{ id: episode.id, number: episode.number, title: episode.title, hasThumb: !!episode.thumb_key }}
        series={{ id: show.id, title: show.title, hasCover }}
        next={next ? { id: next.id, number: next.number, title: next.title } : null}
      />
    </div>
    <div className="watch-body page-shell">
      <div className="watch-main">
        <Link className="watch-series" href={`/series/${show.id}`}>{show.title}</Link>
        <h1>{episode.title}</h1>
        <p className="watch-meta">
          <span>Episode {episode.number}</span>
          {episode.duration ? <span>{formatDuration(episode.duration)}</span> : null}
          <span>{formatDate(episode.created_at)}</span>
          <span>{formatCount(episode.views)} {episode.views === 1 ? "view" : "views"}</span>
        </p>
        {show.synopsis && <p className="watch-synopsis">{show.synopsis}</p>}
        {(previous || next) && <div className="watch-nav">
          {previous ? <Link className="btn btn-secondary" href={`/watch/${previous.id}`}><CaretLeft size={16} weight="bold" /> Episode {previous.number}</Link> : <span />}
          {next && <Link className="btn btn-primary" href={`/watch/${next.id}`}>Episode {next.number} <CaretRight size={16} weight="bold" /></Link>}
        </div>}
      </div>
      <aside className="watch-side" aria-labelledby="episode-list-title">
        <h2 id="episode-list-title">Episodes <span className="count">{episodes.length}</span></h2>
        <ol className="episode-list">
          {episodes.map((item) => <li key={item.id}>
            <Link href={`/watch/${item.id}`} className={item.id === episode.id ? "current" : ""} aria-current={item.id === episode.id ? "page" : undefined}>
              <span className="episode-list-thumb">
                {item.thumb_key ? <img src={`/api/thumb/${item.id}`} alt="" loading="lazy" /> : <Poster id={show.id} title={show.title} hasCover={hasCover} />}
                {item.duration ? <small>{formatDuration(item.duration)}</small> : null}
              </span>
              <span className="episode-list-text">
                <span>{item.id === episode.id ? "Playing now" : `Episode ${item.number}`}</span>
                <strong>{item.title}</strong>
              </span>
            </Link>
          </li>)}
        </ol>
      </aside>
    </div>
  </div>;
}
