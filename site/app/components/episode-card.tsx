import Link from "next/link";
import type { Episode } from "../../lib/library";
import { formatDuration, timeAgo } from "../../lib/format";
import Poster from "./poster";

type Props = {
  episode: Episode;
  seriesId: string;
  seriesTitle: string;
  seriesHasCover: boolean;
  showSeries?: boolean;
};

export default function EpisodeCard({ episode, seriesId, seriesTitle, seriesHasCover, showSeries }: Props) {
  const duration = formatDuration(episode.duration);
  return <Link className="episode-card" href={`/watch/${episode.id}`}>
    <div className="episode-thumb">
      {episode.thumb_key
        ? <img src={`/api/thumb/${episode.id}`} alt="" loading="lazy" />
        : <Poster id={seriesId} title={seriesTitle} hasCover={seriesHasCover} />}
      {duration && <span className="thumb-badge">{duration}</span>}
    </div>
    {showSeries && <span className="episode-card-series">{seriesTitle}</span>}
    <h3>{episode.title}</h3>
    <span className="card-meta">Episode {episode.number} · {timeAgo(episode.created_at)}</span>
  </Link>;
}
