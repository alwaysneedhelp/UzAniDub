import Link from "next/link";
import type { Series } from "../../lib/library";
import { episodeLabel } from "../../lib/format";
import Poster from "./poster";

export default function SeriesCard({ show }: { show: Series }) {
  return <Link className="series-card" href={`/series/${show.id}`}>
    <div className="series-card-cover"><Poster id={show.id} title={show.title} hasCover={!!show.cover_key} /></div>
    <h3>{show.title}</h3>
    <span className="card-meta">{episodeLabel(show.episode_count)}</span>
  </Link>;
}
