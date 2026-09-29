import type { Metadata } from "next";
import Link from "next/link";
import { listGenres, listSeries, type SeriesSort } from "../../lib/library";
import SeriesCard from "../components/series-card";

export const dynamic = "force-dynamic";

const SORTS: { value: SeriesSort; label: string }[] = [
  { value: "popular", label: "Popular" },
  { value: "newest", label: "New" },
  { value: "alphabetical", label: "A-Z" },
];

type Params = { sort?: string; genre?: string };

export async function generateMetadata({ searchParams }: { searchParams: Promise<Params> }): Promise<Metadata> {
  const { sort, genre } = await searchParams;
  return { title: genre ?? SORTS.find((item) => item.value === sort)?.label ?? "Popular" };
}

export default async function BrowsePage({ searchParams }: { searchParams: Promise<Params> }) {
  const query = await searchParams;
  const sort = SORTS.find((item) => item.value === query.sort)?.value ?? "popular";
  const genre = query.genre?.trim().slice(0, 30) || undefined;
  const [shows, genres] = await Promise.all([listSeries({ sort, genre }), listGenres()]);
  const href = (next: Params) => {
    const params = new URLSearchParams();
    if (next.sort && next.sort !== "popular") params.set("sort", next.sort);
    if (next.genre) params.set("genre", next.genre);
    const text = params.toString();
    return text ? `/browse?${text}` : "/browse";
  };

  return <div className="page page-shell">
    <div className="page-head">
      <h1>{genre ?? SORTS.find((item) => item.value === sort)!.label}</h1>
      <nav className="segmented" aria-label="Sort">
        {SORTS.map((item) => <Link key={item.value} className={item.value === sort ? "active" : ""} aria-current={item.value === sort ? "page" : undefined}
          href={href({ sort: item.value, genre })}>{item.label}</Link>)}
      </nav>
    </div>
    {genres.length > 0 && <nav className="chips" aria-label="Genres">
      <Link className={genre ? "" : "active"} href={href({ sort })}>All genres</Link>
      {genres.map((item) => <Link key={item} className={item.toLowerCase() === genre?.toLowerCase() ? "active" : ""}
        href={href({ sort, genre: item })}>{item}</Link>)}
    </nav>}
    {shows.length
      ? <div className="poster-grid">{shows.map((show) => <SeriesCard show={show} key={show.id} />)}</div>
      : <div className="empty-note">Nothing in this genre yet. <Link href={href({ sort })}>Show all genres</Link></div>}
  </div>;
}
