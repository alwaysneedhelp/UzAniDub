import { MagnifyingGlass } from "@phosphor-icons/react/ssr";
import type { Metadata } from "next";
import Link from "next/link";
import { listSeries } from "../../lib/library";
import SeriesCard from "../components/series-card";

export const dynamic = "force-dynamic";

export async function generateMetadata({ searchParams }: { searchParams: Promise<{ q?: string }> }): Promise<Metadata> {
  const { q } = await searchParams;
  return { title: q?.trim() ? `Search: ${q.trim()}` : "Search" };
}

export default async function SearchPage({ searchParams }: { searchParams: Promise<{ q?: string }> }) {
  const q = (await searchParams).q?.trim().slice(0, 100) ?? "";
  const results = q ? await listSeries({ query: q, sort: "popular" }) : [];
  return <div className="page page-shell">
    <form className="search-box" action="/search" role="search">
      <MagnifyingGlass size={22} />
      <input name="q" type="search" defaultValue={q} placeholder="Search by title or genre" aria-label="Search anime" autoFocus />
    </form>
    {q && (results.length
      ? <>
        <p className="result-note">{results.length} {results.length === 1 ? "result" : "results"} for “{q}”</p>
        <div className="poster-grid">{results.map((show) => <SeriesCard show={show} key={show.id} />)}</div>
      </>
      : <div className="empty-note">Nothing matches “{q}”. <Link href="/browse">Browse all anime</Link></div>)}
  </div>;
}
