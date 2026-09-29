import { CaretDown, List, MagnifyingGlass, UploadSimple, X } from "@phosphor-icons/react/ssr";
import Link from "next/link";
import { listGenres } from "../../lib/library";

export function Wordmark() {
  return <Link className="wordmark" href="/" aria-label="UzAniDub home">UzAniDub</Link>;
}

export default async function SiteHeader() {
  const genres = await listGenres().catch(() => []);
  return <header className="site-header">
    <div className="header-inner">
      <details className="mobile-menu">
        <summary aria-label="Menu"><List size={22} className="menu-open" /><X size={22} className="menu-close" /></summary>
        <nav className="mobile-menu-panel" aria-label="Mobile navigation">
          <Link href="/browse">Popular</Link>
          <Link href="/browse?sort=newest">New</Link>
          <Link href="/browse?sort=alphabetical">A-Z</Link>
          {genres.length > 0 && <p className="menu-heading">Genres</p>}
          {genres.map((genre) => <Link key={genre} href={`/browse?genre=${encodeURIComponent(genre)}`}>{genre}</Link>)}
        </nav>
      </details>
      <Wordmark />
      <nav className="header-nav" aria-label="Main navigation">
        <Link href="/browse" className="nav-link">Popular</Link>
        <Link href="/browse?sort=newest" className="nav-link">New</Link>
        {genres.length > 0 && <div className="nav-dropdown">
          <Link href="/browse" className="nav-link">Genres <CaretDown size={14} weight="bold" /></Link>
          <div className="nav-dropdown-panel">
            {genres.map((genre) => <Link key={genre} href={`/browse?genre=${encodeURIComponent(genre)}`}>{genre}</Link>)}
          </div>
        </div>}
      </nav>
      <div className="header-actions">
        <form className="header-search" action="/search" role="search">
          <MagnifyingGlass size={18} />
          <input name="q" type="search" placeholder="Search anime" aria-label="Search anime" />
        </form>
        <Link className="icon-button search-link" href="/search" aria-label="Search"><MagnifyingGlass size={22} /></Link>
        <Link className="btn btn-primary btn-small" href="/upload"><UploadSimple size={18} weight="bold" /><span className="upload-label">Upload</span></Link>
      </div>
    </div>
  </header>;
}
