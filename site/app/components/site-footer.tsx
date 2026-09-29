import Link from "next/link";
import { Wordmark } from "./site-header";

export default function SiteFooter() {
  return <footer className="site-footer">
    <div className="page-shell footer-inner">
      <div>
        <Wordmark />
        <p>Uzbek dubbed anime, free to watch.</p>
      </div>
      <nav aria-label="Footer">
        <Link href="/browse">Popular</Link>
        <Link href="/browse?sort=newest">New</Link>
        <Link href="/browse?sort=alphabetical">A-Z</Link>
        <Link href="/upload">Upload</Link>
      </nav>
    </div>
    <div className="page-shell footer-note">
      <span>© {new Date().getUTCFullYear()} UzAniDub</span>
      <span>Upload only videos you have the rights to share.</span>
    </div>
  </footer>;
}
