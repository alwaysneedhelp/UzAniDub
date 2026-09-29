import Link from "next/link";

export default function NotFound() {
  return <div className="empty-page page-shell">
    <h1>Page not found</h1>
    <p>This anime or episode doesn’t exist, or it was removed.</p>
    <div className="actions">
      <Link className="btn btn-primary" href="/">Go home</Link>
      <Link className="btn btn-secondary" href="/browse">Browse anime</Link>
    </div>
  </div>;
}
