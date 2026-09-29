/** A series cover, or a plain title card when no cover was uploaded. */
export default function Poster({ id, title, hasCover, eager }: { id: string; title: string; hasCover: boolean; eager?: boolean }) {
  if (hasCover) return <img className="poster-img" src={`/api/cover/${id}`} alt="" loading={eager ? "eager" : "lazy"} />;
  return <div className="poster-blank"><span>{title}</span></div>;
}
