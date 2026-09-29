"use client";

import { CaretLeft, CaretRight } from "@phosphor-icons/react/ssr";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

/** A horizontally scrolling row of cards; the arrows page through it. */
export default function Rail({ title, href, variant = "poster", children }: {
  title: string; href?: string; variant?: "poster" | "episode"; children: React.ReactNode;
}) {
  const track = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ start: true, end: true });

  const measure = useCallback(() => {
    const el = track.current;
    if (!el) return;
    setEdges({ start: el.scrollLeft < 4, end: el.scrollLeft + el.clientWidth >= el.scrollWidth - 4 });
  }, []);

  useEffect(() => {
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [measure]);

  const page = (direction: 1 | -1) => {
    const el = track.current;
    if (el) el.scrollBy({ left: direction * el.clientWidth * 0.85, behavior: "smooth" });
  };

  return <section className={`rail rail-${variant}`}>
    <div className="section-head page-shell">
      <h2>{title}</h2>
      <div className="section-tools">
        {href && <Link className="text-link" href={href}>See all</Link>}
        {!(edges.start && edges.end) && <div className="rail-arrows">
          <button type="button" aria-label={`Scroll ${title} back`} disabled={edges.start} onClick={() => page(-1)}><CaretLeft size={16} weight="bold" /></button>
          <button type="button" aria-label={`Scroll ${title} forward`} disabled={edges.end} onClick={() => page(1)}><CaretRight size={16} weight="bold" /></button>
        </div>}
      </div>
    </div>
    <div className="rail-track" ref={track} onScroll={measure}>{children}</div>
  </section>;
}
