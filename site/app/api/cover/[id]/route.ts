import { bindings, findSeries } from "../../../../lib/library";

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const show = await findSeries(id);
  if (!show?.cover_key) return new Response("Cover not found", { status: 404 });
  const object = await bindings().MEDIA.get(show.cover_key);
  if (!object) return new Response("Cover not found", { status: 404 });
  return new Response(object.body, { headers: {
    "Content-Type": object.httpMetadata?.contentType ?? "image/jpeg",
    "Cache-Control": "public, max-age=3600",
    "X-Content-Type-Options": "nosniff",
  } });
}
