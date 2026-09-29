import { bindings, findEpisode } from "../../../../lib/library";

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const episode = await findEpisode(id);
  if (!episode?.thumb_key) return new Response("Thumbnail not found", { status: 404 });
  const object = await bindings().MEDIA.get(episode.thumb_key);
  if (!object) return new Response("Thumbnail not found", { status: 404 });
  return new Response(object.body, { headers: {
    "Content-Type": object.httpMetadata?.contentType ?? "image/jpeg",
    "Cache-Control": "public, max-age=86400",
    "X-Content-Type-Options": "nosniff",
  } });
}
