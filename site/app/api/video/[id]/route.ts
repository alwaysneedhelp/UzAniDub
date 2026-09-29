import { bindings, findEpisode } from "../../../../lib/library";

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const episode = await findEpisode(id);
  if (!episode) return new Response("Episode not found", { status: 404 });
  const { MEDIA } = bindings();
  const range = request.headers.get("range");
  const size = episode.bytes;
  let offset = 0;
  let length = size;
  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (!match || (!match[1] && !match[2])) return invalidRange(size);
    if (match[1]) {
      offset = Number(match[1]);
      const end = match[2] ? Number(match[2]) : size - 1;
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(end) || offset >= size || end < offset) return invalidRange(size);
      length = Math.min(end, size - 1) - offset + 1;
    } else {
      const suffix = Number(match[2]);
      if (!Number.isSafeInteger(suffix) || suffix < 1) return invalidRange(size);
      length = Math.min(suffix, size);
      offset = size - length;
    }
  }
  const object = await MEDIA.get(episode.object_key, range ? { range: { offset, length } } : undefined);
  if (!object) return new Response("Video file missing", { status: 404 });
  const headers = new Headers({
    "Content-Type": episode.content_type,
    "Content-Length": String(length),
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
  });
  if (range) headers.set("Content-Range", `bytes ${offset}-${offset + length - 1}/${size}`);
  return new Response(object.body, { status: range ? 206 : 200, headers });
}

function invalidRange(size: number): Response {
  return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
}
