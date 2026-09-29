import { bindings, findSeries } from "../../../../../lib/library";
import { mutationUser } from "../../../../../lib/auth";

const MAX_COVER_BYTES = 5 * 1024 * 1024;

export async function PUT(request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await mutationUser(request);
  if (user instanceof Response) return user;
  const { id } = await context.params;
  const show = await findSeries(id);
  if (!show) return Response.json({ error: "Series not found" }, { status: 404 });
  const type = request.headers.get("content-type");
  const length = Number(request.headers.get("content-length"));
  if (!type || !["image/jpeg", "image/png", "image/webp"].includes(type) || !Number.isSafeInteger(length) || length < 1 || length > MAX_COVER_BYTES || !request.body) {
    return Response.json({ error: "Choose a JPEG, PNG, or WebP image under 5 MB" }, { status: 400 });
  }
  const key = `covers/${id}/${crypto.randomUUID()}`;
  const { DB, MEDIA } = bindings();
  await MEDIA.put(key, request.body, { httpMetadata: { contentType: type } });
  try {
    await DB.prepare("UPDATE series SET cover_key = ? WHERE id = ?").bind(key, id).run();
  } catch (error) {
    await MEDIA.delete(key);
    throw error;
  }
  if (show.cover_key) await MEDIA.delete(show.cover_key);
  return Response.json({ ok: true });
}
