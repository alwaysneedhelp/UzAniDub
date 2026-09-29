import { bindings, findSeries, validText } from "../../../../lib/library";
import { mutationUser } from "../../../../lib/auth";

const MAX_BYTES = 20 * 1024 * 1024 * 1024;

export async function POST(request: Request) {
  const user = await mutationUser(request);
  if (user instanceof Response) return user;
  let body: Record<string, unknown>;
  try { body = await request.json(); } catch { return Response.json({ error: "Invalid JSON" }, { status: 400 }); }
  const seriesId = validText(body.seriesId, 100);
  const title = validText(body.title, 100);
  const number = body.number;
  const bytes = body.bytes;
  const duration = typeof body.duration === "number" && Number.isFinite(body.duration) && body.duration > 0 && body.duration < 86400
    ? Math.round(body.duration) : null;
  const filename = typeof body.filename === "string" ? body.filename.toLowerCase() : "";
  const contentType = filename.endsWith(".mp4") ? "video/mp4" : filename.endsWith(".webm") ? "video/webm" : null;
  if (!seriesId || !title || !Number.isSafeInteger(number) || Number(number) < 1 || Number(number) > 10000 ||
      !Number.isSafeInteger(bytes) || Number(bytes) < 1 || Number(bytes) > MAX_BYTES || !contentType) {
    return Response.json({ error: "Enter a valid title, episode number, and MP4 or WebM file (up to 20 GB)" }, { status: 400 });
  }
  if (!await findSeries(seriesId)) return Response.json({ error: "Series not found" }, { status: 404 });
  const existing = await bindings().DB.prepare(
    "SELECT id FROM episodes WHERE series_id = ? AND number = ?",
  ).bind(seriesId, number).first();
  if (existing) return Response.json({ error: "This episode number already exists" }, { status: 409 });
  const id = crypto.randomUUID();
  const key = `episodes/${id}/${contentType === "video/mp4" ? "video.mp4" : "video.webm"}`;
  const { DB, MEDIA } = bindings();
  const upload = await MEDIA.createMultipartUpload(key, { httpMetadata: { contentType } });
  try {
    await DB.prepare(
      `INSERT INTO uploads (id, owner_email, series_id, number, title, object_key, upload_id, content_type, bytes, duration)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).bind(id, user, seriesId, number, title, key, upload.uploadId, contentType, bytes, duration).run();
  } catch (error) {
    await upload.abort();
    throw error;
  }
  return Response.json({ id, partSize: 8 * 1024 * 1024 }, { status: 201 });
}
