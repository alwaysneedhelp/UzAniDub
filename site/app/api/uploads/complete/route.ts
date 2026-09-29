import { bindings, findUpload } from "../../../../lib/library";
import { mutationUser } from "../../../../lib/auth";

const PART_SIZE = 8 * 1024 * 1024;

export async function POST(request: Request) {
  const user = await mutationUser(request);
  if (user instanceof Response) return user;
  let body: { id?: string; parts?: { partNumber: number; etag: string }[] };
  try { body = await request.json(); } catch { return Response.json({ error: "Invalid JSON" }, { status: 400 }); }
  const upload = await findUpload(body.id ?? "");
  if (!upload || upload.owner_email !== user) return Response.json({ error: "Upload not found" }, { status: 404 });
  const count = Math.ceil(upload.bytes / PART_SIZE);
  if (!Array.isArray(body.parts) || body.parts.length !== count ||
      body.parts.some((part, index) => part.partNumber !== index + 1 || typeof part.etag !== "string" || !part.etag)) {
    return Response.json({ error: "Incomplete upload" }, { status: 400 });
  }
  const { DB, MEDIA } = bindings();
  const existing = await DB.prepare("SELECT id FROM episodes WHERE series_id = ? AND number = ?")
    .bind(upload.series_id, upload.number).first();
  if (existing) return Response.json({ error: "This episode number already exists" }, { status: 409 });
  const multipart = MEDIA.resumeMultipartUpload(upload.object_key, upload.upload_id);
  await multipart.complete(body.parts);
  try {
    await DB.batch([
      DB.prepare(
        `INSERT INTO episodes (id, series_id, number, title, object_key, content_type, bytes, thumb_key, duration)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).bind(upload.id, upload.series_id, upload.number, upload.title, upload.object_key, upload.content_type, upload.bytes,
        upload.thumb_key, upload.duration),
      DB.prepare("DELETE FROM uploads WHERE id = ?").bind(upload.id),
    ]);
  } catch (error) {
    await MEDIA.delete(upload.thumb_key ? [upload.object_key, upload.thumb_key] : upload.object_key);
    await DB.prepare("DELETE FROM uploads WHERE id = ?").bind(upload.id).run();
    throw error;
  }
  return Response.json({ episodeId: upload.id });
}
