import { bindings, findUpload } from "../../../../lib/library";
import { mutationUser } from "../../../../lib/auth";

export async function POST(request: Request) {
  const user = await mutationUser(request);
  if (user instanceof Response) return user;
  let body: { id?: string };
  try { body = await request.json(); } catch { return Response.json({ error: "Invalid JSON" }, { status: 400 }); }
  const upload = await findUpload(body.id ?? "");
  if (!upload || upload.owner_email !== user) return Response.json({ error: "Upload not found" }, { status: 404 });
  const { DB, MEDIA } = bindings();
  await MEDIA.resumeMultipartUpload(upload.object_key, upload.upload_id).abort();
  if (upload.thumb_key) await MEDIA.delete(upload.thumb_key);
  await DB.prepare("DELETE FROM uploads WHERE id = ?").bind(upload.id).run();
  return Response.json({ ok: true });
}
