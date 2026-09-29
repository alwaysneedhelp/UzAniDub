import { bindings, findUpload } from "../../../../lib/library";
import { mutationUser } from "../../../../lib/auth";

const MAX_THUMB_BYTES = 2 * 1024 * 1024;

export async function PUT(request: Request) {
  const user = await mutationUser(request);
  if (user instanceof Response) return user;
  const upload = await findUpload(new URL(request.url).searchParams.get("id") ?? "");
  if (!upload || upload.owner_email !== user) return Response.json({ error: "Upload not found" }, { status: 404 });
  const type = request.headers.get("content-type");
  const length = Number(request.headers.get("content-length"));
  if (!type || !["image/jpeg", "image/webp"].includes(type) || !Number.isSafeInteger(length) || length < 1 || length > MAX_THUMB_BYTES || !request.body) {
    return Response.json({ error: "Thumbnail must be a JPEG or WebP image under 2 MB" }, { status: 400 });
  }
  const key = `episodes/${upload.id}/thumb`;
  const { DB, MEDIA } = bindings();
  await MEDIA.put(key, request.body, { httpMetadata: { contentType: type } });
  await DB.prepare("UPDATE uploads SET thumb_key = ? WHERE id = ?").bind(key, upload.id).run();
  return Response.json({ ok: true });
}
