import { bindings, findUpload } from "../../../../lib/library";
import { mutationUser } from "../../../../lib/auth";

const PART_SIZE = 8 * 1024 * 1024;

export async function PUT(request: Request) {
  const user = await mutationUser(request);
  if (user instanceof Response) return user;
  const url = new URL(request.url);
  const id = url.searchParams.get("id") ?? "";
  const partNumber = Number(url.searchParams.get("part"));
  const upload = await findUpload(id);
  if (!upload || upload.owner_email !== user) return Response.json({ error: "Upload not found" }, { status: 404 });
  const partCount = Math.ceil(upload.bytes / PART_SIZE);
  if (!Number.isInteger(partNumber) || partNumber < 1 || partNumber > partCount || !request.body) {
    return Response.json({ error: "Invalid upload part" }, { status: 400 });
  }
  const expected = partNumber === partCount ? upload.bytes - PART_SIZE * (partCount - 1) : PART_SIZE;
  if (Number(request.headers.get("content-length")) !== expected) {
    return Response.json({ error: "Unexpected part size" }, { status: 400 });
  }
  const multipart = bindings().MEDIA.resumeMultipartUpload(upload.object_key, upload.upload_id);
  const part = await multipart.uploadPart(partNumber, request.body);
  return Response.json(part);
}
