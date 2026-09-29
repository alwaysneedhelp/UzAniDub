import { bindings, sameOrigin } from "../../../../../lib/library";

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  if (!sameOrigin(request)) return Response.json({ error: "Invalid request origin" }, { status: 403 });
  const { id } = await context.params;
  const result = await bindings().DB.prepare("UPDATE episodes SET views = views + 1 WHERE id = ?").bind(id).run();
  if (!result.meta.changes) return Response.json({ error: "Episode not found" }, { status: 404 });
  return Response.json({ ok: true });
}
