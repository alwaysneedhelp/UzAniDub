import { bindings, listSeries, normalizeGenres, validText } from "../../../lib/library";
import { mutationUser } from "../../../lib/auth";

export async function GET() {
  return Response.json({ series: await listSeries() });
}

export async function POST(request: Request) {
  const user = await mutationUser(request);
  if (user instanceof Response) return user;
  let body: Record<string, unknown>;
  try { body = await request.json(); } catch { return Response.json({ error: "Invalid JSON" }, { status: 400 }); }
  const title = validText(body.title, 100);
  const synopsis = typeof body.synopsis === "string" ? body.synopsis.trim().slice(0, 1000) : "";
  const genre = normalizeGenres(body.genre);
  if (!title) return Response.json({ error: "Enter a series title" }, { status: 400 });
  const id = crypto.randomUUID();
  await bindings().DB.prepare(
    "INSERT INTO series (id, title, synopsis, genre) VALUES (?, ?, ?, ?)",
  ).bind(id, title, synopsis, genre).run();
  return Response.json({ id }, { status: 201 });
}
