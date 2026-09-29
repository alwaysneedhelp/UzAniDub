import { getChatGPTUser } from "../app/chatgpt-auth";
import { sameOrigin } from "./library";

export async function mutationUser(request: Request): Promise<string | Response> {
  if (!sameOrigin(request)) return Response.json({ error: "Invalid request origin" }, { status: 403 });
  const user = await getChatGPTUser();
  if (!user) return Response.json({ error: "Sign in to upload" }, { status: 401 });
  return user.email;
}
