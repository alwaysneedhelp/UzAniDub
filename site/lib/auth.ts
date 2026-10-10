import { sameOrigin } from "./library";

/**
 * Uploading is open: anyone can add anime and episodes without an account.
 * Uploads are still tied to one owner label so an in-progress upload can only
 * be continued, completed or aborted by the same flow that started it (the
 * upload id is an unguessable UUID).
 */
const GUEST_OWNER = "guest";

export async function mutationUser(request: Request): Promise<string | Response> {
  if (!sameOrigin(request)) return Response.json({ error: "Invalid request origin" }, { status: 403 });
  return GUEST_OWNER;
}
