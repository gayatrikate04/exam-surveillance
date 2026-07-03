import { NextResponse } from "next/server";
import { appendAudit } from "@/lib/audit-store";
import { getSessionFromCookies } from "@/lib/session";

const LEVELS = new Set(["info", "warn", "error"]);

export async function POST(req: Request) {
  const session = await getSessionFromCookies();
  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }
  if (!body || typeof body !== "object") {
    return NextResponse.json({ error: "Invalid body" }, { status: 400 });
  }
  const b = body as Record<string, unknown>;
  const action = typeof b.action === "string" ? b.action.trim() : "";
  const message = typeof b.message === "string" ? b.message.trim() : "";
  if (!action || action.length > 120 || !message || message.length > 2000) {
    return NextResponse.json({ error: "Invalid action or message" }, { status: 400 });
  }
  const levelRaw = typeof b.level === "string" ? b.level : "info";
  const level = LEVELS.has(levelRaw) ? (levelRaw as "info" | "warn" | "error") : "info";
  const meta =
    b.meta && typeof b.meta === "object" && !Array.isArray(b.meta)
      ? (b.meta as Record<string, unknown>)
      : undefined;

  appendAudit({
    level,
    action,
    message,
    actorUserId: session.userId,
    actorEmail: session.email,
    meta,
  });
  return NextResponse.json({ ok: true });
}
