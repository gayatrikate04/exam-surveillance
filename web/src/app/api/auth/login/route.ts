import { NextResponse } from "next/server";
import { resolveUserRole } from "@/lib/admin-role";
import { appendAudit } from "@/lib/audit-store";
import { normalizeEmail, readUsers, verifyPassword } from "@/lib/auth-store";
import { AUTH_COOKIE, signSessionToken } from "@/lib/auth-token";

export async function POST(req: Request) {
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
  const email = typeof b.email === "string" ? normalizeEmail(b.email) : "";
  const password = typeof b.password === "string" ? b.password : "";
  if (!email || !password) {
    return NextResponse.json({ error: "Email and password required" }, { status: 400 });
  }
  const users = readUsers();
  const user = users.find((u) => u.email === email);
  if (!user || !verifyPassword(password, user)) {
    appendAudit({
      level: "warn",
      action: "auth.login_failed",
      message: "Invalid email or password",
      actorEmail: email,
    });
    return NextResponse.json({ error: "Invalid email or password" }, { status: 401 });
  }
  const role = resolveUserRole(user);
  const token = await signSessionToken(user.id, user.email, role);
  appendAudit({
    level: "info",
    action: "auth.login",
    message: "Signed in",
    actorUserId: user.id,
    actorEmail: user.email,
  });
  const res = NextResponse.json({ ok: true, user: { id: user.id, email: user.email, role } });
  res.cookies.set(AUTH_COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    secure: process.env.NODE_ENV === "production",
    maxAge: 60 * 60 * 24 * 7,
  });
  return res;
}
