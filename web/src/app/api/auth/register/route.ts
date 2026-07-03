import { NextResponse } from "next/server";
import { isAdminEmail } from "@/lib/admin-role";
import { appendAudit } from "@/lib/audit-store";
import {
  appendUserCsv,
  hashPassword,
  newUserId,
  normalizeEmail,
  readUsers,
  writeUsers,
} from "@/lib/auth-store";
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
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return NextResponse.json({ error: "Invalid email" }, { status: 400 });
  }
  if (password.length < 6) {
    return NextResponse.json(
      { error: "Password must be at least 6 characters" },
      { status: 400 },
    );
  }
  const users = readUsers();
  if (users.some((u) => u.email === email)) {
    return NextResponse.json({ error: "Email already registered" }, { status: 409 });
  }
  const { salt, hash } = hashPassword(password);
  const id = newUserId();
  const role = isAdminEmail(email) ? "admin" : "user";
  const newUser = {
    id,
    email,
    passwordHash: hash,
    salt,
    createdAt: new Date().toISOString(),
    role,
  };
  users.push(newUser);
  writeUsers(users);
  appendUserCsv(newUser);
  const token = await signSessionToken(id, email, role);
  appendAudit({
    level: "info",
    action: "auth.register",
    message: "New account registered",
    actorUserId: id,
    actorEmail: email,
    meta: { role },
  });
  const res = NextResponse.json({ ok: true, user: { id, email, role } });
  res.cookies.set(AUTH_COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    secure: process.env.NODE_ENV === "production",
    maxAge: 60 * 60 * 24 * 7,
  });
  return res;
}
