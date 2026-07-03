import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { appendAudit } from "@/lib/audit-store";
import { AUTH_COOKIE, verifySessionToken } from "@/lib/auth-token";

export async function POST() {
  const jar = await cookies();
  const token = jar.get(AUTH_COOKIE)?.value;
  if (token) {
    try {
      const s = await verifySessionToken(token);
      if (s) {
        appendAudit({
          level: "info",
          action: "auth.logout",
          message: "Signed out",
          actorUserId: s.userId,
          actorEmail: s.email,
        });
      }
    } catch {
      /* invalid token */
    }
  }
  const res = NextResponse.json({ ok: true });
  res.cookies.set(AUTH_COOKIE, "", {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    secure: process.env.NODE_ENV === "production",
    maxAge: 0,
  });
  return res;
}
