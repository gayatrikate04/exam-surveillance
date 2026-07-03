import { SignJWT, jwtVerify } from "jose";
import { isAdminEmail } from "@/lib/admin-role";

export const AUTH_COOKIE = "exam-monitor-auth";

export type SessionPayload = {
  userId: string;
  email: string;
  role: "admin" | "user";
};

function getSecretKey() {
  const s = process.env.AUTH_SECRET;
  if (s && s.length >= 16) return new TextEncoder().encode(s);
  if (process.env.NODE_ENV === "production") {
    throw new Error("AUTH_SECRET must be set (min 16 chars) in production");
  }
  return new TextEncoder().encode("dev-only-insecure-auth-secret");
}

export async function signSessionToken(
  userId: string,
  email: string,
  role: "admin" | "user",
) {
  return new SignJWT({ email, role })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(userId)
    .setIssuedAt()
    .setExpirationTime("7d")
    .sign(getSecretKey());
}

export async function verifySessionToken(token: string): Promise<SessionPayload | null> {
  const { payload } = await jwtVerify(token, getSecretKey());
  const userId = typeof payload.sub === "string" ? payload.sub : null;
  const email = typeof payload.email === "string" ? payload.email : null;
  if (!userId || !email) return null;
  const jwtRole = payload.role === "admin" ? "admin" : "user";
  const role = isAdminEmail(email) ? "admin" : jwtRole;
  return { userId, email, role };
}
