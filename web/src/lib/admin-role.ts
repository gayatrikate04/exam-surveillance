import type { StoredUser } from "@/lib/auth-types";
import { normalizeEmail } from "@/lib/email-normalize";

/** Comma-separated admin emails (e.g. `ADMIN_EMAILS=you@school.edu`). */
export function isAdminEmail(email: string) {
  const raw = process.env.ADMIN_EMAILS ?? "";
  const parts = raw
    .split(/[,;\s]+/)
    .map((e) => normalizeEmail(e))
    .filter(Boolean);
  return new Set(parts).has(normalizeEmail(email));
}

export function resolveUserRole(user: Pick<StoredUser, "email" | "role">) {
  if (user.role === "admin" || isAdminEmail(user.email)) return "admin" as const;
  return "user" as const;
}

export function sessionIsAdmin(session: { email: string; role: "admin" | "user" }) {
  return session.role === "admin";
}
