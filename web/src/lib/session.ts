import { cookies } from "next/headers";
import { AUTH_COOKIE, verifySessionToken } from "@/lib/auth-token";

export { AUTH_COOKIE } from "@/lib/auth-token";

export async function getSessionFromCookies() {
  const jar = await cookies();
  const token = jar.get(AUTH_COOKIE)?.value;
  if (!token) return null;
  try {
    return await verifySessionToken(token);
  } catch {
    return null;
  }
}
