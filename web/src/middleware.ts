import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";
import { AUTH_COOKIE, verifySessionToken } from "@/lib/auth-token";

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  const token = req.cookies.get(AUTH_COOKIE)?.value;

  let session: { userId: string; email: string; role: "admin" | "user" } | null = null;
  if (token) {
    try {
      session = await verifySessionToken(token);
    } catch {
      session = null;
    }
  }

  if (pathname === "/login" || pathname === "/register") {
    if (session) return NextResponse.redirect(new URL("/", req.url));
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/auth/login") || pathname.startsWith("/api/auth/register")) {
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/auth/me") || pathname.startsWith("/api/auth/logout")) {
    return NextResponse.next();
  }

  if (pathname.startsWith("/api/")) {
    return NextResponse.next();
  }

  if (!session) {
    const res = NextResponse.redirect(new URL("/login", req.url));
    if (token) res.cookies.delete(AUTH_COOKIE);
    return res;
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)",
  ],
};
