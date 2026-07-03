import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { AUTH_COOKIE, verifySessionToken } from "@/lib/auth-token";

export async function GET() {
  const jar = await cookies();
  const token = jar.get(AUTH_COOKIE)?.value;
  if (!token) {
    return NextResponse.json({ user: null });
  }
  try {
    const session = await verifySessionToken(token);
    if (!session) return NextResponse.json({ user: null });
    return NextResponse.json({
      user: { id: session.userId, email: session.email, role: session.role },
    });
  } catch {
    return NextResponse.json({ user: null });
  }
}
