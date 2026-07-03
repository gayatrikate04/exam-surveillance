import { NextResponse } from "next/server";
import { sessionIsAdmin } from "@/lib/admin-role";
import { computeAdminDashboardStats } from "@/lib/audit-stats";
import { readAudit } from "@/lib/audit-store";
import { getSessionFromCookies } from "@/lib/session";

export async function GET() {
  const session = await getSessionFromCookies();
  if (!session || !sessionIsAdmin(session)) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }
  const entries = readAudit();
  const slice = entries.slice(0, 800);
  const alerts = slice.filter((e) => e.level === "warn" || e.level === "error");
  const stats = computeAdminDashboardStats(entries);
  return NextResponse.json({
    entries: slice,
    alerts,
    stats,
    meta: { historyReturned: slice.length, totalStored: stats.totalStored },
  });
}
