import type { AuditEntry } from "@/lib/audit-store";

export type HourlyPoint = { label: string; count: number };
export type ActionCount = { action: string; count: number };
export type LevelCount = { level: string; count: number };

export type LiveSeriesPoint = {
  time: string;
  people: number;
  suspicious: number;
  riskScore: number;
  back: number;
  phone: number;
};

export type LiveAlertRow = {
  id: string;
  ts: string;
  level: string;
  action: string;
  message: string;
};

export type AdminDashboardStats = {
  serverTime: string;
  totalStored: number;
  totals24h: { total: number; info: number; warn: number; error: number };
  hourly24hUtc: HourlyPoint[];
  byLevel: LevelCount[];
  topActions: ActionCount[];
  rates: { m1: number; m5: number; m15: number };
  /** Chronological points from `app.live.status` audit rows (for charts). */
  liveSeries: LiveSeriesPoint[];
  /** Recent live / detection warnings & errors for the dashboard strip. */
  liveAlertsRecent: LiveAlertRow[];
};

function riskToScore(r: unknown): number {
  const x = String(r ?? "low").toLowerCase();
  if (x === "high") return 2;
  if (x === "medium") return 1;
  return 0;
}

function numMeta(m: Record<string, unknown> | undefined, key: string): number {
  if (!m) return 0;
  const v = m[key];
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isNaN(n) ? 0 : n;
  }
  return 0;
}

function buildLiveSeries(entries: AuditEntry[]): LiveSeriesPoint[] {
  const rows = entries.filter((e) => e.action === "app.live.status");
  const sorted = [...rows]
    .filter((e) => !Number.isNaN(parseTs(e.ts)))
    .sort((a, b) => parseTs(a.ts) - parseTs(b.ts))
    .slice(-120);
  return sorted.map((e) => {
    const meta = e.meta as Record<string, unknown> | undefined;
    const d = new Date(parseTs(e.ts));
    return {
      time: d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      people: numMeta(meta, "people"),
      suspicious: numMeta(meta, "suspicious"),
      riskScore: riskToScore(meta?.overall_risk),
      back: numMeta(meta, "back_turned"),
      phone: numMeta(meta, "phone_near_person"),
    };
  });
}

function buildLiveAlertsRecent(entries: AuditEntry[]): LiveAlertRow[] {
  const out: LiveAlertRow[] = [];
  for (const e of entries) {
    if (e.level !== "warn" && e.level !== "error") continue;
    const a = e.action || "";
    if (!a.startsWith("app.live") && !a.startsWith("app.detection")) continue;
    out.push({
      id: e.id,
      ts: e.ts,
      level: e.level,
      action: a,
      message: e.message.length > 160 ? `${e.message.slice(0, 157)}…` : e.message,
    });
    if (out.length >= 16) break;
  }
  return out;
}

function parseTs(ts: string): number {
  const n = Date.parse(ts);
  return Number.isNaN(n) ? NaN : n;
}

function floorHourUtc(ms: number): number {
  const u = new Date(ms);
  return Date.UTC(
    u.getUTCFullYear(),
    u.getUTCMonth(),
    u.getUTCDate(),
    u.getUTCHours(),
    0,
    0,
    0,
  );
}

export function computeAdminDashboardStats(allEntries: AuditEntry[]): AdminDashboardStats {
  const now = Date.now();
  const serverTime = new Date().toISOString();
  const totalStored = allEntries.length;
  const dayMs = 24 * 3600_000;

  const in24 = allEntries.filter((e) => {
    const t = parseTs(e.ts);
    return !Number.isNaN(t) && t >= now - dayMs && t <= now + 60_000;
  });

  const totals24h = { total: in24.length, info: 0, warn: 0, error: 0 };
  for (const e of in24) {
    if (e.level === "info") totals24h.info += 1;
    else if (e.level === "warn") totals24h.warn += 1;
    else if (e.level === "error") totals24h.error += 1;
  }

  const endHourUtc = floorHourUtc(now);
  const firstHourUtc = endHourUtc - 23 * 3600_000;
  const hourly24hUtc: HourlyPoint[] = Array.from({ length: 24 }, (_, i) => {
    const t = firstHourUtc + i * 3600_000;
    const d = new Date(t);
    const label = `${String(d.getUTCMonth() + 1).padStart(2, "0")}/${String(d.getUTCDate()).padStart(2, "0")} ${String(d.getUTCHours()).padStart(2, "0")}:00`;
    return { label, count: 0 };
  });

  for (const e of in24) {
    const et = parseTs(e.ts);
    if (Number.isNaN(et)) continue;
    const bucket = floorHourUtc(et);
    if (bucket < firstHourUtc || bucket > endHourUtc) continue;
    const idx = Math.floor((bucket - firstHourUtc) / 3600_000);
    if (idx >= 0 && idx < 24) hourly24hUtc[idx].count += 1;
  }

  const actionMap = new Map<string, number>();
  for (const e of in24) {
    const a = e.action || "unknown";
    actionMap.set(a, (actionMap.get(a) ?? 0) + 1);
  }
  const topActions: ActionCount[] = [...actionMap.entries()]
    .map(([action, count]) => ({ action, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  const byLevel: LevelCount[] = [
    { level: "info", count: totals24h.info },
    { level: "warn", count: totals24h.warn },
    { level: "error", count: totals24h.error },
  ];

  const countSince = (msAgo: number) =>
    allEntries.filter((e) => {
      const t = parseTs(e.ts);
      return !Number.isNaN(t) && t >= now - msAgo;
    }).length;

  const liveSeries = buildLiveSeries(allEntries);
  const liveAlertsRecent = buildLiveAlertsRecent(allEntries);

  return {
    serverTime,
    totalStored,
    totals24h,
    hourly24hUtc,
    byLevel,
    topActions,
    rates: {
      m1: countSince(60_000),
      m5: countSince(5 * 60_000),
      m15: countSince(15 * 60_000),
    },
    liveSeries,
    liveAlertsRecent,
  };
}
