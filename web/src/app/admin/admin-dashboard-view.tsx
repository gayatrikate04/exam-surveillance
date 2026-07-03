"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { AdminDashboardStats } from "@/lib/audit-stats";

const LEVEL_FILL: Record<string, string> = {
  info: "oklch(0.45 0.15 250)",
  warn: "oklch(0.75 0.15 75)",
  error: "oklch(0.55 0.22 25)",
};

function Kpi({ label, value, hint }: { label: string; value: number | string; hint?: string }) {
  return (
    <Card>
      <CardHeader className="pb-1 pt-4">
        <CardDescription className="text-xs">{label}</CardDescription>
        <CardTitle className="font-heading text-2xl tabular-nums">{value}</CardTitle>
        {hint ? <p className="text-[11px] text-muted-foreground">{hint}</p> : null}
      </CardHeader>
    </Card>
  );
}

const tooltipBox = {
  borderRadius: 8,
  border: "1px solid oklch(0.9 0 0)",
  fontSize: 12,
  background: "oklch(0.99 0 0)",
};

export function AdminDashboardView({ stats }: { stats: AdminDashboardStats }) {
  const hourly = stats.hourly24hUtc;
  const topActionsChart = stats.topActions.map((a) => ({
    ...a,
    short: a.action.length > 28 ? `${a.action.slice(0, 28)}…` : a.action,
  }));

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Kpi label="Events (last 24h, UTC buckets)" value={stats.totals24h.total} hint="Auth + monitor activity" />
        <Kpi label="Warnings" value={stats.totals24h.warn} />
        <Kpi label="Errors" value={stats.totals24h.error} />
        <Kpi label="Last 5 min" value={stats.rates.m5} hint="Rolling event count" />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Live monitor status</CardTitle>
            <CardDescription>
              From <code className="rounded bg-muted px-1">app.live.status</code> audit rows: people, suspicious count,
              and risk (0 low · 1 medium · 2 high). Updates while Live runs and admin refresh / live polling is on.
              {stats.liveSeries.length > 0 ? ` ${stats.liveSeries.length} samples in view.` : " No samples yet."}
            </CardDescription>
          </CardHeader>
          <CardContent className="h-[280px] pt-0">
            {stats.liveSeries.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={stats.liveSeries} margin={{ top: 8, right: 10, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border/80" />
                  <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                  <YAxis yAxisId="left" allowDecimals={false} width={34} tick={{ fontSize: 10 }} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    allowDecimals={false}
                    domain={[0, 2]}
                    width={30}
                    tick={{ fontSize: 10 }}
                  />
                  <Tooltip contentStyle={tooltipBox} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="people"
                    name="People"
                    stroke="oklch(0.48 0.18 280)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="suspicious"
                    name="Suspicious"
                    stroke="oklch(0.68 0.17 55)"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="riskScore"
                    name="Risk"
                    stroke="oklch(0.35 0 0)"
                    strokeWidth={2}
                    strokeDasharray="5 4"
                    dot={false}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center px-4 text-center text-sm text-muted-foreground">
                Start Live with the API running; status points are written to
                the audit log every few seconds and appear here on the next refresh.
              </div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Live and detection alerts</CardTitle>
            <CardDescription>Recent warnings/errors from live and detection actions</CardDescription>
          </CardHeader>
          <CardContent className="max-h-[280px] overflow-y-auto pr-1 pt-0">
            {stats.liveAlertsRecent.length === 0 ? (
              <p className="py-8 text-center text-xs text-muted-foreground">No live alerts in the log yet.</p>
            ) : (
              <ul className="space-y-3 text-sm">
                {stats.liveAlertsRecent.map((a) => (
                  <li key={a.id} className="rounded-md border border-border/70 bg-muted/25 p-2.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant={a.level === "error" ? "destructive" : "secondary"} className="text-[10px]">
                        {a.level}
                      </Badge>
                      <span className="font-mono text-[10px] text-muted-foreground">{a.action}</span>
                    </div>
                    <p className="mt-1.5 leading-snug text-foreground">{a.message}</p>
                    <p className="mt-1 text-[10px] text-muted-foreground">{new Date(a.ts).toLocaleString()}</p>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Volume over time</CardTitle>
            <CardDescription>Hourly event counts (UTC), last 24 hours</CardDescription>
          </CardHeader>
          <CardContent className="h-[240px] pt-0">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hourly} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="adminVol" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="oklch(0.45 0.12 260)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="oklch(0.45 0.12 260)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border/80" />
                <XAxis dataKey="label" tick={{ fontSize: 10 }} interval={3} angle={-35} textAnchor="end" height={52} />
                <YAxis allowDecimals={false} width={32} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={tooltipBox} formatter={(v: number) => [`${v} events`, "Count"]} />
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="oklch(0.4 0.14 260)"
                  strokeWidth={2}
                  fill="url(#adminVol)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">By severity (24h)</CardTitle>
            <CardDescription>Info vs warning vs error</CardDescription>
          </CardHeader>
          <CardContent className="h-[240px] pt-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.byLevel} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-border/80" />
                <XAxis dataKey="level" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} width={28} tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={tooltipBox} />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} maxBarSize={48}>
                  {stats.byLevel.map((row) => (
                    <Cell key={row.level} fill={LEVEL_FILL[row.level] ?? "oklch(0.5 0 0)"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {topActionsChart.length > 0 ? (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Top actions (24h)</CardTitle>
            <CardDescription>Most frequent audit action keys</CardDescription>
          </CardHeader>
          <CardContent className="h-[min(320px,28vh)] min-h-[200px] pt-0">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={topActionsChart}
                layout="vertical"
                margin={{ top: 4, right: 16, left: 4, bottom: 4 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal className="stroke-border/80" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="short" width={120} tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={tooltipBox}
                  formatter={(v: number) => [`${v}`, "Count"]}
                  labelFormatter={(label) => {
                    const row = topActionsChart.find((r) => r.short === label);
                    return row?.action ?? String(label);
                  }}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} fill="oklch(0.35 0.02 260)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-3">
        <Card>
          <CardHeader className="py-3">
            <CardDescription>Last 1 min</CardDescription>
            <CardTitle className="text-lg tabular-nums">{stats.rates.m1}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="py-3">
            <CardDescription>Last 15 min</CardDescription>
            <CardTitle className="text-lg tabular-nums">{stats.rates.m15}</CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader className="py-3">
            <CardDescription>Stored rows (cap 2500)</CardDescription>
            <CardTitle className="text-lg tabular-nums">{stats.totalStored}</CardTitle>
          </CardHeader>
        </Card>
      </div>
    </div>
  );
}
