"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { AdminDashboardView } from "@/app/admin/admin-dashboard-view";
import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { AdminDashboardStats } from "@/lib/audit-stats";
import type { AuditEntry } from "@/lib/audit-store";
import { cn } from "@/lib/utils";

type LogsResponse = {
  entries: AuditEntry[];
  alerts: AuditEntry[];
  stats: AdminDashboardStats;
  meta: { historyReturned: number; totalStored: number };
};

function levelBadgeVariant(level: AuditEntry["level"]) {
  if (level === "error") return "destructive" as const;
  if (level === "warn") return "secondary" as const;
  return "outline" as const;
}

function EntryRow({ e }: { e: AuditEntry }) {
  return (
    <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 border-b border-border/60 py-2 text-sm last:border-0">
      <Badge variant={levelBadgeVariant(e.level)} className="h-5 shrink-0 px-1.5 text-[10px] uppercase">
        {e.level}
      </Badge>
      <div className="min-w-0 space-y-0.5">
        <p className="font-medium leading-tight text-foreground">{e.message}</p>
        <p className="text-xs text-muted-foreground">
          <span className="font-mono">{e.action}</span>
          <span className="mx-1">·</span>
          <time dateTime={e.ts}>{new Date(e.ts).toLocaleString()}</time>
          {e.actorEmail ? (
            <>
              <span className="mx-1">·</span>
              <span title={e.actorEmail}>{e.actorEmail}</span>
            </>
          ) : null}
        </p>
      </div>
    </div>
  );
}

function formatSync(iso: string | null) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString();
  } catch {
    return "—";
  }
}

export function AdminPanel({ embedded = false }: { embedded?: boolean }) {
  const [data, setData] = useState<LogsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [liveRefresh, setLiveRefresh] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<string | null>(null);

  const load = useCallback(async (silent: boolean) => {
    if (!silent) {
      setLoading(true);
      setError(null);
    } else {
      setSyncing(true);
    }
    try {
      const res = await fetch("/api/admin/logs", { credentials: "same-origin" });
      if (res.status === 403) {
        if (!silent) {
          setError("You do not have admin access.");
          setData(null);
        }
        return;
      }
      if (!res.ok) {
        if (!silent) {
          setError("Could not load logs.");
          setData(null);
        }
        return;
      }
      const json = (await res.json()) as LogsResponse;
      setData(json);
      setLastSync(new Date().toISOString());
    } catch {
      if (!silent) {
        setError("Network error.");
        setData(null);
      }
    } finally {
      if (!silent) setLoading(false);
      else setSyncing(false);
    }
  }, []);

  useEffect(() => {
    void load(false);
  }, [load]);

  useEffect(() => {
    if (!liveRefresh || error) return;
    const id = window.setInterval(() => {
      if (document.visibilityState !== "visible") return;
      void load(true);
    }, 4000);
    return () => window.clearInterval(id);
  }, [liveRefresh, error, load]);

  return (
    <div
      className={cn(
        "mx-auto flex w-full flex-col gap-5",
        embedded ? "max-w-7xl" : "max-w-6xl p-4",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          {embedded ? (
            <>
              <h2 className="font-heading text-lg font-semibold tracking-tight">Admin mode</h2>
              <p className="text-xs text-muted-foreground">Live dashboard, history, and alerts · switch to Monitor when done.</p>
            </>
          ) : (
            <>
              <h1 className="font-heading text-2xl font-semibold tracking-tight">Admin</h1>
              <p className="text-sm text-muted-foreground">Live metrics and audit history (JSON-backed).</p>
            </>
          )}
        </div>
        <div className="flex flex-col items-end gap-2 sm:flex-row sm:items-center">
          {data && !error ? (
            <div className="flex flex-wrap items-center justify-end gap-3 rounded-lg border border-border/80 bg-muted/30 px-3 py-2">
              <div className="flex items-center gap-2">
                <Label htmlFor="admin-live" className="text-xs text-muted-foreground">
                  Live (4s)
                </Label>
                <Switch id="admin-live" checked={liveRefresh} onCheckedChange={setLiveRefresh} />
              </div>
              <Separator orientation="vertical" className="hidden h-6 sm:block" />
              <div className="text-xs text-muted-foreground">
                {syncing ? <span className="text-foreground">Updating…</span> : <span>Idle</span>}
                <span className="mx-1.5">·</span>
                <span title={lastSync ?? undefined}>Sync {formatSync(lastSync)}</span>
                {data.meta ? (
                  <>
                    <span className="mx-1.5">·</span>
                    <span>{data.meta.historyReturned} / {data.meta.totalStored} rows</span>
                  </>
                ) : null}
              </div>
            </div>
          ) : null}
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => void load(false)} disabled={loading}>
              {loading ? "Loading…" : "Refresh now"}
            </Button>
            {!embedded ? (
              <Link href="/" className={cn(buttonVariants({ variant: "secondary", size: "sm" }))}>
                Back to monitor
              </Link>
            ) : null}
          </div>
        </div>
      </div>

      {error ? (
        <Card className="border-destructive/50">
          <CardHeader>
            <CardTitle className="text-base text-destructive">Access</CardTitle>
            <CardDescription>{error}</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Set <code className="rounded bg-muted px-1">ADMIN_EMAILS</code> to your email (comma-separated) or
              assign <code className="rounded bg-muted px-1">role: &quot;admin&quot;</code> on your user in{" "}
              <code className="rounded bg-muted px-1">data/users.json</code>, then sign in again.
            </p>
          </CardContent>
        </Card>
      ) : null}

      {!error && data ? (
        <Tabs defaultValue="dashboard">
          <TabsList className="h-auto flex-wrap gap-1">
            <TabsTrigger value="dashboard">Dashboard</TabsTrigger>
            <TabsTrigger value="history">
              History
              <Badge variant="secondary" className="ml-1.5 h-5 min-w-6 px-1 text-[10px]">
                {data.entries.length}
              </Badge>
            </TabsTrigger>
            <TabsTrigger value="alerts">
              Alerts
              <Badge variant="secondary" className="ml-1.5 h-5 min-w-6 px-1 text-[10px]">
                {data.alerts.length}
              </Badge>
            </TabsTrigger>
          </TabsList>
          <Separator className="my-3" />
          <TabsContent value="dashboard" className="mt-0">
            <AdminDashboardView stats={data.stats} />
          </TabsContent>
          <TabsContent value="history" className="mt-0">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Event history</CardTitle>
                <CardDescription>Newest first · same feed used for live charts (up to 800 rows returned).</CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <ScrollArea className="h-[min(62vh,640px)] pr-3">
                  {data.entries.length === 0 ? (
                    <p className="py-8 text-center text-sm text-muted-foreground">No entries yet.</p>
                  ) : (
                    data.entries.map((e) => <EntryRow key={e.id} e={e} />)
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="alerts" className="mt-0">
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Alerts</CardTitle>
                <CardDescription>Warnings and errors · refreshes with live polling.</CardDescription>
              </CardHeader>
              <CardContent className="pt-0">
                <ScrollArea className="h-[min(62vh,640px)] pr-3">
                  {data.alerts.length === 0 ? (
                    <p className="py-8 text-center text-sm text-muted-foreground">No alerts.</p>
                  ) : (
                    data.alerts.map((e) => <EntryRow key={e.id} e={e} />)
                  )}
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      ) : null}

      {!error && !data && loading ? (
        <p className="text-sm text-muted-foreground">Loading…</p>
      ) : null}
    </div>
  );
}
