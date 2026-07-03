"use client";

import { useState } from "react";
import { AdminPanel } from "@/app/admin/admin-panel";
import { LogoutButton } from "@/components/logout-button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BackendStatusBar } from "@/components/backend-status-bar";
import { APP_TITLE } from "@/lib/branding";

type DashboardMode = "monitor" | "admin";

function AdminViewGate() {
  return (
    <div className="mx-auto flex w-full max-w-lg flex-col gap-4 py-4">
      <Card>
        <CardHeader>
          <CardTitle className="font-heading text-lg">Admin view</CardTitle>
          <CardDescription>This area is for designated administrators.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>To enable logs and alerts for your account:</p>
          <ul className="list-inside list-disc space-y-1">
            <li>
              Add <code className="rounded bg-muted px-1">ADMIN_EMAILS=your@email.com</code> in{" "}
              <code className="rounded bg-muted px-1">web/.env.local</code>, or
            </li>
            <li>
              Set <code className="rounded bg-muted px-1">&quot;role&quot;: &quot;admin&quot;</code> on your user in{" "}
              <code className="rounded bg-muted px-1">web/data/users.json</code>
            </li>
          </ul>
          <p>Then sign out and sign in again so your session picks up admin access.</p>
        </CardContent>
      </Card>
    </div>
  );
}

export function HomeDashboard({
  isAdmin,
  email,
  children,
}: {
  isAdmin: boolean;
  email: string | null;
  children: React.ReactNode;
}) {
  const [mode, setMode] = useState<DashboardMode>("monitor");

  return (
    <>
      <header className="flex flex-wrap items-start justify-between gap-3 border-b px-4 py-3">
        <div className="min-w-0 flex-1">
          <h1 className="text-xl font-semibold tracking-tight">{APP_TITLE}</h1>
        </div>
        <div className="flex shrink-0 flex-row flex-wrap items-center justify-end gap-2 sm:gap-3">
          {email ? (
            <span className="max-w-[10rem] truncate text-xs text-muted-foreground sm:max-w-[14rem]" title={email}>
              {email}
            </span>
          ) : null}
          {email ? (
            <Tabs
              value={mode}
              onValueChange={(v) => setMode(v as DashboardMode)}
              className="w-[min(100%,260px)] shrink-0 sm:w-auto"
            >
              <TabsList className="grid h-8 w-full grid-cols-2 p-0.5">
                <TabsTrigger value="monitor" className="px-2 text-xs sm:text-sm">
                  Monitor
                </TabsTrigger>
                <TabsTrigger value="admin" className="px-2 text-xs sm:text-sm">
                  Admin view
                </TabsTrigger>
              </TabsList>
            </Tabs>
          ) : null}
          <LogoutButton />
        </div>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
        {mode === "monitor" ? <BackendStatusBar /> : null}
        {mode === "monitor" ? (
          children
        ) : isAdmin ? (
          <AdminPanel embedded />
        ) : (
          <AdminViewGate />
        )}
      </div>
    </>
  );
}
