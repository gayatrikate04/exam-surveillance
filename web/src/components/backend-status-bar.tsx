"use client";

import { useCallback, useEffect, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  checkDetectionApiHealth,
  DETECTION_API_UPSTREAM,
  getApiDisplayLabel,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type Status = "checking" | "connected" | "disconnected";

export function BackendStatusBar() {
  const [status, setStatus] = useState<Status>("checking");
  const [detail, setDetail] = useState<string | null>(null);
  const [label, setLabel] = useState(DETECTION_API_UPSTREAM);

  const probe = useCallback(async () => {
    setStatus((s) => (s === "connected" ? "connected" : "checking"));
    setLabel(getApiDisplayLabel());
    const result = await checkDetectionApiHealth();
    if (result.ok) {
      setStatus("connected");
      setDetail(null);
    } else {
      setStatus("disconnected");
      setDetail(result.detail);
    }
  }, []);

  useEffect(() => {
    void probe();
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") void probe();
    }, 5000);
    return () => window.clearInterval(id);
  }, [probe]);

  if (status === "connected") {
    return (
      <div
        className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-3 py-2 text-sm"
        role="status"
      >
        <div className="flex min-w-0 items-center gap-2">
          <span className="relative flex size-2 shrink-0">
            <span className="absolute inline-flex size-full animate-ping rounded-full bg-emerald-500 opacity-40" />
            <span className="relative inline-flex size-2 rounded-full bg-emerald-600" />
          </span>
          <span className="font-medium text-emerald-950 dark:text-emerald-100">Backend connected</span>
          <span className="truncate text-xs text-muted-foreground" title={label}>
            {label}
          </span>
        </div>
        <Button type="button" variant="ghost" size="xs" className="h-7 shrink-0" onClick={() => void probe()}>
          Recheck
        </Button>
      </div>
    );
  }

  return (
    <Alert
      variant="destructive"
      className={cn(
        status === "checking" && "border-border bg-muted/40 text-foreground [&>svg]:text-muted-foreground",
      )}
    >
      <AlertTitle className="text-sm font-semibold">
        {status === "checking" ? "Checking backend…" : "Backend not connected"}
      </AlertTitle>
      <AlertDescription className="space-y-2 text-sm">
        {status === "disconnected" ? (
          <>
            <p>
              The detection API at <strong>{DETECTION_API_UPSTREAM}</strong> is not reachable
              {detail ? ` (${detail})` : ""}. Image, video, and live modes need it running.
            </p>
            <p className="text-xs">
              From the project folder, double-click <code className="rounded bg-muted px-1">run_web_stack.bat</code>{" "}
              (starts API + web), or in a terminal:
            </p>
            <code className="block rounded-md bg-muted px-2 py-1.5 text-xs">
              uvicorn api_server:app --host 127.0.0.1 --port 8000
            </code>
            <p className="text-xs text-muted-foreground">
              After starting the API, click Recheck. The web app uses a same-origin proxy (
              <code className="rounded bg-muted px-0.5">/detection-api</code>) so CORS errors should not appear once
              the backend is up.
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">Probing {label}…</p>
        )}
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="mt-1 h-8"
          onClick={() => void probe()}
          disabled={status === "checking"}
        >
          Recheck
        </Button>
      </AlertDescription>
    </Alert>
  );
}
