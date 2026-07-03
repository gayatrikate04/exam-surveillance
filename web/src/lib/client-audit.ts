export type ClientAuditLevel = "info" | "warn" | "error";

export function postClientAudit(payload: {
  action: string;
  message: string;
  level?: ClientAuditLevel;
  meta?: Record<string, unknown>;
}) {
  void fetch("/api/audit/event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      level: payload.level ?? "info",
      action: payload.action,
      message: payload.message,
      meta: payload.meta,
    }),
    credentials: "same-origin",
  }).catch(() => {});
}
