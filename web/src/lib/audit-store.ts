import { randomUUID } from "crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import path from "path";

export type AuditLevel = "info" | "warn" | "error";

export type AuditEntry = {
  id: string;
  ts: string;
  level: AuditLevel;
  action: string;
  message: string;
  actorUserId?: string;
  actorEmail?: string;
  meta?: Record<string, unknown>;
};

const AUDIT_FILE = path.join(process.cwd(), "data", "audit.json");
const MAX_ENTRIES = 2500;

function ensureDir() {
  const dir = path.dirname(AUDIT_FILE);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

export function readAudit(): AuditEntry[] {
  try {
    if (!existsSync(AUDIT_FILE)) return [];
    const raw = readFileSync(AUDIT_FILE, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed as AuditEntry[];
  } catch {
    return [];
  }
}

function writeAudit(entries: AuditEntry[]) {
  ensureDir();
  writeFileSync(AUDIT_FILE, JSON.stringify(entries, null, 2), "utf8");
}

export function appendAudit(
  partial: Omit<AuditEntry, "id" | "ts"> & { id?: string; ts?: string },
) {
  const entries = readAudit();
  const entry: AuditEntry = {
    id: partial.id ?? randomUUID(),
    ts: partial.ts ?? new Date().toISOString(),
    level: partial.level,
    action: partial.action,
    message: partial.message,
    actorUserId: partial.actorUserId,
    actorEmail: partial.actorEmail,
    meta: partial.meta,
  };
  entries.unshift(entry);
  writeAudit(entries.slice(0, MAX_ENTRIES));
}
