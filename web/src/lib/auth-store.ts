import { createHash, randomBytes, scryptSync, timingSafeEqual } from "crypto";
import { mkdirSync, readFileSync, writeFileSync, existsSync, appendFileSync } from "fs";
import path from "path";
import type { StoredUser } from "@/lib/auth-types";

export { normalizeEmail } from "@/lib/email-normalize";

const USERS_FILE = path.join(process.cwd(), "data", "users.json");
const USERS_CSV_FILE = path.join(process.cwd(), "data", "users.csv");
const CSV_HEADER = "id,email,role,createdAt,salt,passwordHash\n";

function ensureDataDir() {
  const dir = path.dirname(USERS_FILE);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
}

function csvEscape(value: string) {
  if (/[",\r\n]/.test(value)) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

export function appendUserCsv(user: StoredUser) {
  ensureDataDir();
  if (!existsSync(USERS_CSV_FILE)) {
    writeFileSync(USERS_CSV_FILE, CSV_HEADER, "utf8");
  }
  const row = [
    user.id,
    user.email,
    user.role,
    user.createdAt,
    user.salt,
    user.passwordHash,
  ]
    .map(csvEscape)
    .join(",");
  appendFileSync(USERS_CSV_FILE, row + "\n", "utf8");
}


export function readUsers(): StoredUser[] {
  try {
    if (!existsSync(USERS_FILE)) return [];
    const raw = readFileSync(USERS_FILE, "utf8");
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed as StoredUser[];
  } catch {
    return [];
  }
}

export function writeUsers(users: StoredUser[]) {
  ensureDataDir();
  writeFileSync(USERS_FILE, JSON.stringify(users, null, 2), "utf8");
}

export function hashPassword(password: string, salt?: string) {
  const s = salt ?? randomBytes(16).toString("hex");
  const hash = scryptSync(password, s, 64).toString("hex");
  return { salt: s, hash };
}

export function verifyPassword(password: string, user: StoredUser) {
  const { hash } = hashPassword(password, user.salt);
  const a = Buffer.from(hash, "hex");
  const b = Buffer.from(user.passwordHash, "hex");
  if (a.length !== b.length) return false;
  return timingSafeEqual(a, b);
}

export function newUserId() {
  return createHash("sha256")
    .update(randomBytes(32))
    .digest("hex")
    .slice(0, 32);
}
