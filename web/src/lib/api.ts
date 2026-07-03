/** Upstream FastAPI URL (used for display; proxy target in dev). */
export const DETECTION_API_UPSTREAM = "http://127.0.0.1:8000";

/**
 * Base URL for detection API calls from the browser.
 * Default: same-origin `/detection-api` proxy (no CORS). Set `NEXT_PUBLIC_API_PROXY=false`
 * and `NEXT_PUBLIC_API_URL` to call the API directly.
 */
export function getApiBaseUrl(): string {
  const env = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");
  const proxyOff = process.env.NEXT_PUBLIC_API_PROXY === "false";
  const proxyOn = process.env.NEXT_PUBLIC_API_PROXY === "true";

  if (typeof window !== "undefined") {
    if (proxyOn) return "/detection-api";
    if (proxyOff) return env || DETECTION_API_UPSTREAM;
    if (!env || env === DETECTION_API_UPSTREAM || env === "http://localhost:8000") {
      return "/detection-api";
    }
    return env;
  }
  return env || DETECTION_API_UPSTREAM;
}

function apiBase(): string {
  return getApiBaseUrl();
}

/** Human-readable label for the status bar. */
export function getApiDisplayLabel(): string {
  const base = getApiBaseUrl();
  if (base === "/detection-api") {
    return `Proxy → ${DETECTION_API_UPSTREAM}`;
  }
  return base;
}

export type ApiHealthResult =
  | { ok: true }
  | { ok: false; detail: string };

/** GET /health — uses the same base URL as inference requests. */
export async function checkDetectionApiHealth(): Promise<ApiHealthResult> {
  try {
    const res = await fetch(`${apiBase()}/health`, { method: "GET" });
    if (!res.ok) {
      return { ok: false, detail: res.statusText || `HTTP ${res.status}` };
    }
    return { ok: true };
  } catch (e) {
    const detail = e instanceof Error ? e.message : "Request failed";
    return { ok: false, detail };
  }
}

/** True when the browser could not complete the request (server down, DNS, CORS, etc.). */
export function isUnreachableApiError(e: unknown): boolean {
  if (!(e instanceof Error)) return false;
  const m = e.message.toLowerCase();
  return (
    m.includes("failed to fetch") ||
    m.includes("networkerror") ||
    m.includes("load failed") ||
    m.includes("fetch failed")
  );
}

export type Summary = Record<string, unknown> & {
  people?: number;
  normal?: number;
  suspicious?: number;
  overall_risk?: string;
  phone_near_person?: number;
  back_turned?: number;
  objects?: Record<string, unknown>;
  posture_mode?: string;
};

export type StatusItem = { label: string; confidence: number };

export type ImageProcessResponse = {
  original_png_base64: string;
  annotated_png_base64: string;
  summary: Summary;
  statuses: StatusItem[];
};

export type VideoFrameItem = {
  index: number;
  annotated_png_base64: string;
  media_type?: string;
  summary: Summary;
  statuses: StatusItem[];
};

export type VideoProcessResponse = {
  frame_count: number;
  frames: VideoFrameItem[];
  last_summary: Summary;
};

export type LiveFrameResponse = {
  session_id: string;
  annotated_png_base64: string;
  summary: Summary;
  statuses: StatusItem[];
};

/** Map vague proxy/network failures to actionable hints (e.g. truncated large uploads). */
export function formatDetectionApiError(e: unknown, opts?: { uploadBytes?: number }): string {
  const raw = e instanceof Error ? e.message : "Request failed";
  const lower = raw.toLowerCase();
  const mb = opts?.uploadBytes != null ? opts.uploadBytes / (1024 * 1024) : 0;

  if (
    mb > 9 &&
    (lower.includes("internal server error") ||
      lower.includes("socket hang") ||
      lower.includes("econnreset") ||
      lower.includes("failed to fetch"))
  ) {
    return (
      `Video upload failed (${mb.toFixed(0)} MB). The dev proxy may have truncated the file ` +
      `(default 10 MB limit). Restart the Next.js dev server so next.config proxyClientMaxBodySize applies, ` +
      `or set NEXT_PUBLIC_API_PROXY=false and use the API on port 8000 directly.`
    );
  }
  if (lower.includes("socket hang") || lower.includes("econnreset")) {
    return "Connection to the detection API was reset. Check that uvicorn is running and the video file is valid (try MP4 if MOV fails).";
  }
  return raw;
}

function parseApiError(text: string, statusText: string): string {
  const trimmed = text.trim();
  if (!trimmed) return statusText || "Request failed";
  try {
    const j = JSON.parse(trimmed) as { detail?: unknown };
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail.map((d) => (typeof d === "string" ? d : JSON.stringify(d))).join("; ");
    }
  } catch {
    /* plain text body */
  }
  return trimmed.length > 400 ? `${trimmed.slice(0, 400)}…` : trimmed;
}

async function formPost<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${apiBase()}${path}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(parseApiError(text, res.statusText));
  }
  return res.json() as Promise<T>;
}

export async function processImageApi(
  file: File,
  params: Record<string, string | boolean | number>,
): Promise<ImageProcessResponse> {
  const fd = new FormData();
  fd.append("file", file);
  Object.entries(params).forEach(([k, v]) => fd.append(k, String(v)));
  return formPost<ImageProcessResponse>("/api/process/image", fd);
}

export async function processVideoApi(
  file: File,
  params: Record<string, string | boolean | number>,
): Promise<VideoProcessResponse> {
  const fd = new FormData();
  fd.append("file", file);
  Object.entries(params).forEach(([k, v]) => fd.append(k, String(v)));
  return formPost<VideoProcessResponse>("/api/process/video", fd);
}

export async function processLiveFrameApi(
  blob: Blob,
  sessionId: string,
  params: Record<string, string | boolean | number>,
): Promise<LiveFrameResponse> {
  const fd = new FormData();
  fd.append("file", blob, "frame.jpg");
  fd.append("session_id", sessionId);
  Object.entries(params).forEach(([k, v]) => fd.append(k, String(v)));
  return formPost<LiveFrameResponse>("/api/live/frame", fd);
}

export async function deleteLiveSession(sessionId: string): Promise<void> {
  await fetch(`${apiBase()}/api/live/session/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}
