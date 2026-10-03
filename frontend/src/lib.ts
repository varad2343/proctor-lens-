import { useEffect, useState } from "react";

// Same order as core.types.REVIEW_TYPES (MONITORING_DEGRADED last = the blind-spot lane).
export const REVIEW_TYPES = [
  "FACE_ABSENT", "MULTIPLE_PEOPLE", "PROHIBITED_OBJECT", "OFF_SCREEN_SUSTAINED", "REPEATED_GLANCING",
  "MOUTH_ACTIVITY", "IDENTITY_MISMATCH", "BROWSER_INTEGRITY", "MONITORING_DEGRADED",
] as const;
// Observations only: what was seen, never a judgement of the person.
export const LABEL: Record<string, string> = {
  FACE_ABSENT: "No one in view",
  MULTIPLE_PEOPLE: "More than one person",
  PROHIBITED_OBJECT: "Phone or notes visible",
  OFF_SCREEN_SUSTAINED: "Attention off screen",
  REPEATED_GLANCING: "Repeated glancing",
  MOUTH_ACTIVITY: "Mouth activity",
  IDENTITY_MISMATCH: "Face differs from reference",
  BROWSER_INTEGRITY: "Browser activity",
  MONITORING_DEGRADED: "Blind spot",
};

export type Priority = "high" | "medium" | "low";
export type Review = { decision: "confirm" | "dismiss" | "needs_more_info"; note: string; reviewer: string; created_at: string };
export type Session = {
  id: string; exam_id: string; candidate_label: string; policy_id: string; status: "created" | "live" | "ended";
  phase: string; mode: string | null; created_at: string; started_at: string | null; ended_at: string | null;
  join_url: string; duration_s: number | null; n_events: number; n_reviewed: number; blind_spot_s: number;
  flagged_s: number; counts: Record<string, number>; priority: Partial<Record<Priority, number>>;
  calibration: { mode?: string; error?: number | null; accepted?: boolean; tries?: number };
  policy: { allow_notes: boolean; allow_looking_down: boolean; allow_reading_aloud: boolean; active: string[] };
};
export type Ev = {
  id: number; session_id: string; type: string; start_ms: number; end_ms: number; confidence: number;
  detector: string; explanation: string; clip_path: string | null; status: string;
  details: Record<string, unknown>; attribution: Record<string, number> | null;
  thumbs: Partial<Record<"onset" | "peak" | "end", string>>; review: Review | null;
};
export type Seg = {
  id: number; lane: "review" | "blind_spot"; start_ms: number; end_ms: number; events: number[]; types: string[];
  review_priority: number; priority_label: Priority;
};
export type Timeline = { t_ms: number[]; series: Record<string, (number | null)[]> };
export type Msg = { type: string; [k: string]: any };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T>(path: string, opts: { method?: string; json?: unknown } = {}): Promise<T> {
  const r = await fetch("/api" + path, {
    method: opts.method ?? (opts.json === undefined ? "GET" : "POST"),
    headers: opts.json === undefined ? undefined : { "Content-Type": "application/json" },
    body: opts.json === undefined ? undefined : JSON.stringify(opts.json),
  });
  if (!r.ok) {
    let detail: unknown = r.statusText;
    try {
      detail = (await r.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    if (r.status === 401 && path !== "/auth/login") location.hash = "#/login";
    throw new ApiError(r.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}

/** GET path (and again every `everyMs` when > 0). path null = idle. */
export function useApi<T>(path: string | null, everyMs = 0) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string>();
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!path) return;
    let live = true;
    const go = () =>
      api<T>(path).then(
        (d) => live && (setData(d), setError(undefined)),
        (e: Error) => live && setError(e.message),
      );
    go();
    const id = everyMs > 0 ? setInterval(go, everyMs) : undefined;
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [path, everyMs, n]);
  return { data, error, reload: () => setN((x) => x + 1) };
}

export const wsUrl = (path: string) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;

/** Live messages from /ws/proctor (event_started / event_ended / session_status); reconnects. */
export function useProctorFeed(onMsg: (m: Msg) => void) {
  useEffect(() => {
    let ws: WebSocket | undefined;
    let stop = false;
    const open = () => {
      ws = new WebSocket(wsUrl("/ws/proctor"));
      ws.onmessage = (e) => onMsg(JSON.parse(e.data));
      ws.onclose = () => !stop && setTimeout(open, 2000);
    };
    open();
    return () => {
      stop = true;
      ws?.close();
    };
  }, []); // one connection per mount: onMsg must only call state setters
}

export function ts(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  const p = (x: number) => String(x).padStart(2, "0");
  return `${p(Math.floor(s / 3600))}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}`;
}

export const dur = (s: number | null | undefined) =>
  s == null ? "n/a" : s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;

export const PRIORITY_CLASS: Record<string, string> = {
  high: "bg-red-700 text-white",
  medium: "bg-amber-500 text-stone-950",
  low: "bg-stone-400 text-stone-950",
  blind: "bg-slate-500 text-white",
};

/** Candidate side of /ws/stream: JPEG frames at 10 fps from a worker timer (header: seq uint32 LE, client time
 * float64 LE, the master spec 10.3 layout), at most 2 unacknowledged; control messages queue while reconnecting. */
export class CandidateStream {
  private ws?: WebSocket;
  private seq = 0;
  private acked = 0;
  private busy = false;
  private queue: string[] = [];
  private closed = false;
  private canvas = document.createElement("canvas");
  private ticker = new Worker(new URL("./ticker.ts", import.meta.url), { type: "module" });

  constructor(private sid: string, private token: string, private video: HTMLVideoElement, private onMsg: (m: Msg) => void) {
    this.ticker.onmessage = () => this.tick();
    this.ticker.postMessage(100);
    this.connect();
  }

  /** Epoch ms that is monotonic within the page and comparable across a reload (the server's session clock). */
  now = () => performance.timeOrigin + performance.now();

  private connect() {
    const ws = (this.ws = new WebSocket(wsUrl(`/ws/stream/${this.sid}?token=${encodeURIComponent(this.token)}`)));
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: "hello" }));
      this.queue.splice(0).forEach((m) => ws.send(m));
    };
    ws.onmessage = (e) => {
      const m: Msg = JSON.parse(e.data);
      if (m.type === "ack") this.acked = Math.max(this.acked, m.seq);
      else this.onMsg(m);
    };
    ws.onclose = (e) => {
      this.acked = this.seq; // nothing in flight on a dead socket
      if (this.closed) return;
      if (e.code === 4401) this.onMsg({ type: "rejected" });
      else setTimeout(() => this.connect(), 1000);
    };
  }

  send(m: Msg) {
    const s = JSON.stringify(m);
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(s);
    else this.queue.push(s);
  }

  private tick() {
    const v = this.video;
    if (this.busy || this.ws?.readyState !== WebSocket.OPEN || v.readyState < 2 || this.seq - this.acked >= 2) return;
    const t = this.now();
    const w = 640;
    const h = Math.round((w * v.videoHeight) / v.videoWidth) || 480;
    this.canvas.width = w;
    this.canvas.height = h;
    this.canvas.getContext("2d")!.drawImage(v, 0, 0, w, h);
    this.busy = true;
    this.canvas.toBlob(
      (b) => {
        this.busy = false;
        if (!b || this.ws?.readyState !== WebSocket.OPEN) return;
        const head = new DataView(new ArrayBuffer(12));
        head.setUint32(0, ++this.seq, true);
        head.setFloat64(4, t, true);
        this.ws.send(new Blob([head, b]));
      },
      "image/jpeg",
      0.7,
    );
  }

  close() {
    this.closed = true;
    this.ticker.postMessage(0);
    this.ticker.terminate();
    this.ws?.close();
  }
}
