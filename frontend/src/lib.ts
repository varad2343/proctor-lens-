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
  roll: string | null; // class-link students
  join_url: string; duration_s: number | null; n_events: number; n_reviewed: number; blind_spot_s: number;
  flagged_s: number; counts: Record<string, number>; priority: Partial<Record<Priority, number>>;
  ongoing: { type: string; start_ms: number }[]; // flags in progress now (live sessions)
  review_note: string; // the reviewer's session note
  live?: Msg | null; // browser mode: latest status from the candidate's browser (phase, readouts, ongoing)
  calibration: { mode?: string; error?: number | null; accepted?: boolean; tries?: number };
  policy: { allow_notes: boolean; allow_looking_down: boolean; allow_reading_aloud: boolean; active: string[] };
};
export type Ev = {
  id: number; session_id: string; type: string; start_ms: number; end_ms: number; confidence: number;
  detector: string; explanation: string; clip_path: string | null; status: string;
  details: Record<string, unknown>; attribution: Record<string, number> | null;
  thumbs: Partial<Record<"onset" | "peak" | "end" | "reference", string>>; review: Review | null; // reference: face-mismatch only
  thumbs_overlay?: Partial<Record<"onset" | "peak" | "end", string>>; // same frames with the detector overlay
  clip_overlay?: string | null; // the clip with the detector overlay (boxes + track numbers)
};
export type Seg = {
  id: number; lane: "review" | "blind_spot"; start_ms: number; end_ms: number; events: number[]; types: string[];
  review_priority: number; priority_label: Priority;
};
/** How a student is shown: "23 · Asha" for class-link students, the label otherwise. */
export const who = (s: { roll?: string | null; candidate_label: string }) => (s.roll ? `${s.roll} · ${s.candidate_label}` : s.candidate_label);

export type Klass = {
  id: string; name: string; policy_id: string; open: boolean; join_url: string; n_students: number; n_live: number;
  created_at: string;
};
export type Timeline = { t_ms: number[]; series: Record<string, (number | null)[]> };
export type Msg = { type: string; [k: string]: any };

export async function api<T>(path: string, opts: { method?: string; json?: unknown } = {}): Promise<T> {
  if (opts.method || opts.json !== undefined) shared.forEach((e) => (e.t = 0)); // a change: the next reads go to the server
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
    if (r.status === 401 && path !== "/auth/login" && path !== "/auth/link") location.hash = "#/login";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return r.json();
}

/** GETs shared by every component on the page: one request per path per second, however many ask (the header's
 * alerts and the sessions list both poll /sessions); the last answer per path shows at once when a page opens again,
 * while the fresh one loads (each request is a round trip to the cloud server). */
const shared = new Map<string, { t: number; p: Promise<unknown>; last?: unknown }>();
function sharedGet<T>(path: string): Promise<T> {
  const s = shared.get(path);
  if (s && Date.now() - s.t < 1000) return s.p as Promise<T>;
  const e: { t: number; p: Promise<unknown>; last?: unknown } = { t: Date.now(), p: api<T>(path), last: s?.last };
  shared.set(path, e);
  e.p.then((d) => (e.last = d), () => undefined);
  return e.p as Promise<T>;
}

/** GET path (and again every `everyMs` when > 0). path null = idle. */
export function useApi<T>(path: string | null, everyMs = 0) {
  const [data, setData] = useState<T | undefined>(() => (path ? (shared.get(path)?.last as T) : undefined));
  const [error, setError] = useState<string>();
  const [n, setN] = useState(0);
  useEffect(() => {
    if (!path) return;
    let live = true;
    const last = shared.get(path)?.last as T | undefined;
    if (last !== undefined) setData(last);
    if (n) shared.delete(path); // reload(): skip the shared answer
    const go = () =>
      sharedGet<T>(path).then(
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
// One /ws/proctor connection shared by every component on the page (header alerts + live view).
/** "server": this server runs the models (live sockets). "browser": candidates' browsers do (ADR-019; the cloud
 * build): the page polls instead of holding sockets. */
let modeP: Promise<"server" | "browser"> | undefined;
export const serverMode = (): Promise<"server" | "browser"> =>
  (modeP ??= fetch("/api/health").then((r) => r.json()).then((h): "server" | "browser" => (h.mode === "browser" ? "browser" : "server"))
    .catch((): "server" => "server"));

const feedSubs = new Set<(m: Msg) => void>();
let feedWs: WebSocket | { close(): void } | undefined;
function openFeed() {
  void serverMode().then((mode) => (mode === "browser" ? pollFeed() : openSocket()));
}

/** Browser mode: /api/feed every 2 s (same messages as /ws/proctor: event_started, event_ended). */
function pollFeed() {
  let stop = false, last = -1;
  feedWs = { close: () => (stop = true) };
  const step = async () => {
    if (stop) return;
    try {
      const r = await fetch(last < 0 ? "/api/feed" : `/api/feed?since=${last}`);
      if (r.ok) {
        const d = await r.json();
        last = d.last;
        d.msgs.forEach((m: Msg) => feedSubs.forEach((f) => f(m)));
      }
    } catch {
      /* offline: try again */
    }
    setTimeout(step, 2000);
  };
  void step();
}

function openSocket() {
  const ws = new WebSocket(wsUrl("/ws/proctor"));
  feedWs = ws;
  ws.onmessage = (e: MessageEvent) => {
    const m = JSON.parse(e.data);
    feedSubs.forEach((f) => f(m));
  };
  ws.onclose = () => {
    feedWs = undefined;
    setTimeout(() => feedSubs.size && !feedWs && openFeed(), 2000); // reconnect while anyone listens
  };
}

export function useProctorFeed(onMsg: (m: Msg) => void) {
  useEffect(() => {
    feedSubs.add(onMsg);
    if (!feedWs) openFeed();
    return () => {
      feedSubs.delete(onMsg);
      setTimeout(() => !feedSubs.size && feedWs?.close(), 1000); // page switches resubscribe within the second
    };
  }, []); // onMsg is captured once: read changing values through refs
}

/** Proctor-side alert preferences, per browser (a convenience, so localStorage; it may be unavailable). */
export const pref = {
  get: (k: string, d: boolean) => {
    try {
      const v = localStorage.getItem(`pl.${k}`);
      return v === null ? d : v === "1";
    } catch {
      return d;
    }
  },
  set: (k: string, v: boolean) => {
    try {
      localStorage.setItem(`pl.${k}`, v ? "1" : "0");
    } catch {
      /* private window: the choice lasts until reload */
    }
  },
};

let audio: AudioContext | undefined;
/** Short two-note chime (Web Audio, no sound file). Browsers start audio only after a click on the page. */
export function chime() {
  audio ??= new AudioContext();
  void audio.resume();
  [660, 880].forEach((f, i) => {
    const o = audio!.createOscillator(), g = audio!.createGain(), t = audio!.currentTime + i * 0.15;
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.15, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.3);
    o.connect(g).connect(audio!.destination);
    o.start(t);
    o.stop(t + 0.32);
  });
}

export function ts(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  const p = (x: number) => String(x).padStart(2, "0");
  return `${p(Math.floor(s / 3600))}:${p(Math.floor((s % 3600) / 60))}:${p(s % 60)}`;
}

export const dur = (s: number | null | undefined) =>
  s == null ? "n/a" : s < 60 ? `${s.toFixed(1)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;

// Review priority orders the work; it is not a verdict, so it is shown by ink weight (filled > outlined > plain),
// never by alarm colours: red stays for destructive actions and the recording indicator.
export const PRIORITY_CLASS: Record<string, string> = {
  high: "bg-stone-900 text-white dark:bg-stone-100 dark:text-stone-900",
  medium: "border border-stone-500 text-stone-800 dark:border-stone-400 dark:text-stone-100",
  low: "border border-dashed border-stone-300 text-stone-600 dark:border-stone-600 dark:text-stone-300",
  blind: "bg-slate-500 text-white",
};

/** Candidate side of /ws/stream: JPEG frames at 10 fps from a worker timer (header: seq uint32 LE, client time
 * float64 LE, the master spec 10.3 layout), at most 2 unacknowledged; control messages queue while reconnecting. */
/** What the candidate page needs from its connection: the server's (CandidateStream, a WebSocket) or its own browser's
 * (BrowserStream, a worker running the session: browser mode). */
export interface Stream {
  now(): number;
  send(m: Msg): void;
  close(): void;
}

export async function openStream(sid: string, token: string, video: HTMLVideoElement, onMsg: (m: Msg) => void): Promise<Stream> {
  return (await serverMode()) === "browser" ? new BrowserStream(sid, token, video, onMsg) : new CandidateStream(sid, token, video, onMsg);
}

const RTC: RTCConfiguration = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };
/** Waits for the connection candidates (at most 3 s), so one offer / answer each way is the whole handshake. */
const gathered = (pc: RTCPeerConnection) =>
  new Promise<void>((done) => {
    if (pc.iceGatheringState === "complete") return done();
    pc.onicegatheringstatechange = () => pc.iceGatheringState === "complete" && done();
    setTimeout(done, 3000);
  });

/** Proctor's page, browser mode: the candidate's camera as live video (~0.3 s behind) instead of a picture a second.
 * undefined while connecting, or when the networks block a direct connection (no TURN relay): the picture stays. */
export function useLiveVideo(sid: string, on: boolean): MediaStream | undefined {
  const [stream, setStream] = useState<MediaStream>();
  useEffect(() => {
    if (!on) return;
    let stop = false;
    const pc = new RTCPeerConnection(RTC);
    const viewer = Math.random().toString(36).slice(2, 12);
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.ontrack = (e) => setStream(e.streams[0] ?? new MediaStream([e.track]));
    pc.onconnectionstatechange = () => ["failed", "closed"].includes(pc.connectionState) && setStream(undefined);
    void (async () => {
      if ((await serverMode()) !== "browser") return;
      await pc.setLocalDescription(await pc.createOffer());
      await gathered(pc);
      await api(`/sessions/${sid}/rtc`, { json: { viewer, sdp: pc.localDescription!.sdp } });
      for (let i = 0; i < 30 && !stop; i++) { // the candidate's browser answers with its next sync (~1 s)
        const { answer } = await api<{ answer: string | null }>(`/sessions/${sid}/rtc/${viewer}`);
        if (answer) return void (stop || (await pc.setRemoteDescription({ type: "answer", sdp: answer })));
        await new Promise((r) => setTimeout(r, 1000));
      }
    })().catch(() => undefined);
    return () => {
      stop = true;
      pc.close();
      setStream(undefined);
    };
  }, [sid, on]);
  return stream;
}

/** Browser mode: start the session software (~80 MB on a first visit) while the candidate reads the consent page;
 * the BrowserStream for this session takes it over, with the messages it posted meanwhile. */
let preloaded: { sid: string; worker: Worker; early: MessageEvent[] } | undefined;
export function preloadSession(sid: string, token: string) {
  if (preloaded?.sid === sid) return;
  const worker = new Worker(new URL("./session.worker.ts", import.meta.url));
  const early: MessageEvent[] = [];
  worker.onmessage = (e) => early.push(e);
  worker.postMessage({ type: "init", sid, token });
  preloaded = { sid, worker, early };
}

/** Browser mode: frames go to session.worker.ts (ProctorLens's session code, run here), which answers with the same
 * messages the server sends over /ws/stream. One frame in flight; JPEG at the same size and quality as to the server. */
export class BrowserStream implements Stream {
  private worker: Worker;
  private ticker = new Worker(new URL("./ticker.ts", import.meta.url), { type: "module" });
  private seq = 0;
  private busy = false;

  constructor(sid: string, token: string, private video: HTMLVideoElement, onMsg: (m: Msg) => void) {
    if (preloaded?.sid !== sid) preloadSession(sid, token);
    const { worker, early } = preloaded!;
    preloaded = undefined;
    this.worker = worker;
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ack") this.busy = false;
      else if (m.type === "msgs") m.msgs.forEach((x: Msg) => x.type !== "ack" && onMsg(x));
      else if (m.type === "progress") onMsg({ type: "loading", text: m.text });
      else if (m.type === "fatal") onMsg({ type: "loading", text: m.detail, fatal: true });
      else if (m.type === "rtc") m.offers.forEach((o: { viewer: string; offer: string }) => this.answer(sid, token, o));
    };
    early.forEach((e) => this.worker.onmessage!(e));
    this.send({ type: "hello" }); // answered once the session software has loaded
    this.ticker.onmessage = () => this.tick();
    this.ticker.postMessage(100);
  }

  now = () => performance.timeOrigin + performance.now();

  send(m: Msg) {
    this.worker.postMessage({ type: "control", msg: m });
  }

  private tick() {
    const v = this.video;
    if (this.busy || v.readyState < 2) return;
    const t = this.now();
    this.busy = true;
    // a GPU-side copy of the frame; the worker scales and JPEG-encodes it, so the exam page itself never stutters
    createImageBitmap(v).then(
      (bitmap) => this.worker.postMessage({ type: "frame", seq: ++this.seq, t, bitmap }, [bitmap]),
      () => (this.busy = false),
    );
  }

  /** Live video straight to a proctor's page (WebRTC); the 1 s picture in the sync stays as the fallback. */
  private peers = new Map<string, RTCPeerConnection>();
  private async answer(sid: string, token: string, o: { viewer: string; offer: string }) {
    const stream = this.video.srcObject as MediaStream | null;
    if (!stream || this.peers.has(o.viewer)) return;
    const pc = new RTCPeerConnection(RTC);
    this.peers.set(o.viewer, pc);
    stream.getVideoTracks().forEach((t) => pc.addTrack(t, stream));
    pc.onconnectionstatechange = () => ["failed", "closed"].includes(pc.connectionState) && this.peers.delete(o.viewer);
    await pc.setRemoteDescription({ type: "offer", sdp: o.offer });
    await pc.setLocalDescription(await pc.createAnswer());
    await gathered(pc);
    await fetch(`/api/c/${sid}/rtc`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, viewer: o.viewer, sdp: pc.localDescription!.sdp }) });
  }

  close() {
    this.peers.forEach((pc) => pc.close());
    this.ticker.postMessage(0);
    this.ticker.terminate();
    // the worker keeps running until its uploads finish; the page ends it by navigating away
  }
}

export class CandidateStream implements Stream {
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
