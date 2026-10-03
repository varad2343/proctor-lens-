import { useEffect, useState, type ReactNode } from "react";
import { api, dur, LABEL, PRIORITY_CLASS, ts, useApi, useProctorFeed, type Msg, type Session } from "./lib";

export function Shell({ children, title }: { children: ReactNode; title?: ReactNode }) {
  const logout = () => api("/auth/logout", { method: "POST" }).finally(() => (location.hash = "#/login"));
  return (
    <div className="min-h-screen">
      <header className="border-b border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-2.5">
          <a href="#/" className="font-semibold tracking-tight">ProctorLens</a>
          <span className="muted hidden text-sm sm:inline">observations for human review, not verdicts</span>
          <button className="btn-ghost ml-auto py-1" onClick={logout}>Log out</button>
        </div>
      </header>
      <main className="mx-auto max-w-7xl space-y-4 px-4 py-5">
        {title && <h1 className="text-xl font-semibold">{title}</h1>}
        {children}
      </main>
    </div>
  );
}

export function Login() {
  const [username, setUser] = useState("proctor");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string>();
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    api("/auth/login", { json: { username, password } }).then(() => (location.hash = "#/"), (x: Error) => setError(x.message));
  };
  return (
    <form onSubmit={submit} className="card mx-auto mt-24 max-w-sm space-y-3">
      <h1 className="text-xl font-semibold">ProctorLens proctor login</h1>
      <label className="block text-sm">Username<input className="input mt-1" value={username} onChange={(e) => setUser(e.target.value)} autoComplete="username" /></label>
      <label className="block text-sm">Password<input className="input mt-1" type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" autoFocus /></label>
      {error && <p className="text-sm text-red-700 dark:text-red-400" role="alert">{error}</p>}
      <button className="btn-primary w-full">Log in</button>
      <p className="muted text-xs">
        Username: <b>proctor</b> (unless the server was started with a different <code>PROCTORLENS_USER</code>). Password: the
        value <code>PROCTORLENS_PASSWORD</code> was set to when the server started; if it was not set, the server printed a
        password in its terminal at start.
      </p>
    </form>
  );
}

const STATUS_CLASS: Record<string, string> = {
  created: "bg-stone-200 text-stone-800 dark:bg-stone-700 dark:text-stone-100",
  live: "bg-emerald-100 text-emerald-900 dark:bg-emerald-900 dark:text-emerald-100",
  ended: "bg-sky-100 text-sky-900 dark:bg-sky-900 dark:text-sky-100",
};

export function Sessions() {
  const { data, error, reload } = useApi<Session[]>("/sessions", 3000);
  return (
    <Shell title="Sessions">
      <NewSession onCreated={reload} />
      {error && <p className="text-red-700">{error}</p>}
      <div className="card overflow-x-auto p-0">
        <table className="w-full text-sm">
          <thead className="muted border-b border-stone-200 text-left text-xs uppercase tracking-wide dark:border-stone-800">
            <tr>{["Candidate", "Status", "Started", "Duration", "Review segments", "Blind spots", "Reviewed", ""].map((h) => <th key={h} className="px-3 py-2 font-medium">{h}</th>)}</tr>
          </thead>
          <tbody>
            {data?.map((s) => (
              <tr key={s.id} className="border-b border-stone-100 last:border-0 dark:border-stone-800">
                <td className="px-3 py-2 font-medium">{s.candidate_label}<div className="muted text-xs">{s.id} · {s.policy_id}</div></td>
                <td className="px-3 py-2">
                  <span className={`rounded px-2 py-0.5 text-xs ${STATUS_CLASS[s.status]}`}>{s.status === "created" && s.phase !== "created" ? `setup: ${s.phase}` : s.status}</span>
                </td>
                <td className="px-3 py-2">{s.started_at ? new Date(s.started_at).toLocaleString() : <span className="muted">not yet</span>}</td>
                <td className="px-3 py-2">{dur(s.duration_s)}</td>
                <td className="px-3 py-2">
                  <span className="flex gap-1">
                    {(["high", "medium", "low"] as const).map((p) => (s.priority[p] ? <span key={p} className={`rounded px-1.5 text-xs ${PRIORITY_CLASS[p]}`}>{s.priority[p]} {p}</span> : null))}
                    {!s.n_events && <span className="muted">none</span>}
                  </span>
                </td>
                <td className="px-3 py-2">{s.blind_spot_s ? dur(s.blind_spot_s) : <span className="muted">none</span>}</td>
                <td className="px-3 py-2">{s.n_events ? `${s.n_reviewed} / ${s.n_events}` : ""}</td>
                <td className="whitespace-nowrap px-3 py-2 text-right">
                  {s.status !== "ended" && <a className="btn-ghost py-1" href={`#/live/${s.id}`}>Live</a>}{" "}
                  {s.status !== "created" && <a className="btn-ghost py-1" href={`#/review/${s.id}`}>Review</a>}{" "}
                  <JoinLink url={s.join_url} small />{" "}
                  <button className="btn-ghost py-1 text-red-700 dark:text-red-400" onClick={() => confirm(`Delete session ${s.candidate_label} and all of its data?`) && api(`/sessions/${s.id}`, { method: "DELETE" }).then(reload)}>Delete</button>
                </td>
              </tr>
            ))}
            {data?.length === 0 && <tr><td colSpan={8} className="muted px-3 py-6 text-center">No sessions yet. Create one above and send the candidate its link.</td></tr>}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}

function JoinLink({ url, small }: { url: string; small?: boolean }) {
  const [copied, setCopied] = useState(false);
  const full = location.origin + url;
  const copy = () =>
    navigator.clipboard?.writeText(full).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  return small ? (
    <button className="btn-ghost py-1" onClick={copy} title={full}>{copied ? "Copied" : "Copy link"}</button>
  ) : (
    <div className="flex items-center gap-2">
      <code className="input truncate">{full}</code>
      <button className="btn-ghost" onClick={copy}>{copied ? "Copied" : "Copy"}</button>
    </div>
  );
}

function NewSession({ onCreated }: { onCreated: () => void }) {
  const { data: policies } = useApi<{ id: string }[]>("/policies");
  const [label, setLabel] = useState("");
  const [policy, setPolicy] = useState("default");
  const [made, setMade] = useState<{ join_url: string; label: string }>();
  const [error, setError] = useState<string>();
  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    api<{ join_url: string }>("/sessions", { json: { candidate_label: label, policy_id: policy } }).then(
      (r) => {
        setMade({ join_url: r.join_url, label });
        setLabel("");
        setError(undefined);
        onCreated();
      },
      (x: Error) => setError(x.message),
    );
  };
  return (
    <form onSubmit={submit} className="card space-y-3">
      <div className="flex flex-wrap items-end gap-3">
        <label className="block min-w-48 flex-1 text-sm">Candidate (pseudonym)<input className="input mt-1" value={label} onChange={(e) => setLabel(e.target.value)} required maxLength={100} placeholder="e.g. P07" /></label>
        <label className="block text-sm">Policy<select className="input mt-1" value={policy} onChange={(e) => setPolicy(e.target.value)}>{policies?.map((p) => <option key={p.id}>{p.id}</option>)}</select></label>
        <button className="btn-primary">New session</button>
      </div>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {made && (
        <div className="space-y-1">
          <p className="text-sm">Send this link to <b>{made.label}</b> (it opens the consent page and the exam; open it in Chrome or Edge on the candidate's computer):</p>
          <JoinLink url={made.join_url} />
        </div>
      )}
    </form>
  );
}

type FeedItem = { at: number; text: string; type: string };
const num = (v: unknown, f: (x: number) => string) => (typeof v === "number" ? f(v) : "n/a");

export function Live({ sid }: { sid: string }) {
  const { data: s } = useApi<Session>(`/sessions/${sid}`, 3000);
  const [frame, setFrame] = useState(0);
  const [hasFrame, setHasFrame] = useState(true);
  const [overlay, setOverlay] = useState(true);
  const [status, setStatus] = useState<Msg>();
  const [feed, setFeed] = useState<FeedItem[]>([]);
  useEffect(() => {
    const id = setInterval(() => setFrame((n) => n + 1), 1000); // ~1 fps view is enough to follow (master spec 11.2)
    return () => clearInterval(id);
  }, []);
  useProctorFeed((m) => {
    if (m.session_id !== sid) return;
    if (m.type === "session_status") setStatus((old) => ({ ...old, ...m }));
    else if (m.type === "event_started" || m.type === "event_ended") {
      const e = m.event;
      const text = m.type === "event_started"
        ? `${LABEL[e.type] ?? e.type} observed since ${ts(e.start_ms)} (ongoing)`
        : `${LABEL[e.type] ?? e.type}: ${ts(e.start_ms)} - ${ts(e.end_ms)}`;
      setFeed((f) => [{ at: Date.now(), text, type: m.type }, ...f].slice(0, 100));
    }
  });
  const q = typeof status?.quality === "number" ? status.quality : undefined;
  return (
    <Shell title={<>Live: {s?.candidate_label ?? sid} <span className="muted text-sm font-normal">{s?.status === "ended" ? "(ended)" : status?.phase ?? s?.phase}</span></>}>
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="card space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
              Show detector overlay {overlay && <span className="muted">(camera view, not mirrored)</span>}
            </label>
            {overlay && <span className="muted text-xs">Face mesh, boxes and numbers show what the detectors saw; they can be wrong. Track numbers follow boxes and do not identify anyone.</span>}
          </div>
          {/* reloaded every second; hidden while there is no frame, so it shows up again by itself */}
          <img src={`/api/sessions/${sid}/frame.jpg?n=${frame}${overlay ? "&overlay=1" : ""}`} onError={() => setHasFrame(false)}
            onLoad={() => setHasFrame(true)} alt={overlay ? "latest camera frame with detector overlay" : "latest camera frame"}
            className={hasFrame ? `w-full rounded bg-black ${overlay ? "" : "-scale-x-100"}` : "hidden"} />
          {!hasFrame && (
            <div className="muted flex aspect-video items-center justify-center rounded bg-stone-100 px-4 text-center dark:bg-stone-800">
              No frame: the candidate has not connected yet, has lost the connection, or the session ended.
            </div>
          )}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-4">
            <dt className="muted">Image quality</dt><dd>{q === undefined ? "n/a" : q.toFixed(2)}</dd>
            <dt className="muted">Frame rate</dt><dd>{num(status?.fps, (v) => `${v.toFixed(1)} fps`)}</dd>
            <dt className="muted">Attention zone</dt><dd>{status?.zone ?? "n/a"}</dd>
            <dt className="muted">Head turn (yaw / pitch)</dt><dd>{num(status?.d_yaw, (v) => `${v.toFixed(0)} deg`)} / {num(status?.d_pitch, (v) => `${v.toFixed(0)} deg`)}</dd>
            <dt className="muted">Faces in view</dt><dd>{num(status?.n_faces, String)}</dd>
            <dt className="muted">People in view</dt><dd>{num(status?.n_persons, String)}</dd>
            <dt className="muted">Phone detector</dt><dd>{num(status?.phone_conf, (v) => v.toFixed(2))}</dd>
            <dt className="muted">Book / notes detector</dt><dd>{num(status?.notes_conf, (v) => v.toFixed(2))}</dd>
            <dt className="muted">Processing</dt><dd>{num(status?.proc_ms, (v) => `${v.toFixed(0)} ms / frame`)}</dd>
            <dt className="muted">Quality notes</dt><dd>{(status?.reasons as string[] | undefined)?.join(", ") || "none"}</dd>
          </dl>
        </div>
        <div className="card space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Live observations</h2>
            {s?.status !== "created" && <a className="btn-ghost py-1" href={`#/review/${sid}`}>Open review</a>}
          </div>
          <p className="muted text-xs">Flags for later human review; nothing here is a finding about the candidate.</p>
          <ul className="max-h-[60vh] space-y-1 overflow-y-auto text-sm" aria-live="polite">
            {feed.map((f) => (
              <li key={f.at + f.text} className={f.type === "event_started" ? "text-amber-800 dark:text-amber-300" : ""}>
                <span className="muted mr-2 font-mono text-xs">{new Date(f.at).toLocaleTimeString()}</span>{f.text}
              </li>
            ))}
            {!feed.length && <li className="muted">Nothing observed since this page opened.</li>}
          </ul>
        </div>
      </div>
    </Shell>
  );
}
