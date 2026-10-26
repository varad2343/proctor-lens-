import { useEffect, useRef, useState, type ReactNode } from "react";
import { api, chime, dur, LABEL, pref, PRIORITY_CLASS, ts, useApi, useProctorFeed, who, type Klass, type Msg, type Session } from "./lib";

export function Shell({ children, title }: { children: ReactNode; title?: ReactNode }) {
  const logout = () => api("/auth/logout", { method: "POST" }).finally(() => (location.hash = "#/login"));
  return (
    <div className="min-h-screen">
      <header className="border-b border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
          <a href="#/" className="font-semibold tracking-tight">ProctorLens</a>
          <span className="muted hidden text-sm sm:inline">observations for human review, not verdicts</span>
          <span className="ml-auto" />
          <Alerts />
          <button className="btn-ghost py-1" onClick={logout}>Log out</button>
        </div>
      </header>
      <main className="mx-auto max-w-7xl space-y-4 px-4 py-5">
        {title && <h1 className="text-xl font-semibold">{title}</h1>}
        {children}
      </main>
    </div>
  );
}

const NOTIFY_OK = typeof Notification !== "undefined";

/** Header alerts on every proctor page: tab title counts flags in progress; a chime and (opt-in) a desktop
 * notification when a flag starts. Flags are for review, so the wording never says more than what was observed. */
function Alerts() {
  const [sound, setSound] = useState(() => pref.get("sound", true));
  const [notify, setNotify] = useState(() => NOTIFY_OK && pref.get("notify", false) && Notification.permission === "granted");
  const [blocked, setBlocked] = useState(false);
  const { data } = useApi<Session[]>("/sessions", 3000);
  const n = data?.reduce((k, s) => k + (s.ongoing?.length ?? 0), 0) ?? 0;
  useEffect(() => {
    document.title = n ? `(${n}) ProctorLens` : "ProctorLens";
  }, [n]);
  const now = useRef({ sound, notify, data });
  now.current = { sound, notify, data };
  useProctorFeed((m) => {
    if (m.type !== "event_started") return;
    const { sound, notify, data } = now.current;
    const s = data?.find((x) => x.id === m.session_id), name = s ? who(s) : m.session_id;
    if (sound) chime();
    if (notify) new Notification(`${LABEL[m.event.type] ?? m.event.type}: ${name}`, {
      body: `Observed since ${ts(m.event.start_ms)}. Flagged for review, not a finding.`, tag: `${m.session_id}-${m.event.type}`,
    });
  });
  const toggleNotify = async () => {
    if (notify) return (pref.set("notify", false), setNotify(false));
    const ok = (await Notification.requestPermission()) === "granted";
    setBlocked(!ok);
    pref.set("notify", ok);
    setNotify(ok);
  };
  return (
    <>
      {n > 0 && <span className="rounded border border-amber-500 px-2 py-0.5 text-xs text-amber-900 dark:text-amber-200">{n} flag{n > 1 ? "s" : ""} in progress</span>}
      <button className="btn-ghost py-1" aria-pressed={sound} onClick={() => (pref.set("sound", !sound), setSound(!sound))}>
        Sound: {sound ? "on" : "off"}
      </button>
      {NOTIFY_OK && (
        <button className="btn-ghost py-1" aria-pressed={notify} onClick={toggleNotify} title={blocked ? "Notifications are blocked in this browser's site settings" : undefined}>
          Notifications: {notify ? "on" : blocked ? "blocked" : "off"}
        </button>
      )}
    </>
  );
}

function Ongoing({ items }: { items?: Session["ongoing"] }) {
  return items?.length ? (
    <span className="flex flex-wrap gap-1">
      {items.map((o) => (
        <span key={o.type} className="rounded border border-amber-500 px-1.5 text-xs text-amber-900 dark:text-amber-200">
          {LABEL[o.type] ?? o.type} <span className="muted">since {ts(o.start_ms)}</span>
        </span>
      ))}
    </span>
  ) : null;
}

export function Login({ linkKey }: { linkKey?: string }) {
  const [username, setUser] = useState("proctor");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string>();
  useEffect(() => {
    if (!linkKey) return;
    history.replaceState(null, "", "#/login"); // the key leaves the address bar and the history entry
    api("/auth/link", { json: { key: linkKey } }).then(() => (location.hash = "#/"), (x: Error) => setError(x.message));
  }, [linkKey]);
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
        Easiest: click the <b>one-click proctor login</b> link the server prints in its terminal (a new one after each
        restart or logout). You stay logged in for 30 days. Otherwise the username is <b>proctor</b> and the password is <b>proctor</b>,
        unless the server was started with <code>PROCTORLENS_PASSWORD</code>. On a public <code>--tunnel</code> address, password login works only when <code>PROCTORLENS_PASSWORD</code> is set.
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
  const classes = useApi<Klass[]>("/classes", 5000);
  const className = new Map((classes.data ?? []).map((k) => [k.id, k.name]));
  return (
    <Shell title="Sessions">
      <a className="btn-primary w-full py-2 text-base sm:w-auto" href="#/watch">
        Watch all students now{data ? ` (${data.filter((x) => x.status === "live").length} in an exam)` : ""}
      </a>
      <Classes list={classes.data} reload={classes.reload} />
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
                <td className="px-3 py-2 font-medium">{who(s)}<div className="muted text-xs">{s.id} · {s.policy_id}{className.has(s.exam_id) && <> · <a className="underline" href={`#/class/${s.exam_id}`}>{className.get(s.exam_id)}</a></>}</div></td>
                <td className="px-3 py-2">
                  <span className={`rounded px-2 py-0.5 text-xs ${STATUS_CLASS[s.status]}`}>{s.status === "created" && s.phase !== "created" ? `setup: ${s.phase}` : s.status}</span>
                  <div className="mt-1"><Ongoing items={s.ongoing} /></div>
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

/** Class links: one link for a whole class; every student who opens it gets a session of their own. */
function Classes({ list, reload }: { list?: Klass[]; reload: () => void }) {
  const { data: policies } = useApi<{ id: string }[]>("/policies");
  const [name, setName] = useState("");
  const [policy, setPolicy] = useState("default");
  const [error, setError] = useState<string>();
  const create = (e: React.FormEvent) => {
    e.preventDefault();
    api("/classes", { json: { name, policy_id: policy } }).then(() => (setName(""), setError(undefined), reload()), (x: Error) => setError(x.message));
  };
  const setOpen = (k: Klass, open: boolean) => api(`/classes/${k.id}/open`, { json: { open } }).then(reload);
  return (
    <section className="card space-y-3">
      <h2 className="font-semibold">Class links <span className="muted text-sm font-normal">one link for the whole class; each student gets their own session</span></h2>
      <form onSubmit={create} className="flex flex-wrap items-end gap-3">
        <label className="block min-w-48 flex-1 text-sm">Class or exam name<input className="input mt-1" value={name} onChange={(e) => setName(e.target.value)} required maxLength={100} placeholder="e.g. Physics 10B, mid-term" /></label>
        <label className="block text-sm">Policy<select className="input mt-1" value={policy} onChange={(e) => setPolicy(e.target.value)}>{policies?.map((p) => <option key={p.id}>{p.id}</option>)}</select></label>
        <button className="btn-primary">Create class link</button>
      </form>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {list?.map((k) => (
        <div key={k.id} className="space-y-1 border-t border-stone-200 pt-3 dark:border-stone-800">
          <div className="flex flex-wrap items-center gap-2">
            <b>{k.name}</b>
            <span className="muted text-sm">{k.n_students} student{k.n_students === 1 ? "" : "s"}{k.n_live ? `, ${k.n_live} in the exam now` : ""}</span>
            {!k.open && <span className="rounded border border-stone-400 px-1.5 text-xs">closed to new students</span>}
            <span className="ml-auto" />
            <a className="btn-primary py-1" href={`#/class/${k.id}`}>Watch class</a>
            <button className="btn-ghost py-1" onClick={() => setOpen(k, !k.open)}>{k.open ? "Close link" : "Reopen link"}</button>
          </div>
          {k.open && <JoinLink url={k.join_url} />}
        </div>
      ))}
    </section>
  );
}

/** Class grid: every student of a class with their live camera (detector view) and flags in progress. */
export function ClassView({ cid }: { cid: string }) {
  const { data: k } = useApi<Klass>(`/classes/${cid}`, 5000);
  const { data: students } = useApi<Session[]>(`/sessions?class_id=${cid}`, 2000);
  return (
    <Shell title={<>Class: {k?.name ?? cid} <span className="muted text-sm font-normal">{students ? `${students.length} student${students.length === 1 ? "" : "s"}` : ""}</span></>}>
      {k?.open && <div className="card space-y-1"><p className="text-sm">Students join with this one link:</p><JoinLink url={k.join_url} /></div>}
      {students?.length ? <StudentGrid groups={[["", students]]} /> : students && <p className="card muted">No students yet. Send them the link above.</p>}
    </Shell>
  );
}

/** Every student taking (or setting up) an exam right now, all classes and single links, in one window. */
export function WatchAll() {
  const { data } = useApi<Session[]>("/sessions", 2000);
  const { data: classes } = useApi<Klass[]>("/classes", 10000);
  const names = new Map((classes ?? []).map((k) => [k.id, k.name]));
  const active = (data ?? []).filter((s) => s.status === "live" || (s.status === "created" && s.phase !== "created"));
  const byGroup = new Map<string, Session[]>();
  for (const st of active) {
    const g = names.get(st.exam_id) ?? "Single links";
    byGroup.set(g, [...(byGroup.get(g) ?? []), st]);
  }
  const order = (a: Session, b: Session) =>
    (a.roll ?? "").localeCompare(b.roll ?? "", undefined, { numeric: true }) || a.candidate_label.localeCompare(b.candidate_label);
  const groups = [...byGroup].sort(([a], [b]) => (a === "Single links" ? 1 : b === "Single links" ? -1 : a.localeCompare(b)))
    .map(([g, list]) => [g, list.sort(order)] as [string, Session[]]);
  return (
    <Shell title={<>All students now <span className="muted text-sm font-normal">{data ? `${active.length} in an exam or setting up` : ""}</span></>}>
      {active.length ? <StudentGrid groups={groups} /> : data && <p className="card muted">No student is in an exam right now. This page fills in as students join.</p>}
    </Shell>
  );
}

/** Tiles for many students at once: live camera (detector view), flags in progress, recent observations. Click a
 * camera to enlarge that student in place; Fullscreen puts the whole wall on screen. */
function StudentGrid({ groups }: { groups: [string, Session[]][] }) {
  const [n, setN] = useState(0);
  const [big, setBig] = useState<string>();
  const wall = useRef<HTMLDivElement>(null);
  const count = groups.reduce((k, [, l]) => k + l.length, 0);
  useEffect(() => {
    const id = setInterval(() => setN((x) => x + 1), count > 12 ? 2000 : 1000); // ponytail: 2 s above 12 students (server draws each frame)
    return () => clearInterval(id);
  }, [count > 12]);
  return (
    <div ref={wall} className="space-y-4 overflow-auto bg-stone-50 dark:bg-stone-950 [&:fullscreen]:p-4">
      <div className="flex justify-end">
        <button className="btn-ghost py-1" onClick={() => (document.fullscreenElement ? document.exitFullscreen() : wall.current?.requestFullscreen())}>
          Fullscreen
        </button>
      </div>
      {groups.map(([title, list]) => (
        <section key={title || "all"} className="space-y-2">
          {title && <h2 className="font-semibold">{title} <span className="muted text-sm font-normal">{list.length}</span></h2>}
          <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-3">
            {list.map((st) => <StudentTile key={st.id} s={st} n={n} big={big === st.id} toggle={() => setBig(big === st.id ? undefined : st.id)} />)}
          </div>
        </section>
      ))}
    </div>
  );
}

function StudentTile({ s, n, big, toggle }: { s: Session; n: number; big: boolean; toggle: () => void }) {
  const [hasFrame, setHasFrame] = useState(false);
  const [recent, setRecent] = useState<string[]>([]);
  useProctorFeed((m) => {
    if (m.session_id !== s.id || m.type !== "event_ended") return;
    setRecent((r) => [`${LABEL[m.event.type] ?? m.event.type} ${ts(m.event.start_ms)}-${ts(m.event.end_ms)}`, ...r].slice(0, 3));
  });
  const live = s.status === "live";
  return (
    <div className={`card space-y-2 p-3 ${big ? "col-span-full" : ""} ${s.ongoing?.length ? "border-amber-500" : ""}`}>
      <div className="flex items-center gap-2">
        <b className="truncate">{who(s)}</b>
        <span className={`rounded px-2 py-0.5 text-xs ${STATUS_CLASS[s.status]}`}>{s.status === "created" && s.phase !== "created" ? `setup: ${s.phase}` : s.status}</span>
      </div>
      {live && (
        <button className={`block w-full ${hasFrame ? "" : "hidden"}`} onClick={toggle} aria-label={`${big ? "Shrink" : "Enlarge"} ${who(s)}'s camera`}>
          <img src={`/api/sessions/${s.id}/frame.jpg?overlay=1&n=${n}`} onLoad={() => setHasFrame(true)} onError={() => setHasFrame(false)}
            alt={`${who(s)}: latest camera frame with detector overlay`} className={`w-full rounded bg-black ${big ? "mx-auto max-h-[75vh] w-auto" : ""}`} />
        </button>
      )}
      {live && !hasFrame && <div className="muted flex aspect-video items-center justify-center rounded bg-stone-100 text-sm dark:bg-stone-800">Waiting for camera...</div>}
      {!live && <div className="muted flex aspect-video items-center justify-center rounded bg-stone-100 text-sm dark:bg-stone-800">{s.status === "ended" ? "Exam finished" : "Getting ready (consent, camera check)"}</div>}
      <Ongoing items={s.ongoing} />
      {recent.length > 0 && <ul className="muted text-xs">{recent.map((r, i) => <li key={i}>{r}</li>)}</ul>}
      <div className="flex flex-wrap gap-2 text-sm">
        {s.status !== "ended" && <a className="btn-ghost py-1" href={`#/live/${s.id}`}>Details</a>}
        {s.status !== "created" && <a className="btn-ghost py-1" href={`#/review/${s.id}`}>Review</a>}
        {s.n_events > 0 && <span className="muted self-center">{s.n_events} flag{s.n_events === 1 ? "" : "s"} so far</span>}
      </div>
    </div>
  );
}

function JoinLink({ url, small }: { url: string; small?: boolean }) {
  const [copied, setCopied] = useState(false);
  const full = url.startsWith("http") ? url : location.origin + url; // absolute = the public (tunnel) address
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
          <p className="text-sm">Send this link to <b>{made.label}</b> (it opens the consent page and the exam in a browser on any computer or phone):</p>
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
  const [wsStatus, setStatus] = useState<Msg>();
  const status = wsStatus ?? s?.live ?? undefined; // browser mode: the readouts the candidate's browser last sent
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
    <Shell title={<>Live: {s ? who(s) : sid} <span className="muted text-sm font-normal">{s?.status === "ended" ? "(ended)" : status?.phase ?? s?.phase}</span></>}>
      {!!s?.ongoing?.length && <div className="flex items-center gap-2 text-sm"><span className="font-medium">In progress now:</span><Ongoing items={s.ongoing} /></div>}
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
