import { memo, useEffect, useRef, useState, type ReactNode } from "react";
import { alertName, alertWho, api, chime, dur, LABEL, pref, PRIORITY_CLASS, speak, ts, useApi, useLiveVideo, useProctorFeed, who, type Klass, type Msg, type Session } from "./lib";

/** Exam rules as the teacher sees them; the id sent to the server does not change. */
export const policyName = (id: string) => (id === "default" ? "Standard" : id);
const PHASE_WORD: Record<string, string> = {
  created: "not started", check: "camera check", calib: "eye calibration", exam: "in the exam", ended: "finished",
};
const phaseWord = (p?: string) => (p ? PHASE_WORD[p] ?? p : "");
const STATUS_WORD: Record<Session["status"], string> = { created: "Not started", live: "In the exam", ended: "Finished" };
const statusWord = (s: Session) => (s.status === "created" && s.phase !== "created" ? "Getting ready" : STATUS_WORD[s.status]);

/** "Exam rules" picker: shows "Standard" for the "default" policy id; the helper says which to pick. */
function PolicySelect({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data: policies } = useApi<{ id: string }[]>("/policies");
  return (
    <label className="block text-sm">Exam rules
      <select className="input mt-1" value={value} onChange={(e) => onChange(e.target.value)}>
        {policies?.map((p) => <option key={p.id} value={p.id}>{policyName(p.id)}</option>)}
      </select>
      <span className="muted block text-xs">Standard works for most exams.</span>
    </label>
  );
}

export function Shell({ children, title, hint }: { children: ReactNode; title?: ReactNode; hint?: string }) {
  const logout = () => api("/auth/logout", { method: "POST" }).finally(() => (location.hash = "#/login"));
  return (
    <div className="min-h-screen">
      <header className="border-b border-stone-200 bg-white dark:border-stone-800 dark:bg-stone-900">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
          <a href="#/" className="font-semibold tracking-tight">ProctorLens</a>
          <span className="muted hidden text-sm sm:inline">observations for human review, not verdicts</span>
          <span className="ml-auto" />
          <a href="#/alerts" className="btn-ghost py-1">Alert board</a>
          <Alerts />
          <button className="btn-ghost py-1" onClick={logout}>Log out</button>
        </div>
      </header>
      <main className="mx-auto max-w-7xl space-y-4 px-4 py-5">
        {title && <h1 className="text-xl font-semibold">{title}</h1>}
        {hint && <p className="muted -mt-3 text-sm">{hint}</p>}
        {children}
      </main>
    </div>
  );
}

const NOTIFY_OK = typeof Notification !== "undefined";

/** Header alerts on every proctor page: tab title counts flags in progress; a chime and (opt-in) a desktop
 * notification when a flag starts. Flags are for review, so the wording never says more than what was observed. */
function Alerts() {
  const [, redraw] = useState(0);
  useEffect(() => {
    const f = () => redraw((n) => n + 1);
    addEventListener("pl-prefs", f);
    return () => removeEventListener("pl-prefs", f);
  }, []);
  const sound = pref.get("sound", true), voice = pref.get("voice", false);
  const notify = NOTIFY_OK && pref.get("notify", false) && Notification.permission === "granted";
  const [blocked, setBlocked] = useState(false);
  const { data } = useApi<Session[]>("/sessions?active=1", 3000);
  const flagged = (data ?? []).filter((s) => s.ongoing?.length);
  const n = flagged.reduce((k, s) => k + s.ongoing!.length, 0);
  const first = flagged[0], firstType = first?.ongoing?.[0].type;
  useEffect(() => { // the browser tab itself says who: visible even from another window
    document.title = first ? `(${n}) ${alertName(first)}: ${LABEL[firstType!] ?? firstType}` : "ProctorLens";
  }, [n, first?.id, firstType]);
  const now = useRef({ sound, voice, notify, data });
  now.current = { sound, voice, notify, data };
  useProctorFeed((m) => {
    if (m.type !== "event_started") return;
    const { sound, voice, notify, data } = now.current;
    const s = data?.find((x) => x.id === m.session_id), name = s ? alertName(s) : m.session_id, what = LABEL[m.event.type] ?? m.event.type;
    if (sound) chime();
    if (voice) speak(`${name}. ${what}`);
    if (notify) {
      const note = new Notification(s ? alertWho(s) : name, { body: `${what}. Flagged for review, not a finding.`, tag: `${m.session_id}-${m.event.type}` });
      note.onclick = () => { window.focus(); location.hash = `#/live/${m.session_id}`; note.close(); };
    }
  });
  const toggleNotify = async () => {
    if (notify) return pref.set("notify", false);
    const ok = (await Notification.requestPermission()) === "granted";
    setBlocked(!ok);
    pref.set("notify", ok);
  };
  return (
    <>
      {n > 0 && <a href="#/alerts" className="rounded border border-red-500 px-2 py-0.5 text-xs text-red-800 dark:text-red-200">{n} flag{n > 1 ? "s" : ""} in progress</a>}
      <button className="btn-ghost py-1" aria-pressed={sound} onClick={() => pref.set("sound", !sound)}>
        Sound: {sound ? "on" : "off"}
      </button>
      <button className="btn-ghost py-1" aria-pressed={voice} onClick={() => pref.set("voice", !voice)} title="Reads each flag aloud, e.g. Roll 23, phone or notes visible">
        Voice: {voice ? "on" : "off"}
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
      <p className="muted text-sm">Log in to create exam links, watch your students and review flags.</p>
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
  const { data, error, reload } = useApi<Session[]>("/sessions", 15000); // creating / deleting reloads at once
  const classes = useApi<Klass[]>("/classes", 5000);
  const className = new Map((classes.data ?? []).map((k) => [k.id, k.name]));
  const blindTip = "Time the camera was dark, covered or frozen";
  return (
    <Shell title="Sessions" hint="Every student's exam is listed here: watch it live while it runs, then review its flags.">
      <ol className="card flex flex-wrap gap-x-6 gap-y-1 py-2 text-sm" aria-label="How it works">
        <li><b>1.</b> Create a class link</li>
        <li><b>2.</b> Send it to your students</li>
        <li><b>3.</b> Watch live, then review the flags</li>
      </ol>
      <a className="btn-primary w-full py-2 text-base sm:w-auto" href="#/watch">
        Watch all students{data ? ` (${data.filter((x) => x.status === "live").length} in an exam)` : ""}
      </a>
      <Classes list={classes.data} reload={classes.reload} />
      <NewSession onCreated={reload} />
      {error && <p className="text-red-700">{error}</p>}
      <div className="card overflow-x-auto p-0">
        <table className="w-full text-sm">
          <thead className="muted border-b border-stone-200 text-left text-xs uppercase tracking-wide dark:border-stone-800">
            <tr>{["Student", "Status", "Started", "Duration", "Flags to review", "Camera unclear", "Reviewed", ""].map((h) => (
              <th key={h} className="px-3 py-2 font-medium" title={h === "Camera unclear" ? blindTip : undefined}>{h}</th>
            ))}</tr>
          </thead>
          <tbody>
            {data?.map((s) => (
              <tr key={s.id} className="border-b border-stone-100 last:border-0 dark:border-stone-800">
                <td className="px-3 py-2 font-medium">{who(s)}<div className="muted text-xs">{policyName(s.policy_id)} rules{className.has(s.exam_id) && <> · <a className="underline" href={`#/class/${s.exam_id}`}>{className.get(s.exam_id)}</a></>}</div></td>
                <td className="px-3 py-2">
                  <span className={`rounded px-2 py-0.5 text-xs ${STATUS_CLASS[s.status]}`}>{statusWord(s)}</span>
                  <div className="mt-1"><Ongoing items={s.ongoing} /></div>
                </td>
                <td className="px-3 py-2">{s.started_at ? new Date(s.started_at).toLocaleString() : <span className="muted">not yet</span>}</td>
                <td className="px-3 py-2">{dur(s.duration_s)}</td>
                <td className="px-3 py-2">
                  <span className="flex flex-wrap gap-1">
                    {(["high", "medium", "low"] as const).map((p) => (s.priority[p] ? <span key={p} className={`rounded px-1.5 text-xs ${PRIORITY_CLASS[p]}`}>{s.priority[p]} {p} priority</span> : null))}
                    {!s.n_events && <span className="muted">none</span>}
                  </span>
                </td>
                <td className="px-3 py-2" title={blindTip}>{s.blind_spot_s ? dur(s.blind_spot_s) : <span className="muted">none</span>}</td>
                <td className="px-3 py-2">{s.n_events ? `${s.n_reviewed} of ${s.n_events}` : ""}</td>
                <td className="sticky right-0 whitespace-nowrap bg-white px-3 py-2 text-right dark:bg-stone-900">{/* actions stay in view on narrow screens */}
                  {s.status !== "ended" && <a className="btn-ghost py-1" href={`#/live/${s.id}`}>Watch live</a>}{" "}
                  {s.status !== "created" && <a className="btn-ghost py-1" href={`#/review/${s.id}`}>Review</a>}{" "}
                  <JoinLink url={s.join_url} small />{" "}
                  <button className="btn-ghost py-1 text-red-700 dark:text-red-400" onClick={() => confirm(`Delete the exam of ${who(s)} with all of its clips, pictures and flags? This cannot be undone.`) && api(`/sessions/${s.id}`, { method: "DELETE" }).then(reload, (e: Error) => alert(`Could not delete: ${e.message}`))}>Delete</button>
                </td>
              </tr>
            ))}
            {data?.length === 0 && <tr><td colSpan={8} className="muted px-3 py-6 text-center">No exams yet. Create a class link above (or a link for one student) and send it to your students.</td></tr>}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}

/** One line per question: "text | option | option | ..." (2-8 options). Blank/malformed lines are skipped; no
 * valid lines = no custom exam (the class uses the built-in mock-exam bank). */
function parseQuestions(text: string): { id: string; text: string; options: string[] }[] {
  return text.split("\n").map((ln) => ln.split("|").map((s) => s.trim()).filter(Boolean))
    .filter((parts) => parts.length >= 3 && parts.length <= 9)
    .map(([q, ...options], i) => ({ id: `q${i + 1}`, text: q, options }));
}

/** Class links: one link for a whole class; every student who opens it gets a session of their own. */
function Classes({ list, reload }: { list?: Klass[]; reload: () => void }) {
  const [name, setName] = useState("");
  const [policy, setPolicy] = useState("default");
  const [minutes, setMinutes] = useState(15);
  const [questionText, setQuestionText] = useState("");
  const [error, setError] = useState<string>();
  const questions = parseQuestions(questionText);
  const create = (e: React.FormEvent) => {
    e.preventDefault();
    const exam = questions.length ? { minutes, questions } : undefined;
    api("/classes", { json: { name, policy_id: policy, exam } }).then(
      () => (setName(""), setQuestionText(""), setError(undefined), reload()), (x: Error) => setError(x.message));
  };
  const setOpen = (k: Klass, open: boolean) => api(`/classes/${k.id}/open`, { json: { open } }).then(reload);
  return (
    <section className="card space-y-3">
      <h2 className="font-semibold">Class links <span className="muted text-sm font-normal">one link for the whole class; each student who opens it gets their own exam</span></h2>
      <form onSubmit={create} className="space-y-3">
        <div className="flex flex-wrap items-start gap-3">
          <label className="block min-w-48 flex-1 text-sm">Class or exam name<input className="input mt-1" value={name} onChange={(e) => setName(e.target.value)} required maxLength={100} placeholder="e.g. Physics 10B, mid-term" /></label>
          <PolicySelect value={policy} onChange={setPolicy} />
          <button className="btn-primary mt-6">Create class link</button>
        </div>
        <details className="text-sm">
          <summary className="muted cursor-pointer">Use your own questions (optional; otherwise the built-in practice questions)</summary>
          <div className="mt-2 space-y-2">
            <label className="block w-32">Minutes<input type="number" className="input mt-1" min={1} max={240} value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} /></label>
            <label className="block">One question per line: the question, then each answer choice, separated by |
              <textarea className="input mt-1 h-28 font-mono text-xs" value={questionText} onChange={(e) => setQuestionText(e.target.value)}
                       placeholder="What is 2 + 2? | 3 | 4 | 5" />
            </label>
            <p className="muted text-xs">{questionText.trim() ? `${questions.length} question(s) recognized` : "No questions entered: the built-in practice questions will be used"}</p>
          </div>
        </details>
      </form>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {list?.map((k) => (
        <div key={k.id} className="space-y-1 border-t border-stone-200 pt-3 dark:border-stone-800">
          <div className="flex flex-wrap items-center gap-2">
            <b>{k.name}</b>
            <span className="muted text-sm">{k.n_students} student{k.n_students === 1 ? "" : "s"}{k.n_live ? `, ${k.n_live} in the exam now` : ""}{k.exam ? `, ${k.exam.questions.length} custom question(s)` : ""}</span>
            {!k.open && <span className="rounded border border-stone-400 px-1.5 text-xs">closed to new students</span>}
            <span className="ml-auto" />
            <a className="btn-primary py-1" href={`#/class/${k.id}`}>Watch class</a>
            <a className="btn-ghost py-1" href={`/api/classes/${k.id}/decisions.csv`}>Download flags (CSV)</a>
            <a className="btn-ghost py-1" href={`/api/classes/${k.id}/evidence.zip`}>Download evidence (zip)</a>
            <button className="btn-ghost py-1" onClick={() => setOpen(k, !k.open)} title={k.open ? "Stops new students from joining with this link" : "Lets new students join with this link again"}>{k.open ? "Close link" : "Reopen link"}</button>
            <button className="btn-ghost py-1 text-red-700 dark:text-red-400" onClick={() => confirm(`Delete class ${k.name} and the exams of all ${k.n_students} student(s), with their clips, pictures and flags? This cannot be undone.`) && api(`/classes/${k.id}`, { method: "DELETE" }).then(reload, (e: Error) => alert(`Could not delete: ${e.message}`))}>Delete</button>
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
    <Shell title={<>Class: {k?.name ?? cid} <span className="muted text-sm font-normal">{students ? `${students.length} student${students.length === 1 ? "" : "s"}` : ""}</span></>}
      hint="Watch every student of this class live. Students with a flag now have a red border; click a camera to enlarge it.">

      {k?.open && <div className="card space-y-1"><p className="text-sm">Students join with this one link:</p><JoinLink url={k.join_url} /></div>}
      {students?.length ? <StudentGrid groups={[["", students]]} /> : students && <p className="card muted">No students yet. Send them the link above.</p>}
    </Shell>
  );
}

const SINGLE = "Single-student links";

/** Every student taking (or setting up) an exam right now, all classes and single links, in one window. */
export function WatchAll() {
  const { data } = useApi<Session[]>("/sessions?active=1", 2000);
  const { data: classes } = useApi<Klass[]>("/classes", 10000);
  const names = new Map((classes ?? []).map((k) => [k.id, k.name]));
  const active = (data ?? []).filter((s) => s.status === "live" || (s.status === "created" && s.phase !== "created"));
  const byGroup = new Map<string, Session[]>();
  for (const st of active) {
    const g = names.get(st.exam_id) ?? SINGLE;
    byGroup.set(g, [...(byGroup.get(g) ?? []), st]);
  }
  const order = (a: Session, b: Session) =>
    (a.roll ?? "").localeCompare(b.roll ?? "", undefined, { numeric: true }) || a.candidate_label.localeCompare(b.candidate_label);
  const groups = [...byGroup].sort(([a], [b]) => (a === SINGLE ? 1 : b === SINGLE ? -1 : a.localeCompare(b)))
    .map(([g, list]) => [g, list.sort(order)] as [string, Session[]]);
  return (
    <Shell title={<>Watch all students <span className="muted text-sm font-normal">{data ? `${active.length} in an exam or getting ready` : ""}</span></>}
      hint="Every student in an exam right now, from all classes. Students with a flag now have a red border and are listed on the right.">
      {active.length ? <StudentGrid groups={groups} /> : data && <p className="card muted">No student is in an exam right now. Send a class link to your students; they appear here as they join.</p>}
    </Shell>
  );
}

/** Tiles for many students at once: live camera (detector view), flags in progress, recent observations. Click a
 * camera to enlarge that student in place; Fullscreen puts the whole wall on screen. */
function StudentGrid({ groups }: { groups: [string, Session[]][] }) {
  const [big, setBig] = useState<string>();
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [log, setLog] = useState<{ at: number; sid: string; text: string }[]>([]);
  const wall = useRef<HTMLDivElement>(null);
  const all = groups.flatMap(([, l]) => l);
  const count = all.length;
  const byId = new Map(all.map((x) => [x.id, x]));
  useProctorFeed((m) => { // every flag, newest first, so the proctor does not have to scan all the cameras
    if (m.type !== "event_started" || !byId.has(m.session_id)) return;
    setLog((l) => [{ at: Date.now(), sid: m.session_id, text: `${LABEL[m.event.type] ?? m.event.type} at ${ts(m.event.start_ms)}` }, ...l].slice(0, 50));
  });
  const show = (sid: string) => (setBig(sid), document.getElementById(`tile-${sid}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  const flagged = all.filter((x) => x.ongoing?.length);
  const slow = count > 12; // ponytail: picture refresh 2 s above 12 students (server draws each frame)
  return (
    <div ref={wall} className="grid gap-4 overflow-auto bg-stone-50 lg:grid-cols-[1fr_18rem] dark:bg-stone-950 [&:fullscreen]:p-4">
      <div className="space-y-4">
      <div className="flex items-center justify-end gap-3">
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={onlyFlagged} onChange={(e) => setOnlyFlagged(e.target.checked)} /> Only students with a flag now ({flagged.length})</label>
        <button className="btn-ghost py-1" onClick={() => (document.fullscreenElement ? document.exitFullscreen() : wall.current?.requestFullscreen())}>
          Fullscreen
        </button>
      </div>
      {groups.map(([title, list]) => (
        <section key={title || "all"} className="space-y-2">
          {title && <h2 className="font-semibold">{title} <span className="muted text-sm font-normal">{list.length}</span></h2>}
          <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-3">
            {[...list].sort((a, b) => Number(!!b.ongoing?.length) - Number(!!a.ongoing?.length)) // flagged first
              .filter((st) => !onlyFlagged || st.ongoing?.length).map((st) => <StudentTile key={st.id} s={st} slow={slow} big={big === st.id} setBig={setBig} />)}
          </div>
        </section>
      ))}
      </div>
      <aside className="card h-fit space-y-3 p-3 lg:sticky lg:top-2" aria-label="Flags">
        <h2 className="font-semibold">Flagged now <span className="muted text-sm font-normal">{flagged.length}</span></h2>
        {flagged.length ? flagged.flatMap((x) => (x.ongoing ?? []).map((o) => (
          <button key={x.id + o.type} onClick={() => show(x.id)} className="block w-full rounded border border-amber-500 bg-amber-50 p-2 text-left text-sm dark:bg-amber-950">
            <b>{who(x)}</b><br />{LABEL[o.type] ?? o.type} <span className="muted">since {ts(o.start_ms)}</span>
          </button>))) : <p className="muted text-sm">No flags right now.</p>}
        <h2 className="font-semibold">Recent</h2>
        {log.length ? log.map((e, i) => (
          <button key={i} onClick={() => show(e.sid)} className="block w-full text-left text-sm hover:underline">
            <span className="muted">{new Date(e.at).toLocaleTimeString()}</span> <b>{byId.get(e.sid) ? who(byId.get(e.sid)!) : e.sid}</b>: {e.text}
          </button>)) : <p className="muted text-sm">Flags appear here as they happen.</p>}
      </aside>
    </div>
  );
}

const StudentTile = memo(function StudentTile({ s, slow, big, setBig }: {
  s: Session; slow: boolean; big: boolean; setBig: (id: string | undefined) => void;
}) {
  const live = s.status === "live";
  const video = useLiveVideo(s.id, live);
  const toggle = () => setBig(big ? undefined : s.id);
  const enlarge = `${big ? "Shrink" : "Enlarge"} ${who(s)}'s camera`;
  return (
    <div id={`tile-${s.id}`} className={`card space-y-2 p-3 ${big ? "col-span-full" : ""} ${s.ongoing?.length ? "border-2 border-red-500" : ""}`}>
      <div className="flex items-center gap-2">
        <b className="truncate">{who(s)}</b>
        <span className={`rounded px-2 py-0.5 text-xs ${STATUS_CLASS[s.status]}`}>{statusWord(s)}</span>
      </div>
      {live && video && (
        <button className="block w-full" onClick={toggle} aria-label={enlarge}>
          <LiveVideo stream={video} label={`${who(s)}: live camera`} className={`w-full rounded bg-black -scale-x-100 ${big ? "mx-auto max-h-[75vh] w-auto" : ""}`} />
        </button>
      )}
      {live && !video && <TileFrame sid={s.id} name={who(s)} slow={slow} big={big} toggle={toggle} label={enlarge} />}
      {!live && <div className="muted flex aspect-video items-center justify-center rounded bg-stone-100 text-sm dark:bg-stone-800">{s.status === "ended" ? "Exam finished" : "Getting ready (consent, camera check)"}</div>}
      <Ongoing items={s.ongoing} />
      <div className="flex flex-wrap gap-2 text-sm">
        {s.status !== "ended" && <a className="btn-ghost py-1" href={`#/live/${s.id}`}>Watch live</a>}
        {s.status !== "created" && <a className="btn-ghost py-1" href={`#/review/${s.id}`}>Review</a>}
        {s.n_events > 0 && <span className="muted self-center">{s.n_events} flag{s.n_events === 1 ? "" : "s"} so far</span>}
      </div>
    </div>
  );
});

/** Picture fallback when there is no live video: the only part of a tile that re-renders on the refresh tick. */
function TileFrame({ sid, name, slow, big, toggle, label }: {
  sid: string; name: string; slow: boolean; big: boolean; toggle: () => void; label: string;
}) {
  const [n, setN] = useState(0);
  const [hasFrame, setHasFrame] = useState(false);
  useEffect(() => {
    const id = setInterval(() => setN((x) => x + 1), slow ? 2000 : 1000);
    return () => clearInterval(id);
  }, [slow]);
  return (
    <>
      <button className={`block w-full ${hasFrame ? "" : "hidden"}`} onClick={toggle} aria-label={label}>
        <img src={`/api/sessions/${sid}/frame.jpg?overlay=1&n=${n}`} onLoad={() => setHasFrame(true)} onError={() => setHasFrame(false)}
          alt={`${name}: latest camera picture with what the AI sees`} className={`w-full rounded bg-black ${big ? "mx-auto max-h-[75vh] w-auto" : ""}`} />
      </button>
      {!hasFrame && <div className="muted flex aspect-video items-center justify-center rounded bg-stone-100 text-sm dark:bg-stone-800">Waiting for camera...</div>}
    </>
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
      <h2 className="font-semibold">Link for one student <span className="muted text-sm font-normal">for a student who is not in a class link</span></h2>
      <div className="flex flex-wrap items-start gap-3">
        <label className="block min-w-48 flex-1 text-sm">Student name or code<input className="input mt-1" value={label} onChange={(e) => setLabel(e.target.value)} required maxLength={100} placeholder="e.g. Roll 23 or Asha" /></label>
        <PolicySelect value={policy} onChange={setPolicy} />
        <button className="btn-primary mt-6">Create student link</button>
      </div>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {made && (
        <div className="space-y-1">
          <p className="text-sm">Send this link to <b>{made.label}</b>. It opens the exam in a browser on any computer or phone:</p>
          <JoinLink url={made.join_url} />
        </div>
      )}
    </form>
  );
}

type FeedItem = { at: number; text: string; type: string };
const num = (v: unknown, f: (x: number) => string) => (typeof v === "number" ? f(v) : "n/a");
/** Picture quality in words (score 0..1). */
export const qualityWord = (q: unknown) => num(q, (v) => (v >= 0.7 ? "Good" : v >= 0.4 ? "Fair" : "Poor"));
const lookingAt = (zone: unknown) => (!zone || zone === "none" ? "n/a" : zone === "on_screen" ? "screen" : "away");
const seen = (conf: unknown) => num(conf, (v) => (v >= 0.3 ? "yes" : "no")); // 0.3 = models.detector_conf, the detector's own cut-off

export function Live({ sid }: { sid: string }) {
  const { data: s } = useApi<Session>(`/sessions/${sid}`, 3000);
  const [frame, setFrame] = useState(0);
  const [hasFrame, setHasFrame] = useState(true);
  const [overlay, setOverlay] = useState(false);
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
  const video = useLiveVideo(sid, !overlay && s?.status === "live");
  return (
    <Shell title={<>Watch live: {s ? who(s) : sid} <span className="muted text-sm font-normal">{s?.status === "ended" ? "(finished)" : phaseWord(status?.phase ?? s?.phase)}</span></>}
      hint="Follow this student's camera during the exam. Flags appear on the right as they happen; review them after the exam.">
      {!!s?.ongoing?.length && <div className="flex items-center gap-2 text-sm"><span className="font-medium">Flagged now:</span><Ongoing items={s.ongoing} /></div>}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="card space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} />
              Show what the AI sees (boxes and face outline)
            </label>
            {overlay && <span className="muted text-xs">The AI can be wrong. Numbers on boxes only follow them and do not identify anyone. This view is not mirrored.</span>}
          </div>
          {video && <LiveVideo stream={video} label="live camera" className="w-full rounded bg-black -scale-x-100" />}
          {/* reloaded every second; hidden while there is no frame, so it shows up again by itself */}
          {!video && <img src={`/api/sessions/${sid}/frame.jpg?n=${frame}${overlay ? "&overlay=1" : ""}`} onError={() => setHasFrame(false)}
            onLoad={() => setHasFrame(true)} alt={overlay ? "latest camera picture with what the AI sees" : "latest camera picture"}
            className={hasFrame ? `w-full rounded bg-black ${overlay ? "" : "-scale-x-100"}` : "hidden"} />}
          {!video && !hasFrame && (
            <div className="muted flex aspect-video items-center justify-center rounded bg-stone-100 px-4 text-center dark:bg-stone-800">
              No picture: the student has not connected yet, has lost the connection, or the exam has ended.
            </div>
          )}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-4">
            <dt className="muted">Picture quality</dt><dd>{qualityWord(status?.quality)}</dd>
            <dt className="muted">Looking at</dt><dd>{lookingAt(status?.zone)}</dd>
            <dt className="muted">Faces seen</dt><dd>{num(status?.n_faces, String)}</dd>
            <dt className="muted">People seen</dt><dd>{num(status?.n_persons, String)}</dd>
            <dt className="muted">Phone seen</dt><dd>{seen(status?.phone_conf)}</dd>
            <dt className="muted">Notes seen</dt><dd>{seen(status?.notes_conf)}</dd>
          </dl>
          <details className="text-sm">
            <summary className="muted cursor-pointer">Technical details</summary>
            <dl className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
              <dt className="muted">Image quality score</dt><dd>{num(status?.quality, (v) => v.toFixed(2))}</dd>
              <dt className="muted">Frame rate</dt><dd>{num(status?.fps, (v) => `${v.toFixed(1)} fps`)}</dd>
              <dt className="muted">Attention zone</dt><dd>{status?.zone ?? "n/a"}</dd>
              <dt className="muted">Head turn (yaw / pitch)</dt><dd>{num(status?.d_yaw, (v) => `${v.toFixed(0)} deg`)} / {num(status?.d_pitch, (v) => `${v.toFixed(0)} deg`)}</dd>
              <dt className="muted">Phone detector</dt><dd>{num(status?.phone_conf, (v) => v.toFixed(2))}</dd>
              <dt className="muted">Book / notes detector</dt><dd>{num(status?.notes_conf, (v) => v.toFixed(2))}</dd>
              <dt className="muted">Processing</dt><dd>{num(status?.proc_ms, (v) => `${v.toFixed(0)} ms / frame`)}</dd>
              <dt className="muted">Quality notes</dt><dd>{(status?.reasons as string[] | undefined)?.join(", ") || "none"}</dd>
            </dl>
          </details>
        </div>
        <div className="card space-y-2">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Flags as they happen</h2>
            {s?.status !== "created" && <a className="btn-ghost py-1" href={`#/review/${sid}`}>Review flags</a>}
          </div>
          <p className="muted text-xs">Flags are for you to review later; none of them is a finding about the student.</p>
          <ul className="max-h-[60vh] space-y-1 overflow-y-auto text-sm" aria-live="polite">
            {feed.map((f) => (
              <li key={f.at + f.text} className={f.type === "event_started" ? "text-amber-800 dark:text-amber-300" : ""}>
                <span className="muted mr-2 font-mono text-xs">{new Date(f.at).toLocaleTimeString()}</span>{f.text}
              </li>
            ))}
            {!feed.length && <li className="muted">No flags since this page opened.</li>}
          </ul>
        </div>
      </div>
    </Shell>
  );
}

export function LiveVideo({ stream, label, className }: { stream: MediaStream; label: string; className: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.srcObject = stream;
  }, [stream]);
  return <video ref={ref} autoPlay muted playsInline aria-label={label} className={className} />;
}
