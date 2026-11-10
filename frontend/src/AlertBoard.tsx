import { useEffect, useRef, useState } from "react";
import { LiveVideo, Shell } from "./Proctor";
import { alertName, alertWho, chime, enableAlerts, LABEL, speak, useApi, useLiveVideo, useProctorFeed, type Session } from "./lib";

// The staff-room screen: who is flagged right now, in letters readable from across the room, and a history of
// flags. Sound, voice and pop-ups themselves come from the header's Alerts (every proctor page has them).

type Flag = { at: number; sid: string; who: string; what: string };
const KEY = "pl.flags"; // sessionStorage: the history survives a reload of this page
const load = (): Flag[] => {
  try {
    return JSON.parse(sessionStorage.getItem(KEY) ?? "[]");
  } catch {
    return [];
  }
};
const byRoll = (a: Session, b: Session) =>
  (a.roll ?? "").localeCompare(b.roll ?? "", undefined, { numeric: true }) || a.candidate_label.localeCompare(b.candidate_label);
const secs = (s: number) => (s < 60 ? `${Math.max(0, Math.round(s))} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

/** The flagged student's camera: live video (straight from their browser); the 1-a-second picture while it connects or
 * when the network blocks it. */
function Camera({ s }: { s: Session }) {
  const video = useLiveVideo(s.id, true);
  const [n, setN] = useState(0);
  useEffect(() => {
    if (video) return;
    const id = setInterval(() => setN((x) => x + 1), 1000);
    return () => clearInterval(id);
  }, [!!video]);
  return video
    ? <LiveVideo stream={video} label={`${alertName(s)}: live camera`} className="mb-3 w-full rounded bg-black -scale-x-100" />
    : <img src={`/api/sessions/${s.id}/frame.jpg?overlay=1&n=${n}`} alt={`${alertName(s)}: latest camera picture`} className="mb-3 w-full rounded bg-black" onError={(e) => (e.currentTarget.style.display = "none")} onLoad={(e) => (e.currentTarget.style.display = "")} />;
}

export function AlertBoard() {
  const { data } = useApi<Session[]>("/sessions?active=1", 1500);
  const [flags, setFlags] = useState<Flag[]>(load);
  const [armed, setArmed] = useState(false);
  const [, tick] = useState(0);
  const sessions = useRef<Session[]>([]);
  sessions.current = data ?? [];
  const seen = useRef(new Map<string, number>()); // "student|flag|start" -> when this page first saw it: "for 12 s"

  useEffect(() => {
    const id = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  useEffect(() => {
    try {
      sessionStorage.setItem(KEY, JSON.stringify(flags));
    } catch {
      /* private window: the history lasts until reload */
    }
  }, [flags]);
  useProctorFeed((m) => {
    if (m.type !== "event_started") return;
    const s = sessions.current.find((x) => x.id === m.session_id);
    setFlags((f) => [{ at: Date.now(), sid: m.session_id, who: s ? alertWho(s) : m.session_id, what: LABEL[m.event.type] ?? m.event.type }, ...f].slice(0, 100));
  });
  useEffect(() => { // keep the staff-room screen awake once alerts are on (and again after the tab was hidden)
    const wake = (navigator as any).wakeLock;
    if (!armed || !wake) return;
    let lock: { release(): void } | undefined;
    const get = () => document.visibilityState === "visible" && wake.request("screen").then((l: typeof lock) => (lock = l), () => undefined);
    void get();
    document.addEventListener("visibilitychange", get);
    return () => {
      document.removeEventListener("visibilitychange", get);
      lock?.release();
    };
  }, [armed]);

  const all = data ?? [];
  const inExam = all.filter((s) => s.status === "live").length;
  const flagged = all.filter((s) => s.ongoing?.length).sort(byRoll);
  const now = Date.now();
  for (const s of flagged) for (const o of s.ongoing!) {
    const k = `${s.id}|${o.type}|${o.start_ms}`;
    if (!seen.current.has(k)) seen.current.set(k, now);
  }

  return (
    <Shell title="Alert board" hint="Leave this page open. Every flag appears here the moment it starts, with the roll number, and can be read aloud.">
      <div className="card flex flex-wrap items-center gap-3">
        <span>{armed ? "Alerts are on: sound, voice and pop-ups. This screen stays awake." : "Turn on alerts to hear and see flags even when another window is in front."}</span>
        <span className="ml-auto" />
        {armed
          ? <button className="btn-ghost py-1" onClick={() => (chime(), speak("Roll 45. Test alert."))}>Send a test alert</button>
          : <button className="btn-primary py-2 text-base" onClick={() => (void enableAlerts(), setArmed(true))}>Turn on alerts</button>}
      </div>
      <p className="text-sm">
        <b>{inExam}</b> in the exam · <b className={flagged.length ? "text-red-700 dark:text-red-400" : ""}>{flagged.length}</b> flagged now
      </p>

      <section className="space-y-2" aria-label="Flagged now">
        <h2 className="font-semibold">Flagged now</h2>
        {flagged.length ? (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(20rem,1fr))] gap-3">
            {flagged.map((s) => (
              <a key={s.id} href={`#/live/${s.id}`} className="card block border-2 border-red-500 p-4 hover:bg-red-50 dark:hover:bg-red-950">
                <Camera s={s} />
                <div className="text-3xl font-bold">{alertName(s)}</div>
                {s.roll && <div className="muted">{s.candidate_label}</div>}
                <div className="mt-2 space-y-1">
                  {s.ongoing!.map((o) => (
                    <div key={o.type}>
                      <b>{LABEL[o.type] ?? o.type}</b>{" "}
                      <span className="muted">for {secs((now - seen.current.get(`${s.id}|${o.type}|${o.start_ms}`)!) / 1000)}</span>
                    </div>
                  ))}
                </div>
                <span className="btn-ghost mt-3 inline-block py-1 text-sm">Watch live</span>
              </a>
            ))}
          </div>
        ) : (
          <p className="card muted">{inExam ? "No flags right now." : "No student is in an exam yet. Flags appear here as soon as students start."}</p>
        )}
      </section>

      <section className="space-y-2" aria-label="Recent flags">
        <h2 className="font-semibold">Recent flags <span className="muted text-sm font-normal">newest first; click one to watch that student</span></h2>
        {flags.length ? (
          <ul className="card divide-y divide-stone-200 p-0 dark:divide-stone-800">
            {flags.map((f, i) => (
              <li key={i}>
                <a href={`#/live/${f.sid}`} className="flex flex-wrap gap-x-3 px-3 py-2 hover:bg-stone-100 dark:hover:bg-stone-800">
                  <span className="muted tabular-nums">{new Date(f.at).toLocaleTimeString()}</span>
                  <b>{f.who}</b>
                  <span>{f.what}</span>
                </a>
              </li>
            ))}
          </ul>
        ) : (
          <p className="card muted">Nothing flagged yet. Flags are observations to review, not findings.</p>
        )}
      </section>
    </Shell>
  );
}
