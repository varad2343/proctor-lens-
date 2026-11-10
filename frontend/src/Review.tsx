import { useEffect, useMemo, useRef, useState } from "react";
import { policyName, qualityWord, Shell } from "./Proctor";
import { api, dur, LABEL, PRIORITY_CLASS, REVIEW_TYPES, ts, useApi, who, type Ev, type Seg, type Session, type Timeline } from "./lib";

// Session review (master spec 11.2): timeline, ranked segments, evidence, decisions. Keys: J/K next/previous event,
// C confirm, D dismiss, N needs more info. Wording stays observational: decisions are about the evidence.
// The values sent to the API stay confirm / dismiss / needs_more_info; only the button words are plain.
const DECISIONS = [
  { key: "c", value: "confirm", label: "Real concern" },
  { key: "d", value: "dismiss", label: "Not a concern" },
  { key: "n", value: "needs_more_info", label: "Not sure" },
] as const;
const decisionWord = (v: string) => DECISIONS.find((x) => x.value === v)?.label ?? v.replace(/_/g, " ");
const BLIND = "Camera unclear";
const BLIND_TIP = "Time the camera was dark, covered or frozen";
const kind = (t: string) => (t === "MONITORING_DEGRADED" ? BLIND : LABEL[t] ?? t);
const MOMENT = { onset: "start", peak: "clearest moment", end: "end" } as const;
const EVIDENCE_WAIT_MS = 180000; // browser mode: clips and pictures can arrive up to ~3 minutes after the end
const DECISION_CLASS: Record<string, string> = {
  confirm: "bg-sky-700 text-white", dismiss: "bg-stone-300 text-stone-900", needs_more_info: "bg-amber-400 text-stone-950",
};
const clipWindow = (e: Ev) => {
  const a = Math.max(0, e.start_ms - 5000); // = explain/evidence.py clip_window
  return [a, Math.min(e.end_ms + 3000, a + 30000)] as const;
};

export function Review({ sid }: { sid: string }) {
  const { data: s } = useApi<Session>(`/sessions/${sid}`, 5000);
  const live = s?.status === "live";
  const [evData, setEvData] = useState<Ev[]>();
  // after the end, keep refreshing while some flag still waits for its clip or pictures (browser mode uploads them
  // after the exam), for at most EVIDENCE_WAIT_MS; re-checked on every render (the session poll re-renders each 5 s)
  const waiting = s?.status === "ended" && !!s.ended_at && Date.now() - Date.parse(s.ended_at) < EVIDENCE_WAIT_MS &&
    (evData ?? []).some((e) => !e.clip_path || !Object.keys(e.thumbs).length);
  const every = live || waiting ? 3000 : 0;
  const evs = useApi<Ev[]>(`/sessions/${sid}/events`, every);
  useEffect(() => setEvData(evs.data), [evs.data]);
  const segs = useApi<Seg[]>(`/sessions/${sid}/segments`, every);
  const tl = useApi<Timeline>(s && !live ? `/sessions/${sid}/timeline` : null, waiting ? 3000 : 0); // not while live
  const [sel, setSel] = useState<number>();
  const [type, setType] = useState("");
  const [prio, setPrio] = useState("");
  const [open, setOpen] = useState(false);
  const [zoom, setZoom] = useState<[number, number]>();
  const [playhead, setPlayhead] = useState<number>();
  const [seek, setSeek] = useState<{ t: number; n: number }>(); // n: a new click on the same spot seeks again
  const [speed, setSpeed] = useState(1);

  const byId = useMemo(() => new Map((evs.data ?? []).map((e) => [e.id, e])), [evs.data]);
  const shown = (segs.data ?? []).filter(
    (g) => (!type || g.types.includes(type)) && (!prio || (g.lane === "review" && g.priority_label === prio)) &&
      (!open || g.events.some((id) => !byId.get(id)?.review)),
  );
  const order = shown.flatMap((g) => g.events);
  const cur = sel === undefined ? undefined : byId.get(sel);
  useEffect(() => {
    if (sel === undefined && order.length) setSel(order[0]);
  }, [order.length]);

  const decide = (decision: string, note: string) =>
    cur && api(`/events/${cur.id}/review`, { json: { decision, note } }).then(evs.reload);
  const move = (d: number) => order.length && setSel(order[(Math.max(0, order.indexOf(sel ?? -1)) + d + order.length) % order.length]);
  const onSeek = (t: number, id?: number) => {
    // a click on the timeline: open the flag there (the clicked mark, else the open one if it covers t, else the
    // nearest) and jump its clip to t
    const all = evs.data ?? [];
    const near = (e: Ev) => (t < e.start_ms ? e.start_ms - t : t > e.end_ms ? t - e.end_ms : 0);
    const hit = id ?? (cur && near(cur) === 0 ? cur.id : [...all].sort((x, y) => near(x) - near(y))[0]?.id);
    if (hit === undefined) return;
    setSel(hit);
    setSeek((s) => ({ t, n: (s?.n ?? 0) + 1 }));
  };
  const dur_ms = Math.max((tl.data?.t_ms.at(-1) ?? 0) + 100, ...(evs.data ?? []).map((e) => e.end_ms), 1000);

  return (
    <Shell title={<>Review: {s ? who(s) : sid} <span className="muted text-sm font-normal">{live ? "(exam in progress)" : ""}</span></>}
      hint="Open each flag in the list, watch its clip, then choose Real concern, Not a concern or Not sure.">
      {s && <Summary s={s} sid={sid} evs={evs.data ?? []} reload={evs.reload} />}
      <div className="card space-y-1 p-3">
        <div className="flex items-center justify-between text-xs">
          <span className="muted">Each row is one kind of flag. Click the timeline to jump the clip there; drag across it to zoom, double-click to reset.</span>
          {zoom && <button className="btn-ghost py-0.5 text-xs" onClick={() => setZoom(undefined)}>Reset zoom</button>}
        </div>
        <div className="overflow-x-auto">{/* min width keeps the 12-unit labels >= 10 px; narrower cards scroll */}
          <TimelineView events={evs.data ?? []} segs={segs.data ?? []} tl={tl.data} dur={dur_ms} sel={sel} onSeek={onSeek}
            zoom={zoom} setZoom={setZoom} playhead={playhead} />
        </div>
      </div>
      <div className="grid gap-4 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <aside className="card space-y-3 self-start p-3 lg:sticky lg:top-3 lg:max-h-[calc(100vh-1.5rem)] lg:overflow-y-auto">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <select className="input" value={type} onChange={(e) => setType(e.target.value)} aria-label="show only one kind of flag">
              <option value="">All kinds of flag</option>
              {REVIEW_TYPES.map((t) => <option key={t} value={t}>{kind(t)}</option>)}
            </select>
            <select className="input" value={prio} onChange={(e) => setPrio(e.target.value)} aria-label="show only one priority">
              <option value="">All priorities</option>
              {["high", "medium", "low"].map((p) => <option key={p} value={p}>{p} priority</option>)}
            </select>
            <label className="col-span-2 flex items-center gap-2"><input type="checkbox" checked={open} onChange={(e) => setOpen(e.target.checked)} /> Only flags not reviewed yet</label>
          </div>
          <SegmentList segs={shown} byId={byId} sel={sel} onSelect={setSel} />
        </aside>
        <section className="min-w-0">
          {cur ? (
            <Evidence key={cur.id} sid={sid} e={cur} decide={decide} move={move} setPlayhead={setPlayhead} playhead={playhead}
              seek={seek} speed={speed} setSpeed={setSpeed} />
          ) : (
            <p className="card muted">{evs.data?.length === 0 ? "Nothing was flagged in this exam." : "Select a flag in the list to see its clip."}</p>
          )}
        </section>
      </div>
    </Shell>
  );
}

function Summary({ s, sid, evs, reload }: { s: Session; sid: string; evs: Ev[]; reload: () => void }) {
  const [note, setNote] = useState(s.review_note);
  const [saved, setSaved] = useState(true);
  const blind = evs.filter((e) => e.type === "MONITORING_DEGRADED" && !e.review);
  const dismissBlind = async () => {
    for (const e of blind) await api(`/events/${e.id}/review`, { json: { decision: "dismiss", note: "bulk: blind spot" } });
    reload();
  };
  const saveNote = () => api(`/sessions/${sid}/note`, { method: "PUT", json: { note } }).then(() => setSaved(true));
  const c = s.calibration;
  const calib = !c.mode || c.mode === "none" ? "not done" : c.accepted ? "eye tracking set up" : "head direction only";
  const pol = Object.entries(s.policy).filter(([k, v]) => k.startsWith("allow_") && v).map(([k]) => k.replace("allow_", "").replace(/_/g, " "));
  const ended = s.status === "ended";
  return (
    <div className="card grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
      <Stat k="Duration" v={dur(s.duration_s)} />
      <Stat k="Flagged for review" v={`${dur(s.flagged_s)}${s.duration_s ? ` (${Math.round((100 * s.flagged_s) / s.duration_s)} %)` : ""}`} />
      <Stat k={BLIND} v={s.blind_spot_s ? dur(s.blind_spot_s) : "none"} title={BLIND_TIP} />
      <Stat k="Reviewed" v={`${s.n_reviewed} of ${s.n_events} flags`} />
      <Stat k="Eye calibration" v={calib} />
      <Stat k="Exam rules" v={`${policyName(s.policy_id)}${pol.length ? `: allows ${pol.join(", ")}` : ""}`} />
      <Stat k="Flags by kind" v={Object.entries(s.counts).map(([t, n]) => `${n} ${kind(t)}`).join(", ") || "none"} />
      <div className="flex flex-wrap items-start justify-end gap-2">
        <span className="flex flex-col items-end">
          <a className={`btn-primary ${ended ? "" : "pointer-events-none opacity-50"}`} href={`/api/sessions/${sid}/report`} target="_blank" rel="noreferrer"
            aria-disabled={!ended}>Download report</a>
          {!ended && <span className="muted text-xs">available when the exam ends</span>}
        </span>
        <a className="btn-ghost" href={`/api/sessions/${sid}/decisions.csv`}>Download flags (CSV)</a>
        <button className="btn-ghost text-red-700 dark:text-red-400" onClick={() => confirm(`Delete the exam of ${who(s)} with all of its clips, pictures and flags? This cannot be undone.`) && api(`/sessions/${sid}`, { method: "DELETE" }).then(() => (location.hash = "#/"), (e: Error) => alert(`Could not delete: ${e.message}`))}>Delete exam</button>
        {blind.length > 0 && (
          <button className="btn-ghost" onClick={dismissBlind} title="Marks every camera-unclear flag not reviewed yet as Not a concern; saved in the report">
            Mark {blind.length} camera-unclear flag{blind.length > 1 ? "s" : ""} as Not a concern
          </button>
        )}
      </div>
      <details className="text-xs sm:col-span-2 lg:col-span-4">
        <summary className="muted cursor-pointer">Technical details</summary>
        <p className="mt-1">Session {s.id} · rules id {s.policy_id} · calibration {c.mode ?? "none"}{c.error != null ? ` (error ${c.error.toFixed(2)})` : ""}</p>
      </details>
      <label className="block text-sm sm:col-span-2 lg:col-span-4">
        <span className="muted text-xs uppercase tracking-wide">Note about this exam (goes into the report)</span>
        <textarea className="input mt-1" rows={2} maxLength={5000} value={note} placeholder="Overall context for this exam, e.g. a known camera problem"
          onChange={(x) => (setNote(x.target.value), setSaved(false))} onBlur={() => !saved && saveNote()} />
        <span className="muted text-xs">{saved ? "Saved" : "Saved when you click outside the box"}</span>
      </label>
      <p className="muted text-xs sm:col-span-2 lg:col-span-4">
        Flags are observations for you to review, not findings. A webcam cannot see hands, laps or second screens; where
        someone looks is only roughly known; accuracy varies with lighting, glasses, skin tone, head coverings and camera
        quality; the sensitivity settings are starting values, not yet tuned for your room.
      </p>
    </div>
  );
}

const Stat = ({ k, v, title }: { k: string; v: string; title?: string }) => (
  <div title={title}><div className="muted text-xs uppercase tracking-wide">{k}</div><div>{v}</div></div>
);

function TimelineView(p: {
  events: Ev[]; segs: Seg[]; tl?: Timeline; dur: number; sel?: number; onSeek: (t: number, id?: number) => void;
  zoom?: [number, number]; setZoom: (z?: [number, number]) => void; playhead?: number;
}) {
  const [a, b] = p.zoom ?? [0, p.dur];
  const L = 200, W = 1000, lh = 16, sigH = 34;
  const H = lh * (REVIEW_TYPES.length + 1) + sigH + 22;
  const X = (t: number) => L + ((W - L - 10) * (t - a)) / Math.max(b - a, 1);
  const svg = useRef<SVGSVGElement>(null);
  const down = useRef<number>();
  const dragged = useRef(false); // a drag zooms; only a plain click seeks
  const tAt = (clientX: number) => {
    const r = svg.current!.getBoundingClientRect();
    return a + (((clientX - r.left) / r.width) * W - L) / (W - L - 10) * (b - a);
  };
  const step = [10, 30, 60, 120, 300, 600, 1800, 3600].find((s) => (b - a) / 1000 / s <= 8) ?? 7200;
  const ticks: number[] = [];
  for (let k = Math.ceil(a / 1000 / step) * step; k * 1000 <= b; k += step) ticks.push(k * 1000);
  const trace = (name: string) => {
    const t = p.tl?.t_ms ?? [], y = p.tl?.series[name] ?? [];
    const top = lh * (REVIEW_TYPES.length + 1) + 4;
    return t.map((ti, i) => (y[i] == null ? "" : `${X(ti).toFixed(1)},${(top + sigH - 4 - (y[i] as number) * (sigH - 8)).toFixed(1)}`)).filter(Boolean).join(" ");
  };
  const segOpacity = { high: 0.9, medium: 0.5, low: 0.2 }; // ink weight, like the badges (lib.ts PRIORITY_CLASS)
  return (
    <svg ref={svg} viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[840px] select-none text-[12px]" role="img"
      aria-label={`exam timeline: ${p.events.length} flags in ${p.segs.length} groups; the list below opens each one`}
      onPointerDown={(e) => (down.current = e.clientX)}
      onPointerUp={(e) => {
        const d = down.current;
        dragged.current = d !== undefined && Math.abs(e.clientX - d) > 6;
        if (d !== undefined && dragged.current) {
          const [t0, t1] = [tAt(d), tAt(e.clientX)].sort((x, y) => x - y);
          p.setZoom([Math.max(0, t0), Math.min(p.dur, t1)]);
        }
        down.current = undefined;
      }}
      onClick={(e) => !dragged.current && p.onSeek(Math.min(Math.max(tAt(e.clientX), 0), p.dur))}
      onDoubleClick={() => p.setZoom(undefined)}>
      <defs><clipPath id="plot"><rect x={L} y={0} width={W - L} height={H} /></clipPath></defs>
      <text x={L - 6} y={lh - 4} textAnchor="end" className="fill-current opacity-70">Flags to review</text>
      {REVIEW_TYPES.map((t, i) => (
        <text key={t} x={L - 6} y={(i + 2) * lh - 4} textAnchor="end" className="fill-current opacity-70">{kind(t)}</text>
      ))}
      <text x={L - 6} y={lh * (REVIEW_TYPES.length + 1) + sigH / 2 + 4} textAnchor="end" className="fill-current opacity-70">
        <tspan fill="#0369a1">looking away</tspan> / <tspan fill="#059669">picture quality</tspan>
      </text>
      <g clipPath="url(#plot)">
        {REVIEW_TYPES.map((_, i) => <line key={i} x1={L} x2={W} y1={(i + 2) * lh} y2={(i + 2) * lh} stroke="currentColor" strokeOpacity={0.08} />)}
        {p.segs.filter((g) => g.lane === "review").map((g) => ( /* blind spots have their own lane below */
          <rect key={g.id} x={X(g.start_ms)} y={2} width={Math.max(2, X(g.end_ms) - X(g.start_ms))} height={lh - 4} rx={2}
            fill="currentColor" fillOpacity={segOpacity[g.priority_label]} className="cursor-pointer" onClick={(x) => (x.stopPropagation(), !dragged.current && p.onSeek(tAt(x.clientX), g.events[0]))}>
            <title>{`${g.priority_label} priority: ${ts(g.start_ms)} - ${ts(g.end_ms)}`}</title>
          </rect>
        ))}
        {p.events.map((e) => {
          const row = REVIEW_TYPES.indexOf(e.type as (typeof REVIEW_TYPES)[number]) + 1;
          return (
            <rect key={e.id} x={X(e.start_ms)} y={row * lh + 3} width={Math.max(3, X(e.end_ms) - X(e.start_ms))} height={lh - 6} rx={2}
              fill={e.type === "MONITORING_DEGRADED" ? "#64748b" : "#0369a1"} stroke={e.id === p.sel ? "currentColor" : "none"} strokeWidth={2}
              className="cursor-pointer" onClick={(x) => (x.stopPropagation(), !dragged.current && p.onSeek(tAt(x.clientX), e.id))}>
              <title>{`${kind(e.type)}: ${ts(e.start_ms)} - ${ts(e.end_ms)}`}</title>
            </rect>
          );
        })}
        <polyline points={trace("off_screen_score")} fill="none" stroke="#0369a1" strokeWidth={1} />
        <polyline points={trace("quality")} fill="none" stroke="#059669" strokeWidth={1} />
        {p.playhead !== undefined && <line x1={X(p.playhead)} x2={X(p.playhead)} y1={0} y2={H - 18} stroke="currentColor" strokeWidth={1.5} />}
      </g>
      {ticks.map((t) => <text key={t} x={X(t)} y={H - 4} textAnchor="middle" className="fill-current opacity-60">{ts(t)}</text>)}
    </svg>
  );
}

function SegmentList({ segs, byId, sel, onSelect }: { segs: Seg[]; byId: Map<number, Ev>; sel?: number; onSelect: (id: number) => void }) {
  if (!segs.length) return <p className="muted text-sm">No flags match these filters.</p>;
  return (
    <ol className="space-y-2">
      {segs.map((g) => (
        <li key={g.id} className="rounded-md border border-stone-200 p-2 dark:border-stone-700">
          <div className="flex items-center gap-2 text-sm">
            <span className={`rounded px-1.5 text-xs ${PRIORITY_CLASS[g.lane === "blind_spot" ? "blind" : g.priority_label]}`}
              title={g.lane === "blind_spot" ? BLIND_TIP : "Sets the review order; not a judgement"}>
              {g.lane === "blind_spot" ? BLIND.toLowerCase() : `${g.priority_label} priority`}
            </span>
            <span className="font-mono text-xs">{ts(g.start_ms)} - {ts(g.end_ms)}</span>
          </div>
          <ul className="mt-1 space-y-0.5">
            {g.events.map((id) => {
              const e = byId.get(id);
              if (!e) return null;
              return (
                <li key={id}>
                  <button onClick={() => onSelect(id)} aria-current={id === sel}
                    className={`flex w-full items-center gap-2 rounded px-1.5 py-0.5 text-left text-sm hover:bg-stone-100 dark:hover:bg-stone-800 ${id === sel ? "bg-sky-100 dark:bg-sky-950" : ""}`}>
                    <span className="flex-1">{kind(e.type)}</span>
                    {e.review && <span className={`rounded px-1 text-[10px] ${DECISION_CLASS[e.review.decision]}`}>{decisionWord(e.review.decision)}</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </li>
      ))}
    </ol>
  );
}

function Evidence(p: {
  sid: string; e: Ev; decide: (d: string, note: string) => void; move: (d: number) => void; playhead?: number;
  setPlayhead: (t?: number) => void; seek?: { t: number; n: number }; speed: number; setSpeed: (s: number) => void;
}) {
  const { e } = p;
  const plot = useApi<{ svg: string | null; caption: string | null }>(`/events/${e.id}/plot`);
  const [note, setNote] = useState(e.review?.note ?? "");
  const [boxes, setBoxes] = useState(true); // detector view by default; raw camera frames are one click away
  const clip = (boxes && e.clip_overlay) || e.clip_path;
  const thumb = (k: "onset" | "peak" | "end") => (boxes && e.thumbs_overlay?.[k]) || e.thumbs[k];
  const [a, b] = clipWindow(e);
  const file = (path: string) => `/api/sessions/${p.sid}/files/${path}`;
  const d = e.details;
  const video = useRef<HTMLVideoElement>(null);
  const applySeek = () => {
    const v = video.current;
    if (v && p.seek && p.seek.t >= a && p.seek.t <= b) v.currentTime = (p.seek.t - a) / 1000;
  };
  useEffect(applySeek, [p.seek]);
  useEffect(() => {
    if (video.current) video.current.playbackRate = p.speed;
  }, [p.speed, clip]);
  const step = (dt: number) => {
    const v = video.current;
    if (!v) return;
    v.pause();
    v.currentTime = Math.max(0, v.currentTime + dt);
  };
  useEffect(() => {
    const onKey = (k: KeyboardEvent) => {
      if (k.ctrlKey || k.metaKey || k.altKey || (k.target as HTMLElement).closest?.("input, textarea, select")) return;
      const key = k.key.toLowerCase();
      if (key === ",") step(-0.1); // one frame back / forward (clips run at ~10 fps)
      else if (key === ".") step(0.1);
      else if (key === "j") p.move(1);
      else if (key === "k") p.move(-1);
      else if (DECISIONS.some((x) => x.key === key)) p.decide(DECISIONS.find((x) => x.key === key)!.value, note);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [note, p]);
  useEffect(() => () => p.setPlayhead(undefined), []);
  const flags = (d.benign_flags as string[] | undefined) ?? [];
  return (
    <article className="card space-y-4">
      <header className="flex flex-wrap items-baseline gap-x-3">
        <h2 className="text-lg font-semibold">{kind(e.type)}</h2>
        <span className="font-mono text-sm">{ts(e.start_ms)} - {ts(e.end_ms)}</span>
        <span className="muted ml-auto text-xs">Keys: J / K next / previous flag · , / . frame back / forward</span>
      </header>
      <p className="leading-relaxed">{e.explanation}</p>
      <div className="grid gap-3 xl:grid-cols-2">
        <div>
          {clip ? (
            <>
              <video ref={video} key={clip} controls preload="metadata" src={file(clip)} className="w-full rounded bg-black"
                onLoadedMetadata={(x) => ((x.currentTarget.playbackRate = p.speed), applySeek())}
                onTimeUpdate={(x) => p.setPlayhead(a + x.currentTarget.currentTime * 1000)} />
              <div className="mt-1 flex items-center gap-1 text-xs" role="group" aria-label="playback speed">
                <span className="muted mr-1">Speed</span>
                {[0.5, 1, 2].map((r) => (
                  <button key={r} className={`btn-ghost px-2 py-0.5 text-xs ${p.speed === r ? "bg-stone-200 dark:bg-stone-700" : ""}`}
                    aria-pressed={p.speed === r} onClick={() => p.setSpeed(r)}>{r}x</button>
                ))}
              </div>
            </>
          ) : (
            <p className="muted flex aspect-video items-center justify-center rounded bg-stone-100 px-4 text-center text-sm dark:bg-stone-800">
              {e.status !== "final" ? "The video clip is still being prepared." : Object.keys(e.thumbs).length
                ? "No video clip: this server cannot make clips (ffmpeg is not installed). The still pictures below show the moment."
                : "The clip and pictures are still being prepared, or the camera sent no pictures at this moment. This page checks again by itself for a few minutes."}
            </p>
          )}
          {(e.clip_overlay || Object.keys(e.thumbs_overlay ?? {}).length > 0) && (
            <label className="mt-2 flex items-center gap-2 text-sm">
              <input type="checkbox" checked={boxes} onChange={(x) => setBoxes(x.target.checked)} />
              Show what the AI sees (boxes and face outline)
              {boxes && <span className="muted text-xs">(it can be wrong)</span>}
            </label>
          )}
          <div className="mt-2 grid grid-cols-3 gap-2">
            {(["onset", "peak", "end"] as const).map((k) => thumb(k) ? (
              <a key={k} href={file(thumb(k)!)} target="_blank" rel="noreferrer" className="block">
                <img src={file(thumb(k)!)} alt={`${MOMENT[k]}${boxes ? ", with what the AI sees" : ""}`} className="w-full rounded border border-stone-200 dark:border-stone-700" loading="lazy" />
                <span className="muted text-xs">{MOMENT[k]}</span>
              </a>
            ) : null)}
          </div>
        </div>
        <div className="space-y-3">
          {e.thumbs.reference && e.thumbs.peak && (
            <figure>
              <div className="grid grid-cols-2 gap-2">
                {([["reference", "reference picture (start of exam)"], ["peak", "this moment"]] as const).map(([k, cap]) => (
                  <a key={k} href={file(e.thumbs[k]!)} target="_blank" rel="noreferrer">
                    <img src={file(e.thumbs[k]!)} alt={cap} className="w-full rounded border border-stone-200 dark:border-stone-700" />
                    <span className="muted text-xs">{cap}</span>
                  </a>
                ))}
              </div>
              <figcaption className="muted mt-1 text-xs">Compare the two faces yourself: the similarity score can be wrong with light, angle, glasses or camera changes.</figcaption>
            </figure>
          )}
          {plot.data?.svg && (
            <figure>
              <div className="relative [&_svg]:h-auto [&_svg]:w-full">
                <div dangerouslySetInnerHTML={{ __html: plot.data.svg }} className="rounded border border-stone-200 bg-white dark:border-stone-700" />
                {p.playhead !== undefined && p.playhead >= a && p.playhead <= b && (
                  <div className="absolute inset-y-0 w-px bg-stone-900 dark:bg-stone-100" style={{ left: `${((p.playhead - a) / (b - a)) * 100}%` }} aria-hidden />
                )}
              </div>
              <figcaption className="muted mt-1 text-xs">{plot.data.caption}</figcaption>
            </figure>
          )}
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-sm">
            <dt className="muted">Picture quality</dt><dd>{qualityWord(d.quality_mean)}</dd>
            <dt className="muted">Possible harmless reasons</dt><dd>{flags.length ? flags.join(", ").replace(/_/g, " ") : "none noted"}</dd>
          </dl>
          <details className="text-sm">
            <summary className="muted cursor-pointer">Technical details</summary>
            <div className="mt-1 space-y-2">
              <p className="text-xs">{e.type} · {e.detector}</p>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
                <dt className="muted">Detector confidence (0 to 1)</dt><dd>{e.confidence.toFixed(2)}</dd>
                <dt className="muted">Image quality score</dt><dd>{typeof d.quality_mean === "number" ? d.quality_mean.toFixed(2) : "n/a"}</dd>
                <dt className="muted">Calibration</dt><dd>{String(d.calib_mode ?? "n/a")}{typeof d.calib_error === "number" ? ` (error ${d.calib_error.toFixed(2)})` : ""}</dd>
              </dl>
              {e.attribution && <Attribution a={e.attribution} />}
              <table className="text-xs">
                <caption className="muted text-left">Measured values</caption>
                <tbody>{Object.entries(d).map(([k, v]) => <tr key={k}><th className="muted pr-3 text-left align-top font-normal">{k}</th><td className="break-all">{typeof v === "string" ? v : JSON.stringify(v)}</td></tr>)}</tbody>
              </table>
            </div>
          </details>
        </div>
      </div>
      <footer className="space-y-2 border-t border-stone-200 pt-3 dark:border-stone-800">
        <p className="text-sm">
          {e.review ? <>Last decision: <b>{decisionWord(e.review.decision)}</b> by {e.review.reviewer}, {new Date(e.review.created_at).toLocaleString()}</> : <span className="muted">Not reviewed yet.</span>}
        </p>
        <p className="muted text-xs">Your decision is saved in the report. Nothing is sent to the student.</p>
        <textarea className="input h-16" placeholder="Note (optional): what you saw in the evidence" value={note} onChange={(x) => setNote(x.target.value)} maxLength={2000} aria-label="review note" />
        <div className="flex flex-wrap gap-2">
          {DECISIONS.map((x) => (
            <button key={x.value} className={x.value === "confirm" ? "btn-primary" : "btn-ghost"} onClick={() => p.decide(x.value, note)}
              title={`Keyboard shortcut: ${x.key.toUpperCase()}`}>
              {x.label} <kbd className="ml-1 rounded border border-current px-1 text-[10px] opacity-70">{x.key.toUpperCase()}</kbd>
            </button>
          ))}
        </div>
      </footer>
    </article>
  );
}

function Attribution({ a }: { a: Record<string, number> }) {
  const top = Math.max(...Object.values(a).map(Math.abs), 1e-9);
  return (
    <div className="text-sm">
      <div className="muted text-xs">Score drop when a feature group is neutralized (learned scorer)</div>
      {Object.entries(a).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).map(([k, v]) => (
        <div key={k} className="flex items-center gap-2">
          <span className="w-20">{k.replace("_", " ")}</span>
          <span className="h-2 rounded bg-sky-700" style={{ width: `${(60 * Math.abs(v)) / top}%` }} />
          <span className="font-mono text-xs">{v >= 0 ? "+" : ""}{v.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}
