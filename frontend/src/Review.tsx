import { useEffect, useMemo, useRef, useState } from "react";
import { Shell } from "./Proctor";
import { api, dur, LABEL, PRIORITY_CLASS, REVIEW_TYPES, ts, useApi, type Ev, type Seg, type Session, type Timeline } from "./lib";

// Session review (master spec 11.2): timeline, ranked segments, evidence, decisions. Keys: J/K next/previous event,
// C confirm, D dismiss, N needs more info. Wording stays observational: decisions are about the evidence.
const DECISIONS = [
  { key: "c", value: "confirm", label: "Confirm" },
  { key: "d", value: "dismiss", label: "Dismiss" },
  { key: "n", value: "needs_more_info", label: "Needs more info" },
] as const;
const DECISION_CLASS: Record<string, string> = {
  confirm: "bg-sky-700 text-white", dismiss: "bg-stone-300 text-stone-900", needs_more_info: "bg-amber-400 text-stone-950",
};
const clipWindow = (e: Ev) => {
  const a = Math.max(0, e.start_ms - 5000); // = explain/evidence.py clip_window
  return [a, Math.min(e.end_ms + 3000, a + 30000)] as const;
};

export function Review({ sid }: { sid: string }) {
  const { data: s } = useApi<Session>(`/sessions/${sid}`, 5000);
  const every = s?.status === "live" ? 3000 : 0; // refresh while the exam is still running
  const evs = useApi<Ev[]>(`/sessions/${sid}/events`, every);
  const segs = useApi<Seg[]>(`/sessions/${sid}/segments`, every);
  const tl = useApi<Timeline>(`/sessions/${sid}/timeline`, every);
  const [sel, setSel] = useState<number>();
  const [type, setType] = useState("");
  const [prio, setPrio] = useState("");
  const [open, setOpen] = useState(false);
  const [zoom, setZoom] = useState<[number, number]>();
  const [playhead, setPlayhead] = useState<number>();

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
  const dur_ms = Math.max((tl.data?.t_ms.at(-1) ?? 0) + 100, ...(evs.data ?? []).map((e) => e.end_ms), 1000);

  return (
    <Shell title={<>Review: {s?.candidate_label ?? sid} <span className="muted text-sm font-normal">{s?.status === "live" ? "(exam in progress)" : ""}</span></>}>
      {s && <Summary s={s} sid={sid} />}
      <div className="card space-y-1 p-3">
        <div className="flex items-center justify-between text-xs">
          <span className="muted">Drag across the timeline to zoom, double-click to reset. Click a mark to open it.</span>
          {zoom && <button className="btn-ghost py-0.5 text-xs" onClick={() => setZoom(undefined)}>Reset zoom</button>}
        </div>
        <TimelineView events={evs.data ?? []} segs={segs.data ?? []} tl={tl.data} dur={dur_ms} sel={sel} onSelect={setSel}
          zoom={zoom} setZoom={setZoom} playhead={playhead} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <aside className="card space-y-3 self-start p-3 lg:sticky lg:top-3 lg:max-h-[calc(100vh-1.5rem)] lg:overflow-y-auto">
          <div className="grid grid-cols-2 gap-2 text-sm">
            <select className="input" value={type} onChange={(e) => setType(e.target.value)} aria-label="filter by type">
              <option value="">All types</option>
              {REVIEW_TYPES.map((t) => <option key={t} value={t}>{LABEL[t]}</option>)}
            </select>
            <select className="input" value={prio} onChange={(e) => setPrio(e.target.value)} aria-label="filter by priority">
              <option value="">All priorities</option>
              {["high", "medium", "low"].map((p) => <option key={p}>{p}</option>)}
            </select>
            <label className="col-span-2 flex items-center gap-2"><input type="checkbox" checked={open} onChange={(e) => setOpen(e.target.checked)} /> Unreviewed only</label>
          </div>
          <SegmentList segs={shown} byId={byId} sel={sel} onSelect={setSel} />
        </aside>
        <section className="min-w-0">
          {cur ? (
            <Evidence key={cur.id} sid={sid} e={cur} decide={decide} move={move} setPlayhead={setPlayhead} playhead={playhead} />
          ) : (
            <p className="card muted">{evs.data?.length === 0 ? "Nothing was flagged in this session." : "Select an event."}</p>
          )}
        </section>
      </div>
    </Shell>
  );
}

function Summary({ s, sid }: { s: Session; sid: string }) {
  const c = s.calibration;
  const pol = Object.entries(s.policy).filter(([k, v]) => k.startsWith("allow_") && v).map(([k]) => k.replace("allow_", "").replace(/_/g, " "));
  return (
    <div className="card grid gap-x-6 gap-y-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
      <Stat k="Duration" v={dur(s.duration_s)} />
      <Stat k="Flagged for review" v={`${dur(s.flagged_s)}${s.duration_s ? ` (${Math.round((100 * s.flagged_s) / s.duration_s)} %)` : ""}`} />
      <Stat k="Blind spots" v={s.blind_spot_s ? dur(s.blind_spot_s) : "none"} />
      <Stat k="Reviewed" v={`${s.n_reviewed} of ${s.n_events} events`} />
      <Stat k="Calibration" v={c.mode ? `${c.mode}${c.error != null ? ` (error ${c.error.toFixed(2)})` : ""}` : "none"} />
      <Stat k="Policy" v={`${s.policy_id}${pol.length ? `: allows ${pol.join(", ")}` : ""}`} />
      <Stat k="Events" v={Object.entries(s.counts).map(([t, n]) => `${n} ${LABEL[t] ?? t}`).join(", ") || "none"} />
      <div className="flex items-start justify-end">
        <a className={`btn-primary ${s.status === "ended" ? "" : "pointer-events-none opacity-50"}`} href={`/api/sessions/${sid}/report`} target="_blank" rel="noreferrer"
          aria-disabled={s.status !== "ended"}>Export report</a>
      </div>
      <p className="muted text-xs sm:col-span-2 lg:col-span-4">
        Flags are observations for review, not findings. A webcam cannot see hands, laps or second screens; gaze is coarse;
        accuracy varies with lighting, glasses, skin tone, head coverings and camera quality; thresholds are untuned starting values.
      </p>
    </div>
  );
}

const Stat = ({ k, v }: { k: string; v: string }) => (
  <div><div className="muted text-xs uppercase tracking-wide">{k}</div><div>{v}</div></div>
);

function TimelineView(p: {
  events: Ev[]; segs: Seg[]; tl?: Timeline; dur: number; sel?: number; onSelect: (id: number) => void;
  zoom?: [number, number]; setZoom: (z?: [number, number]) => void; playhead?: number;
}) {
  const [a, b] = p.zoom ?? [0, p.dur];
  const L = 200, W = 1000, lh = 16, sigH = 34;
  const H = lh * (REVIEW_TYPES.length + 1) + sigH + 22;
  const X = (t: number) => L + ((W - L - 10) * (t - a)) / Math.max(b - a, 1);
  const svg = useRef<SVGSVGElement>(null);
  const down = useRef<number>();
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
  const segFill = { high: "#b91c1c", medium: "#f59e0b", low: "#a8a29e" };
  return (
    <svg ref={svg} viewBox={`0 0 ${W} ${H}`} className="w-full select-none text-[10px]" role="img" aria-label="session timeline"
      onPointerDown={(e) => (down.current = e.clientX)}
      onPointerUp={(e) => {
        if (down.current !== undefined && Math.abs(e.clientX - down.current) > 6) {
          const [t0, t1] = [tAt(down.current), tAt(e.clientX)].sort((x, y) => x - y);
          p.setZoom([Math.max(0, t0), Math.min(p.dur, t1)]);
        }
        down.current = undefined;
      }}
      onDoubleClick={() => p.setZoom(undefined)}>
      <defs><clipPath id="plot"><rect x={L} y={0} width={W - L} height={H} /></clipPath></defs>
      <text x={L - 6} y={lh - 4} textAnchor="end" className="fill-current opacity-70">review segments</text>
      {REVIEW_TYPES.map((t, i) => (
        <text key={t} x={L - 6} y={(i + 2) * lh - 4} textAnchor="end" className="fill-current opacity-70">{t === "MONITORING_DEGRADED" ? "Blind spots" : LABEL[t]}</text>
      ))}
      <text x={L - 6} y={lh * (REVIEW_TYPES.length + 1) + sigH / 2 + 4} textAnchor="end" className="fill-current opacity-70">
        <tspan fill="#0369a1">off-screen</tspan> / <tspan fill="#059669">quality</tspan>
      </text>
      <g clipPath="url(#plot)">
        {REVIEW_TYPES.map((_, i) => <line key={i} x1={L} x2={W} y1={(i + 2) * lh} y2={(i + 2) * lh} stroke="currentColor" strokeOpacity={0.08} />)}
        {p.segs.filter((g) => g.lane === "review").map((g) => ( /* blind spots have their own lane below */
          <rect key={g.id} x={X(g.start_ms)} y={2} width={Math.max(2, X(g.end_ms) - X(g.start_ms))} height={lh - 4} rx={2}
            fill={segFill[g.priority_label]} className="cursor-pointer" onClick={() => p.onSelect(g.events[0])}>
            <title>{`${g.priority_label} review priority: ${ts(g.start_ms)} - ${ts(g.end_ms)}`}</title>
          </rect>
        ))}
        {p.events.map((e) => {
          const row = REVIEW_TYPES.indexOf(e.type as (typeof REVIEW_TYPES)[number]) + 1;
          return (
            <rect key={e.id} x={X(e.start_ms)} y={row * lh + 3} width={Math.max(3, X(e.end_ms) - X(e.start_ms))} height={lh - 6} rx={2}
              fill={e.type === "MONITORING_DEGRADED" ? "#64748b" : "#0369a1"} stroke={e.id === p.sel ? "currentColor" : "none"} strokeWidth={2}
              className="cursor-pointer" onClick={() => p.onSelect(e.id)}>
              <title>{`${LABEL[e.type]}: ${ts(e.start_ms)} - ${ts(e.end_ms)}`}</title>
            </rect>
          );
        })}
        <polyline points={trace("off_screen_score")} fill="none" stroke="#0369a1" strokeWidth={1} />
        <polyline points={trace("quality")} fill="none" stroke="#059669" strokeWidth={1} />
        {p.playhead !== undefined && <line x1={X(p.playhead)} x2={X(p.playhead)} y1={0} y2={H - 18} stroke="#dc2626" strokeWidth={1.5} />}
      </g>
      {ticks.map((t) => <text key={t} x={X(t)} y={H - 4} textAnchor="middle" className="fill-current opacity-60">{ts(t)}</text>)}
    </svg>
  );
}

function SegmentList({ segs, byId, sel, onSelect }: { segs: Seg[]; byId: Map<number, Ev>; sel?: number; onSelect: (id: number) => void }) {
  if (!segs.length) return <p className="muted text-sm">No segments match.</p>;
  return (
    <ol className="space-y-2">
      {segs.map((g) => (
        <li key={g.id} className="rounded-md border border-stone-200 p-2 dark:border-stone-700">
          <div className="flex items-center gap-2 text-sm">
            <span className={`rounded px-1.5 text-xs ${PRIORITY_CLASS[g.lane === "blind_spot" ? "blind" : g.priority_label]}`}>
              {g.lane === "blind_spot" ? "blind spot" : g.priority_label}
            </span>
            <span className="font-mono text-xs">{ts(g.start_ms)} - {ts(g.end_ms)}</span>
            <span className="muted ml-auto text-xs" title="review priority (orders review; not a judgement)">{g.review_priority.toFixed(2)}</span>
          </div>
          <ul className="mt-1 space-y-0.5">
            {g.events.map((id) => {
              const e = byId.get(id);
              if (!e) return null;
              return (
                <li key={id}>
                  <button onClick={() => onSelect(id)} aria-current={id === sel}
                    className={`flex w-full items-center gap-2 rounded px-1.5 py-0.5 text-left text-sm hover:bg-stone-100 dark:hover:bg-stone-800 ${id === sel ? "bg-sky-100 dark:bg-sky-950" : ""}`}>
                    <span className="flex-1">{LABEL[e.type] ?? e.type}</span>
                    {e.review && <span className={`rounded px-1 text-[10px] ${DECISION_CLASS[e.review.decision]}`}>{e.review.decision.replace(/_/g, " ")}</span>}
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

function Evidence(p: { sid: string; e: Ev; decide: (d: string, note: string) => void; move: (d: number) => void; playhead?: number; setPlayhead: (t?: number) => void }) {
  const { e } = p;
  const plot = useApi<{ svg: string | null; caption: string | null }>(`/events/${e.id}/plot`);
  const [note, setNote] = useState(e.review?.note ?? "");
  const [boxes, setBoxes] = useState(false);
  const thumb = (k: "onset" | "peak" | "end") => (boxes && e.thumbs_overlay?.[k]) || e.thumbs[k];
  const [a, b] = clipWindow(e);
  const file = (path: string) => `/api/sessions/${p.sid}/files/${path}`;
  const d = e.details;
  useEffect(() => {
    const onKey = (k: KeyboardEvent) => {
      if (k.ctrlKey || k.metaKey || k.altKey || (k.target as HTMLElement).closest("input, textarea, select")) return;
      const key = k.key.toLowerCase();
      if (key === "j") p.move(1);
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
        <h2 className="text-lg font-semibold">{LABEL[e.type] ?? e.type}</h2>
        <span className="font-mono text-sm">{ts(e.start_ms)} - {ts(e.end_ms)}</span>
        <span className="muted text-xs">{e.type} · {e.detector}</span>
        <span className="muted ml-auto text-xs">J / K: next / previous</span>
      </header>
      <p className="leading-relaxed">{e.explanation}</p>
      <div className="grid gap-3 xl:grid-cols-2">
        <div>
          {e.clip_path ? (
            <video key={e.clip_path} controls preload="metadata" src={file(e.clip_path)} className="w-full rounded bg-black"
              onTimeUpdate={(x) => p.setPlayhead(a + x.currentTarget.currentTime * 1000)} />
          ) : (
            <p className="muted flex aspect-video items-center justify-center rounded bg-stone-100 px-4 text-center text-sm dark:bg-stone-800">
              {e.status === "final" ? "No clip for this event (encoding still running, ffmpeg missing, or no frames were buffered)." : "Clip pending."}
            </p>
          )}
          {Object.keys(e.thumbs_overlay ?? {}).length > 0 && (
            <label className="mt-2 flex items-center gap-2 text-sm">
              <input type="checkbox" checked={boxes} onChange={(x) => setBoxes(x.target.checked)} />
              Show detector boxes on keyframes
              {boxes && <span className="muted text-xs">(what the detectors saw at that moment; they can be wrong)</span>}
            </label>
          )}
          <div className="mt-2 grid grid-cols-3 gap-2">
            {(["onset", "peak", "end"] as const).map((k) => thumb(k) ? (
              <a key={k} href={file(thumb(k)!)} target="_blank" rel="noreferrer" className="block">
                <img src={file(thumb(k)!)} alt={`${k} keyframe${boxes ? " with detector boxes" : ""}`} className="w-full rounded border border-stone-200 dark:border-stone-700" loading="lazy" />
                <span className="muted text-xs">{k}</span>
              </a>
            ) : null)}
          </div>
        </div>
        <div className="space-y-3">
          {plot.data?.svg && (
            <figure>
              <div className="relative [&_svg]:h-auto [&_svg]:w-full">
                <div dangerouslySetInnerHTML={{ __html: plot.data.svg }} className="rounded border border-stone-200 bg-white dark:border-stone-700" />
                {p.playhead !== undefined && p.playhead >= a && p.playhead <= b && (
                  <div className="absolute inset-y-0 w-px bg-red-600" style={{ left: `${((p.playhead - a) / (b - a)) * 100}%` }} aria-hidden />
                )}
              </div>
              <figcaption className="muted mt-1 text-xs">{plot.data.caption}</figcaption>
            </figure>
          )}
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-sm">
            <dt className="muted">Detector confidence</dt><dd>{e.confidence.toFixed(2)} <span className="muted text-xs">(that the observation is real)</span></dd>
            <dt className="muted">Image quality</dt><dd>{typeof d.quality_mean === "number" ? d.quality_mean.toFixed(2) : "n/a"}</dd>
            <dt className="muted">Calibration</dt><dd>{String(d.calib_mode ?? "n/a")}{typeof d.calib_error === "number" ? ` (error ${d.calib_error.toFixed(2)})` : ""}</dd>
            <dt className="muted">Possible benign context</dt><dd>{flags.length ? flags.join(", ").replace(/_/g, " ") : "none noted"}</dd>
          </dl>
          {e.attribution && <Attribution a={e.attribution} />}
          <details className="text-sm">
            <summary className="cursor-pointer">Measured values</summary>
            <table className="mt-1 text-xs">
              <tbody>{Object.entries(d).map(([k, v]) => <tr key={k}><th className="muted pr-3 text-left align-top font-normal">{k}</th><td className="break-all">{typeof v === "string" ? v : JSON.stringify(v)}</td></tr>)}</tbody>
            </table>
          </details>
        </div>
      </div>
      <footer className="space-y-2 border-t border-stone-200 pt-3 dark:border-stone-800">
        <p className="text-sm">
          {e.review ? <>Last decision: <b>{e.review.decision.replace(/_/g, " ")}</b> by {e.review.reviewer}, {new Date(e.review.created_at).toLocaleString()}</> : <span className="muted">Not reviewed yet.</span>}
        </p>
        <textarea className="input h-16" placeholder="Note (optional): what you saw in the evidence" value={note} onChange={(x) => setNote(x.target.value)} maxLength={2000} aria-label="review note" />
        <div className="flex flex-wrap gap-2">
          {DECISIONS.map((x) => (
            <button key={x.value} className={x.value === "confirm" ? "btn-primary" : "btn-ghost"} onClick={() => p.decide(x.value, note)}>
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
