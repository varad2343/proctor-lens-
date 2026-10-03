import { useEffect, useRef, useState } from "react";
import { CandidateStream, type Msg } from "./lib";
import { EXAM_MINUTES, QUESTIONS } from "./questions";

// Candidate flow (master spec 11.1): consent -> camera check -> gaze calibration -> identity reference -> exam ->
// finish. Candidates never see proctor alerts; a "monitoring active" indicator is always on during the exam.
type Step = "consent" | "check" | "calib" | "enroll" | "exam" | "done" | "rejected";
type Dot = { dot_id: number; x: number; y: number; phase: string; dur_ms: number };
type CalibResult = { mode: string; accepted: boolean; error: number | null; tries: number; max_tries: number };
const SETUP_HINTS = ["dark", "bright", "blur", "blocked", "small_face", "face_cut"]; // shown during the exam too

export function Candidate({ sid, token }: { sid: string; token: string }) {
  const [step, setStep] = useState<Step>("consent");
  const [camError, setCamError] = useState<string>();
  const [status, setStatus] = useState<Msg>();
  const [dots, setDots] = useState<Dot[]>();
  const [result, setResult] = useState<CalibResult>();
  const [starting, setStarting] = useState(false);
  const [enrollMs, setEnrollMs] = useState(5000);
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<CandidateStream>();

  // only state setters in here: CandidateStream keeps this function for its whole life
  const onMsg = (m: Msg) => {
    if (m.type === "status") setStatus(m);
    else if (m.type === "hello" && m.phase === "exam") setStep("exam"); // page reloaded mid-exam
    else if (m.type === "calib_schedule") setDots(m.dots);
    else if (m.type === "calib_result") {
      setDots(undefined);
      setResult(m as unknown as CalibResult);
    } else if (m.type === "exam_started") {
      setEnrollMs(m.enroll_ms);
      setStarting(false);
      setStep("enroll");
    } else if (m.type === "ended") setStep("done");
    else if (m.type === "rejected") setStep("rejected");
  };

  async function startCamera() {
    setStep("check");
    if (!navigator.mediaDevices) {
      setCamError("The camera is only available on a secure page (https) or on this computer (localhost).");
      return;
    }
    try {
      const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
      video.current!.srcObject = media;
      await video.current!.play();
      stream.current = new CandidateStream(sid, token, video.current!, onMsg);
    } catch (e) {
      const name = (e as DOMException).name;
      setCamError(
        name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access for this page in the browser's site settings, then reload."
          : name === "NotFoundError"
            ? "No camera was found. Connect one and reload this page."
            : `The camera could not be started (${name}). Close other apps that use it, then reload.`,
      );
    }
  }

  const stopCamera = () => {
    stream.current?.close();
    (video.current?.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
  };
  useEffect(() => stopCamera, []);
  useEffect(() => {
    if (step === "done" || step === "rejected") stopCamera();
  }, [step]);

  const send = (m: Msg) => stream.current?.send(m);
  const calibrate = (mode: "full" | "baseline") => {
    document.documentElement.requestFullscreen?.().catch(() => undefined);
    setResult(undefined);
    setStep("calib");
    send({ type: "calib_start", mode });
  };
  const startExam = () => {
    setStarting(true);
    send({ type: "exam_start" });
  };

  const videoClass = {
    consent: "hidden",
    check: "w-full max-w-xl rounded-lg bg-black",
    calib: "fixed bottom-0 right-0 w-24 opacity-0 pointer-events-none", // must keep playing to be captured
    enroll: "w-full max-w-md rounded-lg bg-black",
    exam: "fixed bottom-4 right-4 z-30 w-44 rounded-lg border-2 border-white shadow-lg bg-black",
    done: "hidden",
    rejected: "hidden",
  }[step];

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      {step === "consent" && <Consent onAgree={startCamera} />}
      {step === "check" && (
        <section className="space-y-4">
          <h1 className="text-2xl font-semibold">Camera check</h1>
          {camError ? <p className="card border-red-300 text-red-800 dark:text-red-300">{camError}</p> : <Quality status={status} />}
        </section>
      )}
      {step === "enroll" && <Enroll ms={enrollMs} onDone={() => setStep("exam")} />}
      {/* one video element for the whole flow: frames are captured from it in every step */}
      <div className={step === "check" || step === "enroll" ? "my-4 flex justify-center" : ""}>
        <video ref={video} muted playsInline className={`-scale-x-100 ${videoClass}`} aria-label="Your camera" />
      </div>
      {step === "check" && !camError && (
        <div className="flex flex-wrap items-center gap-3">
          <button className="btn-primary" disabled={!status} onClick={() => calibrate("full")}>
            Continue to calibration
          </button>
          <button className="btn-ghost" disabled={!status} onClick={() => calibrate("baseline")}>
            Skip gaze calibration (use head direction only)
          </button>
        </div>
      )}
      {step === "calib" && dots && (
        <Dots
          dots={dots}
          onPoint={(d) => send({ type: "calib_point", dot_id: d.dot_id, t: stream.current!.now() })}
          onEnd={() => send({ type: "calib_end" })}
        />
      )}
      {step === "calib" && !dots && !result && <p className="muted">Preparing calibration...</p>}
      {step === "calib" && result && (
        <CalibrationResult result={result} starting={starting} onRetry={() => calibrate("full")} onContinue={startExam} />
      )}
      {step === "exam" && <Exam send={send} now={() => stream.current?.now() ?? Date.now()} status={status} />}
      {step === "done" && (
        <section className="card space-y-2 text-center">
          <h1 className="text-2xl font-semibold">Submitted</h1>
          <p>Your answers have been submitted. Thank you. You can close this window.</p>
        </section>
      )}
      {step === "rejected" && (
        <p className="card">This exam link is not valid any more: the session has ended or the link is incomplete. Ask your proctor for a new link.</p>
      )}
    </div>
  );
}

function Consent({ onAgree }: { onAgree: () => void }) {
  const [ok, setOk] = useState(false);
  return (
    <section className="card space-y-3 leading-relaxed">
      <h1 className="text-2xl font-semibold">Before you start</h1>
      <p>This mock exam uses camera and browser monitoring. Please read what that means.</p>
      <ul className="list-disc space-y-1.5 pl-5">
        <li><b>What is captured:</b> your webcam picture, analysed on the exam server while you take the exam, and activity on this page (switching tabs or windows, leaving fullscreen, copy and paste). No audio and no screen recording.</li>
        <li><b>Why:</b> to flag observable moments, such as no one in view, a phone visible, or attention away from the screen, for a person to review.</li>
        <li><b>What is kept:</b> short clips and still images around flagged moments, and measurements such as head direction and image quality. Not a recording of the whole exam. The face reference used to check identity stays in memory and is discarded at the end.</li>
        <li><b>Who sees it:</b> the exam proctor. It stays on the exam server and is deleted after 30 days by default, or earlier if you ask your proctor.</li>
        <li><b>No automatic decisions:</b> the system never decides anything about you. Every flag is reviewed by a person, in context.</li>
        <li><b>Accessibility:</b> if a condition affects your eye or head movement, tell your proctor; settings can be adjusted. You can skip the gaze calibration.</li>
      </ul>
      <label className="flex items-start gap-2 pt-2">
        <input type="checkbox" className="mt-1 size-4" checked={ok} onChange={(e) => setOk(e.target.checked)} />
        <span>I have read this and agree to camera and browser monitoring during this exam.</span>
      </label>
      <button className="btn-primary" disabled={!ok} onClick={onAgree}>
        Continue
      </button>
    </section>
  );
}

function Quality({ status }: { status?: Msg }) {
  if (!status) return <p className="muted">Starting the camera and connecting...</p>;
  const q = typeof status.quality === "number" ? status.quality : 0;
  const ok = q >= 0.5 && !status.reasons.includes("no_face");
  return (
    <div className="card space-y-2">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium">Picture quality</span>
        <div className="h-2 flex-1 rounded bg-stone-200 dark:bg-stone-700" role="meter" aria-valuenow={q} aria-valuemin={0} aria-valuemax={1}>
          <div className={`h-2 rounded ${ok ? "bg-emerald-600" : "bg-amber-500"}`} style={{ width: `${Math.round(q * 100)}%` }} />
        </div>
      </div>
      {ok ? (
        <p className="text-emerald-700 dark:text-emerald-400">Looks good. Sit as you will during the exam.</p>
      ) : (
        <ul className="list-disc pl-5 text-amber-800 dark:text-amber-300">
          {Object.values(status.guidance as Record<string, string>).map((g) => <li key={g}>{g}</li>)}
        </ul>
      )}
    </div>
  );
}

function Dots({ dots, onPoint, onEnd }: { dots: Dot[]; onPoint: (d: Dot) => void; onEnd: () => void }) {
  const [i, setI] = useState(-1); // -1 = instructions
  useEffect(() => {
    if (i < 0) return;
    if (i >= dots.length) {
      onEnd();
      return;
    }
    onPoint(dots[i]);
    const id = setTimeout(() => setI(i + 1), dots[i].dur_ms);
    return () => clearTimeout(id);
  }, [i]);
  const d = dots[Math.min(Math.max(i, 0), dots.length - 1)];
  return (
    <div className="fixed inset-0 z-40 bg-stone-950 text-stone-300">
      {i < 0 ? (
        <div className="flex h-full flex-col items-center justify-center gap-4 px-6 text-center">
          <h1 className="text-2xl font-semibold text-white">Calibration</h1>
          <p className="max-w-md">A dot will appear in {dots.length} places. Look at each dot until it moves. Move your eyes, and your head only as you would naturally. About {Math.round(dots.reduce((s, x) => s + x.dur_ms, 0) / 1000)} seconds.</p>
          <button className="btn-primary" autoFocus onClick={() => setI(0)}>Start</button>
        </div>
      ) : (
        <div
          key={i}
          className="absolute size-10 -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${d.x * 100}%`, top: `${d.y * 100}%` }}
          aria-label={`calibration dot ${Math.min(i + 1, dots.length)} of ${dots.length}`}
        >
          <div className="absolute inset-0 rounded-full border-4 border-sky-400" style={{ animation: `shrink ${d.dur_ms}ms linear forwards` }} />
          <div className="absolute inset-[38%] rounded-full bg-white" />
        </div>
      )}
    </div>
  );
}

function CalibrationResult(p: { result: CalibResult; starting: boolean; onRetry: () => void; onContinue: () => void }) {
  const r = p.result;
  const canRetry = !r.accepted && r.tries < r.max_tries;
  return (
    <section className="card space-y-3">
      <h1 className="text-2xl font-semibold">{r.accepted ? "Calibration complete" : "Calibration finished"}</h1>
      {r.accepted ? (
        <p>Gaze calibration is accurate (error {r.error?.toFixed(2)} of the screen width).</p>
      ) : r.mode === "none" ? (
        <p>Your face was not visible during calibration. Check the camera, then try again or continue.</p>
      ) : (
        <p>
          {r.error == null ? "A head-direction reference was captured." : `Gaze calibration was not accurate enough (error ${r.error.toFixed(2)}).`}{" "}
          Monitoring will use your head direction only. That is fine.
        </p>
      )}
      <p className="muted">Next: look at the camera for a few seconds so the system has a reference picture, then the exam starts.</p>
      <div className="flex gap-3">
        <button className="btn-primary" disabled={p.starting} onClick={p.onContinue}>
          {p.starting ? "Starting..." : "Continue"}
        </button>
        {canRetry && <button className="btn-ghost" disabled={p.starting} onClick={p.onRetry}>Try calibration again</button>}
      </div>
    </section>
  );
}

function Enroll({ ms, onDone }: { ms: number; onDone: () => void }) {
  useEffect(() => {
    const id = setTimeout(onDone, ms);
    return () => clearTimeout(id);
  }, []);
  return (
    <section className="space-y-3 text-center">
      <h1 className="text-2xl font-semibold">Reference picture</h1>
      <p>Look at the camera for a moment.</p>
      <div className="mx-auto h-2 max-w-md overflow-hidden rounded bg-stone-200 dark:bg-stone-700">
        <div className="h-2 bg-sky-600" style={{ animation: `grow ${ms}ms linear forwards` }} />
      </div>
    </section>
  );
}

function Exam({ send, now, status }: { send: (m: Msg) => void; now: () => number; status?: Msg }) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [left, setLeft] = useState(EXAM_MINUTES * 60);
  const [full, setFull] = useState(!!document.fullscreenElement);
  const [confirming, setConfirming] = useState(false);
  const submit = () => send({ type: "exam_end", answers });
  const submitRef = useRef(submit);
  submitRef.current = submit;

  useEffect(() => {
    // browser telemetry (master spec 5.6): recorded as events for review, never blocked
    const ev = (kind: string, detail = "") => send({ type: "browser_event", kind, t: now(), detail });
    const on: [EventTarget, string, () => void][] = [
      [document, "visibilitychange", () => ev(document.hidden ? "hidden" : "visible")],
      [window, "blur", () => ev("blur")],
      [window, "focus", () => ev("focus")],
      [document, "fullscreenchange", () => {
        setFull(!!document.fullscreenElement);
        ev(document.fullscreenElement ? "fullscreen_enter" : "fullscreen_exit");
      }],
      [document, "paste", () => ev("paste")], // content is never sent
      [document, "copy", () => ev("copy")],
      [document, "contextmenu", () => ev("contextmenu")],
      [window, "offline", () => ev("offline")],
      [window, "online", () => ev("online")],
      [window, "beforeunload", () => ev("beforeunload")],
    ];
    on.forEach(([t, k, f]) => t.addEventListener(k, f));
    const tick = setInterval(() => setLeft((s) => (s <= 1 ? (submitRef.current(), 0) : s - 1)), 1000);
    return () => {
      on.forEach(([t, k, f]) => t.removeEventListener(k, f));
      clearInterval(tick);
    };
  }, []);

  const guide = (status?.guidance ?? {}) as Record<string, string>;
  const hints = SETUP_HINTS.filter((r) => guide[r]).map((r) => guide[r]);
  return (
    <section className="space-y-4 pb-48">
      <div className="sticky top-0 z-20 -mx-4 flex flex-wrap items-center gap-3 border-b border-stone-200 bg-stone-50/95 px-4 py-2 text-sm backdrop-blur dark:border-stone-800 dark:bg-stone-950/95">
        <span className="flex items-center gap-2 font-medium text-red-700 dark:text-red-400">
          <span className="size-2.5 animate-pulse rounded-full bg-red-600" aria-hidden /> Monitoring active
        </span>
        <span className="muted">Camera and browser activity are recorded for review.</span>
        <span className="ml-auto font-mono tabular-nums" aria-label="time left">
          {String(Math.floor(left / 60)).padStart(2, "0")}:{String(left % 60).padStart(2, "0")}
        </span>
        {!full && (
          <button className="btn-ghost py-1" onClick={() => document.documentElement.requestFullscreen?.().catch(() => undefined)}>
            Fullscreen
          </button>
        )}
      </div>
      {hints.length > 0 && <p className="card border-amber-300 text-amber-900 dark:text-amber-200">Camera view: {hints.join("; ")}</p>}
      <h1 className="text-2xl font-semibold">Mock exam</h1>
      {QUESTIONS.map((q, n) => (
        <fieldset key={q.id} className="card space-y-2">
          <legend className="px-1 font-medium">{n + 1}. {q.text}</legend>
          {q.options.map((o) => (
            <label key={o} className="flex items-center gap-2">
              <input type="radio" name={q.id} checked={answers[q.id] === o} onChange={() => setAnswers((a) => ({ ...a, [q.id]: o }))} />
              {o}
            </label>
          ))}
        </fieldset>
      ))}
      <div className="flex items-center gap-3">
        {confirming ? (
          <>
            <button className="btn-primary" onClick={submit}>Confirm: submit answers</button>
            <button className="btn-ghost" onClick={() => setConfirming(false)}>Keep working</button>
          </>
        ) : (
          <button className="btn-primary" onClick={() => setConfirming(true)}>Submit</button>
        )}
        <span className="muted text-sm">{Object.keys(answers).length} of {QUESTIONS.length} answered</span>
      </div>
    </section>
  );
}
