/// <reference lib="webworker" />
// Browser-mode candidate session (docs/DECISIONS.md ADR-019): ProctorLens's own server-side session code
// (proctorlens/browser.py BrowserSession = server/live.py LiveSession) runs here in Pyodide; the camera models run in
// JavaScript (MediaPipe Tasks: face landmarker; ONNX Runtime Web: YOLO11n, SCRFD, ArcFace). The page talks to this
// worker with the /ws/stream messages; results go to the API (/api/c/<sid>/...) from here.
// A classic worker (Vite's default): MediaPipe loads its WASM with importScripts, which module workers lack.
// Libraries come from jsDelivr, pinned.

const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
const MEDIAPIPE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0";
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const MUXER = "https://cdn.jsdelivr.net/npm/webm-muxer@5.1.4/build/webm-muxer.mjs";
const FLUSH_MS = 1000; // status + live picture every second (one request); evidence files as they are ready

const ctx = self as unknown as DedicatedWorkerGlobalScope & { loadPyodide: any; plJs: any };
const load = (url: string) => import(/* @vite-ignore */ url);
let py: any, sess: any, ort: any, videoLm: any, imageLm: any;
const ort_s: Record<string, any> = {};
let sid = "", token = "", lastLmT = -1, flushing: Promise<void> = Promise.resolve(), lastFlush = 0, ended = false;
let lastFrame = 0;
let detectors: Promise<void> = Promise.resolve(); // YOLO + face ID: needed from the exam on, loaded meanwhile

const post = (m: any) => ctx.postMessage(m);
const progress = (text: string) => post({ type: "progress", text });
const api = (path: string, init?: RequestInit) => fetch(`/api/c/${sid}${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`, init);

// ---------------------------------------------------------------- models (called from Python)

/** MediaPipe Face Landmarker; returns per face {lm: 478x3 float32, blend: {name: score}, m: 4x4 row-major float32}. */
function landmarks(rgba: Uint8Array, w: number, h: number, t: number, imageMode: boolean) {
  const img = new ImageData(new Uint8ClampedArray(rgba.buffer as ArrayBuffer, rgba.byteOffset, rgba.byteLength), w, h);
  let r;
  if (imageMode) r = imageLm.detect(img);
  else {
    lastLmT = Math.max(Math.round(t), lastLmT + 1); // VIDEO mode needs strictly increasing timestamps (as in Python)
    r = videoLm.detectForVideo(img, lastLmT);
  }
  return r.faceLandmarks.map((pts: any[], i: number) => {
    const lm = new Float32Array(pts.length * 3);
    pts.forEach((p, j) => lm.set([p.x, p.y, p.z], j * 3));
    const blend: Record<string, number> = {};
    for (const c of r.faceBlendshapes?.[i]?.categories ?? []) blend[c.categoryName] = c.score;
    const mx = r.facialTransformationMatrixes?.[i];
    let m: Float32Array | null = null;
    if (mx) { // MediaPipe's JS matrix data is column-major; the Python API (and head_pose.py) reads row-major
      m = new Float32Array(16);
      for (let row = 0; row < 4; row++) for (let col = 0; col < 4; col++) m[row * 4 + col] = mx.data[col * 4 + row];
    }
    return { lm, blend, m };
  });
}

async function runOnnx(name: string, data: Uint8Array, shape: number[]) {
  const s = ort_s[name];
  const x = new ort.Tensor("float32", new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4), shape);
  const out = await s.run({ [s.inputNames[0]]: x });
  return s.outputNames.map((n: string) => {
    const o = out[n];
    return { data: new Uint8Array(o.data.buffer, o.data.byteOffset, o.data.byteLength), shape: o.dims };
  });
}

// ---------------------------------------------------------------- start

async function init(): Promise<void> {
  progress("Preparing the exam on this device (the first time can take a minute)...");
  const bytes = (url: string) => fetch(url).then((r) => r.arrayBuffer());
  // first what the camera check needs (Python, the face model: ~55 MB on a first visit), side by side; the exam's
  // detectors (~35 MB) download after that, during the camera check and calibration
  const [info, yolo] = await Promise.all([api("").then((r) => r.json()), fetch("/models/yolo11n.json").then((r) => r.json())]);
  const python = (async () => {
    ctx.importScripts(PYODIDE + "pyodide.js");
    py = await ctx.loadPyodide({ indexURL: PYODIDE });
    await py.loadPackage(["numpy", "pandas", "opencv-python", "pyyaml", "sqlite3"]);
    py.unpackArchive(await bytes("/py/proctorlens.zip"), "zip", { extractDir: "/app" });
  })();
  const faces = (async () => {
    const [vision, model] = await Promise.all([load(`${MEDIAPIPE}/vision_bundle.mjs`), bytes("/models/face_landmarker.task")]);
    const fileset = await vision.FilesetResolver.forVisionTasks(`${MEDIAPIPE}/wasm`);
    const lmOpts = (runningMode: string, numFaces: number) => ({
      baseOptions: { modelAssetBuffer: new Uint8Array(model.slice(0)), delegate: "CPU" }, runningMode, numFaces,
      outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true,
    });
    videoLm = await vision.FaceLandmarker.createFromOptions(fileset, lmOpts("VIDEO", 3)); // = Landmarker defaults
    imageLm = await vision.FaceLandmarker.createFromOptions(fileset, lmOpts("IMAGE", 2)); // = the recovery landmarker
  })();
  const onnx = async () => {
    const names = { yolo: "yolo11n.onnx", scrfd: "det_500m.onnx", arcface: "w600k_mbf.onnx" };
    const [lib, ...bufs] = await Promise.all([load(`${ORT}ort.webgpu.min.mjs`), ...Object.values(names).map((f) => bytes(`/models/${f}`))]);
    ort = lib;
    ort.env.wasm.wasmPaths = ORT;
    const eps = "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"];
    for (const [i, k] of Object.keys(names).entries()) {
      ort_s[k] = await ort.InferenceSession.create(new Uint8Array(bufs[i]), { executionProviders: eps });
    }
    // one run each now: a model's first run (WebGPU: shader compiles) takes seconds and would stall the first exam frames
    const warm: Record<string, number[]> = { yolo: [1, 3, ...yolo.imgsz], scrfd: [1, 3, 320, 320], arcface: [1, 3, 112, 112] };
    for (const [k, shape] of Object.entries(warm)) await runOnnx(k, new Uint8Array(shape.reduce((a, b) => a * b) * 4), shape);
  };
  await Promise.all([python, faces]);
  detectors = onnx();
  detectors.catch((err) => post({ type: "fatal", detail: `This device could not load the exam's detectors: ${err?.message ?? err}` }));

  ctx.plJs = { landmarks };
  py.globals.set("SID", sid);
  py.globals.set("CFG_JSON", JSON.stringify(info.config));
  py.globals.set("YOLO_JSON", JSON.stringify(yolo));
  sess = py.runPython(`
import sys, json, js
sys.path.insert(0, "/app")
from proctorlens.browser import BrowserSession
from proctorlens.core.config import Config, from_dict
BrowserSession(SID, from_dict(Config, json.loads(CFG_JSON)), js.plJs, json.loads(YOLO_JSON))
`);
  progress("");
  if (info.status === "ended") post({ type: "msgs", msgs: [{ type: "ended" }] });
  // no frame for 0.5 s (camera stalled, tab throttled): advance the session clock, as the server does; queued with the
  // frames, so it never lands between a frame's detection and its processing
  setInterval(() => {
    if (Date.now() - lastFrame > 500) chain = chain.then(() => safe(() => sess.tick_in()));
  }, 500);
}

// ---------------------------------------------------------------- frames and control

const camera = new OffscreenCanvas(640, 480);
/** The page's frame, 640 wide, as a JPEG (the size and quality the server mode gets). */
async function toJpeg(bitmap: ImageBitmap): Promise<Uint8Array> {
  camera.height = Math.round((640 * bitmap.height) / bitmap.width) || 480;
  camera.getContext("2d")!.drawImage(bitmap, 0, 0, camera.width, camera.height);
  bitmap.close();
  return new Uint8Array(await (await camera.convertToBlob({ type: "image/jpeg", quality: 0.7 })).arrayBuffer());
}

async function frame(seq: number, t: number, jpeg: Uint8Array) {
  if (ended) return post({ type: "ack", seq });
  lastFrame = Date.now();
  sess.prepare(jpeg);
  const di = sess.detection_input(t);
  let msgs: string;
  if (di) {
    const [x, shape] = di.toJs();
    di.destroy();
    await detectors;
    const [out] = await runOnnx("yolo", x, shape);
    msgs = sess.frame_in(seq, t, jpeg, out.data, out.shape);
  } else msgs = sess.frame_in(seq, t, jpeg);
  post({ type: "msgs", msgs: JSON.parse(msgs) });
  // identity models (requests the perceiver made in this frame); the answer is ready for the next frame
  const reqs = sess.model_requests();
  for (const r of reqs.toJs({ dict_converter: Object.fromEntries })) {
    sess.model_result(r.model, await runOnnx(r.model, r.input, r.shape));
  }
  reqs.destroy();
  sess.run_slices(Math.max(5, 80 - (Date.now() - lastFrame))); // evidence work in what is left of 100 ms
  post({ type: "ack", seq });
  if (Date.now() - lastFlush > FLUSH_MS) flush();
}

async function control(msg: any) {
  if (msg.type === "exam_end") {
    const out = JSON.parse(sess.end_in(JSON.stringify(msg.answers ?? null)));
    ended = true;
    // "Submitted" as soon as the answers are saved; "uploading" first, so the page never says "close" too early
    await flush(true, () => (progress("Uploading the exam's evidence..."), post({ type: "msgs", msgs: out })));
    sess.run_slices(Infinity); // then the evidence still owed, uploaded while the page says so
    await flush(true);
    return progress("");
  }
  if (msg.type === "exam_start") await detectors; // usually loaded during calibration already
  post({ type: "msgs", msgs: JSON.parse(sess.control_in(JSON.stringify(msg))) });
  if (msg.type === "browser_event" || msg.type === "exam_start" || msg.type === "calib_end") flush();
}

// ---------------------------------------------------------------- results to the API

async function encodeWebm(frames: Uint8Array[], fps: number): Promise<Uint8Array | null> {
  const VE = (ctx as any).VideoEncoder;
  if (!VE || !frames.length) return null;
  const first = await createImageBitmap(new Blob([frames[0] as BlobPart], { type: "image/jpeg" }));
  const width = first.width & ~1, height = first.height & ~1;
  first.close();
  let codec = "";
  for (const c of ["vp8", "vp09.00.10.08"]) {
    if ((await VE.isConfigSupported({ codec: c, width, height, bitrate: 400_000, framerate: fps })).supported) { codec = c; break; }
  }
  if (!codec) return null;
  const { Muxer, ArrayBufferTarget } = await load(MUXER);
  const muxer = new Muxer({ target: new ArrayBufferTarget(), video: { codec: codec === "vp8" ? "V_VP8" : "V_VP9", width, height, frameRate: fps } });
  let failed: unknown = null;
  const enc = new VE({ output: (chunk: any, meta: any) => muxer.addVideoChunk(chunk, meta), error: (e: unknown) => (failed = e) });
  enc.configure({ codec, width, height, bitrate: 400_000, framerate: fps }); // ~1.5 MB for a 30 s clip
  const canvas = new OffscreenCanvas(width, height);
  const g = canvas.getContext("2d")!;
  for (let i = 0; i < frames.length; i++) {
    const bmp = await createImageBitmap(new Blob([frames[i] as BlobPart], { type: "image/jpeg" }));
    g.drawImage(bmp, 0, 0, width, height);
    bmp.close();
    const vf = new (ctx as any).VideoFrame(canvas, { timestamp: Math.round((i * 1e6) / fps), duration: Math.round(1e6 / fps) });
    enc.encode(vf, { keyFrame: i % 30 === 0 });
    vf.close();
  }
  await enc.flush();
  enc.close();
  if (failed) return null;
  muxer.finalize();
  return new Uint8Array(muxer.target.buffer);
}

async function put(path: string, body: Uint8Array, type: string) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const r = await fetch(`/api/c/${sid}${path}${path.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`,
      { method: "PUT", body: body as BodyInit, headers: { "Content-Type": type } }).catch(() => null);
    if (r?.ok) return;
    await new Promise((res) => setTimeout(res, 1000 * (attempt + 1)));
  }
}

/** Runs the tasks, n at a time. */
async function inParallel(tasks: (() => Promise<unknown>)[], n = 4) {
  const next = async () => {
    for (let t = tasks.shift(); t; t = tasks.shift()) await t();
  };
  await Promise.all(Array.from({ length: n }, next));
}

/** One flush at a time: operations, status and the live picture in one request first (an event exists before its
 * evidence), then the new evidence files and clips, uploaded 4 at a time. */
function flush(final = false, synced?: () => void): Promise<void> {
  lastFlush = Date.now();
  flushing = flushing.then(async () => {
    const status = JSON.parse(sess.live_status());
    const d = JSON.parse(sess.drain());
    const feed = d.feed.filter((m: any) => m.type === "event_started" || m.type === "event_ended");
    const snap = final ? null : sess.snapshot();
    let frame = {};
    if (snap) {
      const b64 = (b: Uint8Array) => new FileReaderSync().readAsDataURL(new Blob([b as BlobPart])).split(",")[1];
      const [ov, raw] = snap.toJs();
      snap.destroy();
      frame = { overlay: b64(ov), raw: b64(raw) };
    }
    if (d.ops.length || feed.length || !final) {
      for (let attempt = 0; attempt < 3; attempt++) {
        const r = await api("/sync", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ token, ops: d.ops, feed, status: final ? null : status, ...frame }) }).catch(() => null);
        if (r?.ok) {
          const { rtc } = await r.json(); // proctors asking for live video: the page (which has the camera) answers
          if (rtc?.length) post({ type: "rtc", offers: rtc });
          break;
        }
        await new Promise((res) => setTimeout(res, 1000 * (attempt + 1)));
      }
    }
    const tasks: (() => Promise<unknown>)[] = [
      ...d.files.map((f: string) => () => put(`/files/${f}`, sess.read_file(f), f.endsWith(".jpg") ? "image/jpeg" : "application/octet-stream")),
      ...d.clips.map((c: any) => async () => {
        const fr = sess.clip_frames(c.id);
        const webm = await encodeWebm(fr.toJs(), c.fps);
        fr.destroy();
        if (webm) await put(`/files/${c.path}`, webm, "video/webm");
        sess.clip_done(c.id);
      }),
    ];
    synced?.();
    const total = tasks.length;
    let done = 0;
    await inParallel(tasks.map((t) => () => t().then(() => final && progress(`Uploading the exam's evidence: ${++done} of ${total}.`))));
  });
  return flushing;
}

// ---------------------------------------------------------------- message loop (in order, each after init)

let ready: Promise<void> | null = null;
let chain: Promise<unknown> = Promise.resolve();
const safe = (fn: () => unknown) => {
  try {
    fn();
  } catch (e) {
    post({ type: "progress", text: "" });
    console.error(e);
  }
};

ctx.onmessage = (e: MessageEvent) => {
  const m = e.data;
  if (m.type === "init") {
    sid = m.sid;
    token = m.token;
    ready = init().catch((err) => post({ type: "fatal", detail: `This device could not start the exam software: ${err?.message ?? err}` }));
    return;
  }
  chain = chain.then(() => ready).then(async () => {
    if (!sess) return;
    try {
      if (m.type === "frame") await frame(m.seq, m.t, await toJpeg(m.bitmap));
      else if (m.type === "control") await control(m.msg);
    } catch (err) {
      console.error(err);
      if (m.type === "frame") post({ type: "ack", seq: m.seq });
    }
  });
};
