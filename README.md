# ProctorLens

A Python machine-learning / computer-vision pipeline that turns webcam video into **time-stamped, explainable, observable events**: attention directed away from the screen, a phone visible, a second person present, and so on. It is a coursework ML/CV project, run locally: from the command line, or as a local web app (candidate exam page + proctor review dashboard, see [Web app](#web-app)) that runs the same pipeline.

**Events are observations, not verdicts.** The system never produces a label or score about a person's intent or conduct. Each event says what was seen, when, how strongly the detector believes the observation is real (`confidence`), and what the measured signal was against its threshold. When the camera cannot see properly (dark, blurred, blocked, face unusable, frame stall) it emits `MONITORING_DEGRADED` instead of guessing.

```
video file | camera  ->  quality -> face landmarks / head pose / gaze / mouth ->  10 Hz feature rows -> features.parquet
                                     phone / notes / person detector (every ~3rd frame) ---^
                                     identity check (every ~10 s, optional) --------------^
                         feature rows -> score provider (rules | LightGBM | causal TCN) -> event state machines -> events.json
```

| Event | Observed when |
|---|---|
| `FACE_ABSENT` | no face and no person in view for >= 3 s (not counted while monitoring is degraded) |
| `MULTIPLE_PEOPLE` | >= 2 distinct live faces or persons visible for >= 2 s (static faces such as posters are ignored) |
| `PROHIBITED_OBJECT` | phone visible >= 1.5 s, or notes / book >= 3 s (policy can allow notes) |
| `OFF_SCREEN_SUSTAINED` | attention off the screen for >= 4 s, with a direction (left/right/up/down) |
| `REPEATED_GLANCING` | >= 4 short off-screen looks within 60 s, >= 70% toward one side |
| `MOUTH_ACTIVITY` | mouth movement consistent with speaking, >= 3 s within 10 s |
| `IDENTITY_MISMATCH` | face embedding differs from the enrollment for >= 3 consecutive good checks |
| `MONITORING_DEGRADED` | the system cannot reliably observe (dark, blur, blocked, face too small/cut off, frame gap) |

Defaults are in `src/proctorlens/core/config.py`; `configs/policy.yaml` holds the thresholds and the fairness toggles (`allow_notes`, `allow_looking_down`, `allow_reading_aloud`). The config hash is written into every `events.json`.

Design and scope: `docs/SPEC.md` (source of truth). Deviations and their reasons: `docs/DECISIONS.md`. Data collection and labeling: `docs/DATA_PROTOCOL.md`. Evaluation protocol and result tables: `docs/EVALUATION.md`. Model cards: `docs/model_cards/TEMPLATE.md`.

## Install

Python 3.11+ (developed on 3.12).

```
python -m pip install -e ".[perception,train,dev]"
```

Extras: `perception` = mediapipe, onnxruntime, ultralytics, insightface; `train` = torch, lightgbm; `dev` = pytest, hypothesis, ruff. The core (numpy, pandas, pyarrow, opencv-python, scipy, scikit-learn, pyyaml) installs without any extra. Heavy libraries are imported lazily, so every `proctorlens` module imports with only the core installed, and the pure logic and its tests run that way. Check each heavy library's docs for the Python versions it supports before choosing an interpreter.

`make` is optional. Each Makefile target is one command; on Windows PowerShell without make, run the command shown:

| `make` target | PowerShell equivalent |
|---|---|
| `setup` | `python -m pip install -e ".[perception,train,dev]"` |
| `test` | `python tests/run.py` |
| `replay VIDEO=x.mp4 OUT=out` | `proctorlens replay x.mp4 --config configs/pipeline.yaml --config configs/policy.yaml --out out` |
| `eval` | `python -m ml.evaluation.run_eval` |
| `train-detector` | `python -m ml.training.train_detector` |
| `train-temporal` | `python -m ml.training.train_temporal_gbm; python -m ml.training.train_temporal_tcn` |
| `web` | `python -m pip install -e ".[perception,web]"; cd frontend; npm install; npm run build; cd ..` |
| `serve` | `proctorlens serve --config configs/pipeline.yaml --config configs/policy.yaml` |

## Model files you download yourself

Nothing is bundled or downloaded automatically by this repo. Put the files where `configs/pipeline.yaml` (`models:`) looks for them; paths are relative to the directory you run from. `data/models/` is git-ignored. Check the license of each before use (SPEC Section 19).

| Model | Goes to (`models.` key) | How to get it | License note |
|---|---|---|---|
| MediaPipe Face Landmarker (`.task`) | `data/models/face_landmarker.task` (`landmarker`) | "Face Landmarker" model bundle from the MediaPipe Solutions page, e.g. `Invoke-WebRequest https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -OutFile data/models/face_landmarker.task` (verify the URL on the current docs) | see the MediaPipe model card for the bundle's terms |
| YOLO nano, ONNX | `data/models/yolo11n.onnx` (`detector`) | With ultralytics installed: `yolo export model=yolo11n.pt format=onnx imgsz=640` (it fetches `yolo11n.pt` on first use), then move `yolo11n.onnx` to `data/models/`. A `.pt` path also works and runs through ultralytics. Use ultralytics' own export: the ONNX loader reads the class names from the file's metadata. After fine-tuning, point `detector` at the fine-tuned export | AGPL-3.0 (Ultralytics); fine for coursework, keep that in mind before redistributing |
| InsightFace `buffalo_sc` | `data/models/buffalo_sc/` (`identity`), the folder holding the pack's `.onnx` files | insightface downloads the pack itself the first time if it is missing (`FaceAnalysis(name="buffalo_sc", root="data")` looks in `data/models/buffalo_sc`); place it there beforehand to work offline | pretrained models are non-commercial research only |

Identity is optional. If `data/models/buffalo_sc/` does not exist (or insightface is not installed), identity checks are skipped with a warning and no `IDENTITY_MISMATCH` event can be produced; everything else runs. Enrollment embeddings stay in memory and are never written to disk.

## Run it

All entry points use the same `Pipeline`. `--config` can be repeated; files are merged over the defaults in order, unknown keys are an error.

```
# 1. Calibrate gaze for one person (fullscreen dots); writes calib.json.
#    Each dot advances as soon as enough face samples were seen, so it takes ~15 s, not a fixed 25 s.
#    --calib-mode quick = 5 dots (~8 s, slightly coarser); baseline = centre only (~3 s, head-pose only).
#    Keys: Esc/Q quit, S skip (keeps what was captured), R retry on the result screen. The ring around the
#    dot is green while a face is seen and red when not, with a hint. Closing the window also quits.
proctorlens calibrate --camera 0 [--calib-mode quick]

# 2. Replay a recording: writes features.parquet and events.json into --out.
#    Writes no frames or video unless --render is given.
proctorlens replay session.mp4 --config configs/pipeline.yaml --config configs/policy.yaml \
    --calib calib.json --out out/session [--render out/session/overlay.mp4]

# 3. Live: OpenCV window with overlays and an event ticker; calibrates first unless --calib is given
#    (and saves it to calib.json, so next time pass --calib calib.json and skip the wait). q/Esc or closing
#    the window quits.
proctorlens live --camera 0 --config configs/pipeline.yaml --config configs/policy.yaml [--calib calib.json] [--calib-mode quick]

# 4. Batch over a dataset folder (uses <stem>.calib.json next to each video when present)
proctorlens extract-features data/recordings --config configs/pipeline.yaml --config configs/policy.yaml

# 5. Review report for a replay dir: segments.json + report.html (opens from disk, no server). With --video it also
#    writes evidence/: per event an H.264 clip (start - 5 s to end + 3 s; needs ffmpeg on PATH) and onset / peak / end
#    keyframes. Pass the --render video to get overlays in them. `replay ... --report` does this in one go.
proctorlens report out/session --video out/session/overlay.mp4 --config configs/pipeline.yaml --config configs/policy.yaml
```

The report groups events within 5 s of each other into review segments ranked by review priority (high / medium / low; `review:` in the config, `docs/DECISIONS.md` ADR-015). Priority only orders what to look at first; monitoring-degraded time is listed separately as blind spots. Each event gets a plain-language explanation built from its `details`, a signal plot with its threshold, and its measured values.

Without a calibration file, off-screen evidence comes from head pose only and the event `details` show `calib_mode: none`; `head_pose_only` means a calibration was run but missed the accuracy bar (`gaze.accept_error`), and `full` means the fitted gaze model is in use. Without `pyarrow`, the feature table is written as `features.csv` instead of `features.parquet`. To use a learned scorer, set `scorer.provider` to `gbm` or `tcn` and list the model directories under `scorer.models` (see `configs/pipeline.yaml`).

Each event in `events.json` carries `details` (signal, peak/mean, thresholds, `t_on_s`, duration, zone, calibration error and mode, mean quality, benign-context flags) and, for learned scorers, `attribution` (score drop when a feature group is neutralized). The file starts with a provenance header: schema versions, config hash, model file hashes, thresholds in force.

## Web app

A candidate takes a mock exam in the browser while the server runs the pipeline on their webcam frames; a proctor watches live and reviews the flagged segments afterwards (`docs/DECISIONS.md` ADR-016). Setup, once:

```
python -m pip install -e ".[perception,web]"
cd frontend; npm install; npm run build; cd ..
```

Run (PowerShell; without `PROCTORLENS_PASSWORD` the server prints a random one at start):

```
$env:PROCTORLENS_PASSWORD = "choose-one"
proctorlens serve --config configs/pipeline.yaml --config configs/policy.yaml
```

Open http://127.0.0.1:8000, log in as `proctor`, create a session and send the candidate its link. The candidate sees consent, a camera check, a 9-dot gaze calibration (skippable: head direction only), a short reference picture, then the exam with a "monitoring active" indicator; they never see flags. The proctor gets a sessions list, a live view (~1 frame/s, live observations) and the review workbench: timeline with lanes per event type and a blind-spot lane, segments ranked by review priority, clip + keyframes + signal plot + plain-language explanation per event, confirm / dismiss / needs-more-info with a note (keys J/K, C/D/N), and an exported HTML report. Data lives in `data/proctorlens.db` and `data/sessions/<id>/` (clips around events only, never a full recording); ended sessions are deleted after `PROCTORLENS_RETENTION_DAYS` (30) and can be deleted at any time. Clips need `ffmpeg` on PATH.

The server listens on this machine only. Browsers allow the camera only on localhost or HTTPS, so for a candidate on another computer put HTTPS in front (and `--host 0.0.0.0`). Frontend development: `npm run dev` in `frontend/` (proxies to the server on :8000).

End-to-end check with real models and no webcam: with the server running and `$env:PROCTORLENS_PASSWORD` set in this shell too, it drives Edge through the whole flow with a looping fake camera (face, covered camera, empty room, several people) and checks that events reach the dashboard:

```
python tools/fake_cam.py fake_cam.y4m
cd frontend; $env:FAKE_CAM = "$PWD\..\fake_cam.y4m"; node e2e.mjs
```

## Recording a dataset

`python tools/collect/record.py` captures the webcam to video (no audio), shows the calibration dots and the scripted cue prompts, and logs cue timestamps to CSV. Consent, the session script, labeling and splits are in `docs/DATA_PROTOCOL.md`; read it and get consent **before** recording anyone. Recordings, labels and models stay out of git; only `data/manifest.csv` is tracked.

## Training and evaluation

```
python -m ml.training.train_detector          # fine-tune YOLO nano on phone / notes / person
python -m ml.training.train_temporal_gbm      # LightGBM on window statistics (off_screen, speaking)
python -m ml.training.train_temporal_tcn      # small causal TCN (off_screen, speaking)
python -m ml.evaluation.run_eval              # event-level tables from stored features + labels
```

Trainers write artifacts to `data/models/temporal/<name>/<version>/` (`meta.json` plus `model.txt` or `model.pt`) and a `metrics.json`; fill a model card from `docs/model_cards/TEMPLATE.md` for each. Pass `--help` to any script for its arguments. Splits are participant-disjoint, the test split is used once, and results go into `docs/EVALUATION.md` with participant-level bootstrap CIs.

## Tests

Tests are plain `test_*` functions with bare asserts and no fixtures, so either runner works:

```
python tests/run.py            # no pytest needed
python tests/run.py clock      # only test files whose path contains "clock"
pytest                         # if installed (pip install -e ".[dev]")
```

Tests that need a library that is not installed return early instead of failing.

## Accuracy aids and tools

Each aid has a config switch (`False` = the old behaviour), so you can A/B them:
- `perception.recover_small_faces`: re-runs the landmarker on upscaled person-box crops when no face was found. Measured on the two sample photos: Zidane photo 0 -> 1 faces, bus photo 0 -> 2 faces (wide shots with small faces; a webcam face is large and unaffected).
- `perception.lowlight_enhance`: gamma/CLAHE on the landmarker input only when the frame is dim (the quality check still sees the original, so dark still reports as dark).
- `smooth.enabled`: One Euro filter on pose/gaze signals, with gaze held during blinks, to cut jitter-driven flicker.
- `gaze.robust`: calibration drops blink/outlier samples and fits on per-dot medians, with stability gating of calibration samples.
- `eye_open` now uses the eye-blink blendshape (it used to read exactly 1.0), and the yaw/pitch sign convention was checked against landmark geometry and flipped images (a flip negates yaw).

Commands: `proctorlens doctor [--selftest] [--camera N]` checks dependencies, model files and sample images; `proctorlens fetch-models [--dry-run]` downloads any missing model files (InsightFace weights are non-commercial research only; Ultralytics is AGPL-3.0); `replay`/`live --out` also write `summary.json`.

Only the small-face numbers above were re-measured by hand; the other aids are covered by unit tests (synthetic data and real-image checks), not by a dataset-level accuracy study. No independent review of this batch was run.

## Status: what has and has not been verified

Tested on Windows, Python 3.12, with mediapipe 1.0.1, ultralytics 8.4 (COCO `yolo11n.pt`), insightface 2.0 (`buffalo_sc`), torch (CPU), lightgbm and pyarrow installed.

**Verified:** `pytest` (160 tests, including the torch/LightGBM/Parquet paths); the MediaPipe landmarker on a real face (pose, iris, blendshapes); the COCO detector, whose person detections match raw YOLO output; InsightFace embeddings (512-d; the same face darkened scores 0.96 cosine); and `proctorlens replay` end to end on a real-face video, where a covered-camera stretch gave `MONITORING_DEGRADED` and not `FACE_ABSENT`.

**Web app, verified:** the server protocol end to end with fake models (`tests/unit/test_server.py`), and `frontend/e2e.mjs` with the real models in Edge on a fake camera: camera check, calibration, exam, the covered camera became MONITORING_DEGRADED (not FACE_ABSENT), the bus photo MULTIPLE_PEOPLE, the empty room FACE_ABSENT, focus loss and paste BROWSER_INTEGRITY, every event got an H.264 clip, the report exported. **Not verified:** a real candidate on a real webcam, other browsers than Edge, more than one simultaneous session, an HTTPS deployment.

**Not verified:**
- Faces under roughly 100-150 px tall in the frame were not detected by MediaPipe; webcam framing is fine, wide group shots are not.
- The yaw/pitch/roll sign convention against a live camera (`docs/DECISIONS.md` ADR-008), and whether `eye_open` is scaled correctly (it read exactly 1.0 on a real face).
- The interactive calibration and live window were run only with a fake camera (a real face photo) and on one real webcam session before the exit/speed fixes; the recorder, rendered overlay video and the ONNX detector path (the config uses the `.pt` weights) were not run.
- No dataset exists yet: no participants recorded, no trained or fine-tuned models, no measured accuracy. `docs/EVALUATION.md` is an empty protocol, and thresholds, including the identity threshold, are untuned starting values.

## Known limits

A webcam cannot see hands, laps or second screens; there is no liveness detection, so a photo or video replayed to the camera could fool identity and presence; gaze is coarse and only zone-level claims are made; mouth activity is visual-only; accuracy can vary with lighting, glasses, skin tone, head coverings and camera quality; scripted behavior is not real-world behavior. Atypical gaze or movement is not treated as inherently notable, and the policy toggles exist so thresholds can be relaxed per person.

## Layout

```
src/proctorlens/   core/ (types, config, clock)   perception/ (quality, landmarks, head_pose, gaze, objects, identity)
                   features/ (schema, extractor, windows)   temporal/ (state_machine, excursions, scores_rule, scores_learned, tcn)
                   explain/ (details, attribution, overlays, review, evidence, report)   pipeline/ (runner, calibration)
                   server/ (app: REST + websockets, live: one candidate session, db: SQLite)   io.py   cli.py
frontend/          React app: Candidate.tsx (exam flow), Proctor.tsx (login, sessions, live), Review.tsx; e2e.mjs
ml/                data/ (labels)   training/   evaluation/ (matching, metrics, run_eval)
tools/             collect/record.py   fake_cam.py (fake webcam for the web E2E run)
configs/           pipeline.yaml (rates, model paths, scorer)   policy.yaml (thresholds, toggles)
docs/              SPEC, DECISIONS, DATA_PROTOCOL, EVALUATION, model_cards/
tests/             unit/, run.py
data/              git-ignored except manifest.csv
```
