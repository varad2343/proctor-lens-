# Decisions and deviations from SPEC.md

ADR-style log. SPEC.md stays the source of truth; every place the code differs from it, or fills a gap it left,
is recorded here with the reason. Add a new entry (never rewrite an old one) when a decision changes.
Status: **Accepted** unless noted.

## ADR-001 Dataclasses + YAML config instead of Pydantic

- **Spec:** Sections 2, 3, 4.2, 9.2 call for Pydantic v2 models for config and types.
- **Decision:** `core/config.py` and `core/types.py` are plain `dataclasses` + PyYAML. Pydantic is not installed in the
  target environment and adds nothing the config needs.
- **What we kept:** a strict loader. `load_config(*paths)` deep-merges each YAML over the defaults; unknown keys raise
  `ValueError`; `validate()` checks `off_thr < on_thr`, known event keys, known `policy.active` keys and
  `scorer.provider`. `config_hash()` (first 12 hex of SHA-256 over the sorted-key JSON of the full config) goes in every
  `events.json` header.
- **Consequences:** no type coercion. YAML `t_on_s: 2` and `t_on_s: 2.0` load fine but hash differently, so write
  floats as floats in YAML. Field names are the same as a Pydantic model would use, so swapping `from_dict` for Pydantic later is mechanical.
- **Tests:** `tests/unit/test_config.py`.

## ADR-002 Delivery is CLI-only; results are files

- **Spec:** Sections 1.3, 2, 9.1, Appendix A.
- **Decision:** no web UI, API, database, auth, reports or review workflow. Three entry points (`replay`, `live`,
  `extract-features`, plus `calibrate`) call the same `Pipeline`. Outputs are `features.parquet` and `events.json`; no
  frames or video are written unless `--render` (or the dataset recorder) is used.
- **Consequence:** the OpenCV window is the only UI, and it is also how calibration dots are shown.

## ADR-003 Stack deviations from SPEC 9.2

| Spec | Here | Why |
|---|---|---|
| Pydantic v2 | dataclasses + PyYAML | ADR-001 |
| pytest + Hypothesis | fixture-free `test_*` functions with bare asserts; `python tests/run.py [substring]` runs them without pytest, pytest also collects them; property-style checks are seeded numpy loops | pytest/Hypothesis not installed in the dev environment |
| uv / Poetry lockfile | `pyproject.toml` + pip | no lockfile tool available; pin exact versions (`pip freeze`) when results are frozen for the report |
| mypy on core | type hints only, ruff for style | not installed; low value at this size |
| Python 3.11+ | developed on 3.12 | `requires-python >=3.11` kept |

Heavy libraries (mediapipe, ultralytics, insightface, onnxruntime models, torch, lightgbm, pyarrow) are imported lazily
inside functions or constructors. `import proctorlens.<anything>` therefore works with only numpy, pandas, OpenCV,
scipy, scikit-learn and PyYAML, and the testable logic is kept in pure numpy/pandas functions behind thin adapters
(SPEC Section 18, "adapter layers around MediaPipe/YOLO/InsightFace").

## ADR-004 Fixed 10 Hz grid: resampling rule

- **Spec:** 4.2 ("resampled (last value, with staleness flag); missing steps are explicit, never interpolated").
- **Decision** (`core/clock.py`, `GridResampler`):
  - Grid step is `1000 // hz` ms (100 ms at 10 Hz). The first grid step is the first multiple of the step at or after the first frame.
  - A grid step T uses the latest frame with `t <= T`, never a later one. If that frame is older than `max_age_ms` (default 250, inclusive) the step carries `item=None`, which becomes `frame_valid=False`.
  - A step is emitted only once a frame with `t >= T` has arrived, so a long gap produces one `(T, None, age)` per missed step and nothing is interpolated.
  - Duplicate or out-of-order frames (`t <=` last accepted) are dropped and leave no trace.
  - The age is reported even for stale steps, and feeds `frame_age_ms`.
- **Consequences:** a step is emitted at most one frame interval after its time. Steps after the last frame are never emitted; the pipeline closes open events with `finish(t_ms)`. `stall_ms` (2000) of consecutive invalid steps becomes `quality_reasons = frame_gap`, which counts as degraded.
- **Tests:** `tests/unit/test_clock.py` (jitter, gaps, staleness boundary, duplicates, out-of-order, random-stream invariants).

## ADR-005 Config defaults

Defaults live in `core/config.py`; `configs/pipeline.yaml` and `configs/policy.yaml` restate only what a user is expected to edit. These are starting values from SPEC Sections 3, 5 and 7, **tuned on validation data only, never on test**. Values the spec did not fix are marked *ours* and are open to change once pilot data exists.

| Setting | Default | Source |
|---|---|---|
| `pipeline.grid_hz`, `yolo_every`, `identity_every_s` | 10, 3, 10 s | SPEC 4.2 |
| `pipeline.max_age_ms`, `stall_ms` | 250, 2000 | ours / SPEC 4.2 ("stall > 2 s") |
| `pipeline.enroll_seconds` | 5 | ours (SPEC: "~5 quality-gated frames from the first `enroll_seconds`") |
| `pipeline.turned_away_yaw`, `turned_away_hold_ms` | 50 deg, 1500 ms | ours, below the 60-70 deg tracker limit in SPEC 5.2 |
| `pipeline.static_window_s`, `static_iou`, `min_second_face_frac` | 20 s, 0.92, 0.08 | ours (SPEC 5.4 asks for "near-zero motion over a long window" and a minimum face size) |
| `events.face_absent` | t_on 3 s | SPEC 3 (E1) |
| `events.multiple_people` | t_on 2 s | SPEC 3 (E2) |
| `events.phone` / `events.notes` | t_on 1.5 s / 3 s | SPEC 3 (E3) |
| `events.off_screen` | t_on 4 s, off_thr 0.35 | SPEC 3 (E4); off_thr is ours |
| `events.speaking` | min_dur 3 s, off_thr 0.35 | SPEC 3 (E6); the "3 s in 10 s" window rule lives in the scorer |
| `glance` | n 4, window 60 s, excursion 0.3-3 s, same-zone 0.7 | SPEC 3 (E5) |
| `identity` | tau 0.35, 3 consecutive checks, frontal within 20 deg yaw | SPEC 3 (E7); **tau is provisional**, to be calibrated on our data (ADR-009) |
| `gaze` | accept < 0.15, max 2 tries, margin 0.12, yaw/pitch limit 25/20 deg | SPEC 5.3; limits are ours |
| `quality` | luma 40-220, blur 30, face >= 10% of frame height, min quality 0.5 | ours (SPEC 5.1 names the measures, not the values) |
| `scorer.window` | 60 steps (6 s) | SPEC 7.4 |
| `policy.allow_*` | all false (strictest) | SPEC 3; relax per person for fairness |

## ADR-006 Feature schema v1 differs slightly from SPEC Section 6

- `face_bbox` is four columns `face_x0..face_y1` (normalized), so the table stays flat and Parquet-friendly.
- Added `turned_away` (SPEC 5.2 `TURNED_AWAY` state) and `reliable` (`frame_valid and quality >= quality.min_quality`).
- `quality_reasons` is a string of reason codes; other string column: `zone`. Everything else is numeric or bool, NaN meaning undefined.
- `MODEL_COLUMNS` (35 numeric inputs to the learned scorers) exclude identity, timing and strings (SPEC 7.4). `ATTR_GROUPS` (head pose, eye, mouth, quality) partition them exactly, which is what occlusion attribution (SPEC 8) needs.
- `SCHEMA_VERSION` is written into every `events.json` header. Any column change bumps it.
- **Tests:** `tests/unit/test_schema.py`.

## ADR-007 Event machines: one per score key

- **Spec:** 7.1-7.3. **Decision:** `core.types.MACHINE_EVENT` maps nine score keys to the eight event types; `phone` and `notes` are separate machines (different `t_on`) that both emit `PROHIBITED_OBJECT`.
- Window rules live where the spec puts the evidence: REPEATED_GLANCING in `temporal/excursions.py`, the speaking "3 s in 10 s" rule and the identity "3 consecutive checks" rule in the score providers. Their machines therefore run with `t_on = 0` (`glancing`, `speaking`, `id_mismatch` in `default_events()`).
- Face-derived keys (`core.types.FACE_DERIVED`) are gated on `reliable` and forced to `None` while the `degraded` machine is ACTIVE (SPEC 7.3 cross-event suppression). `degraded` itself is never `None`.
- A score of `None` starts nothing, pauses debounce timers, and closes an ACTIVE event after `max_hold_s` (5 s) of consecutive `None`.

## ADR-008 Head-pose convention and its status

- **Decision** (contract for `perception/head_pose.py`): degrees; yaw > 0 turns the face toward image-right, pitch > 0 tilts up, roll > 0 is clockwise in the image. `rotation_matrix` and `euler_from_matrix` are exact inverses for |pitch| < 90.
- **Status:** the convention is tested with synthetic rotations (SPEC 5.2). Whether MediaPipe's facial transformation matrix follows it, including the sign of each axis and any mirroring, has **not** been checked against a real landmarker output (SPEC 19.1). Do this first once `mediapipe` is installed: look left/right/up/down in front of the camera and confirm the signs; fix in `perception/landmarks.py` only.

## ADR-009 Identity threshold and embeddings

- tau = 0.35 is a placeholder, not a tuned value. It gets calibrated on our data (genuine pairs across conditions vs impostors from other participants; report EER and TAR@FAR = 1%) with `ml/training/calibrate_identity.py`.
- Enrollment embeddings and the mean embedding exist in memory only and are never written to disk (SPEC 5.5, 14). Replay without `--render` writes only `features.parquet` and `events.json`.
- Needs >= 3 enrollment embeddings (from quality-ok frontal frames in the first `enroll_seconds`) to become ready; otherwise identity output is always `None` and no identity event can fire.
- Identity is optional: the checker is built only if the `models.identity` directory exists and insightface imports (`perception/__init__.py`); otherwise identity checks are skipped with a warning.

## ADR-010 Temporal model artifacts

- **Spec:** 15 says `models/<name>/<version>/{weights, model_card.md, hash}`.
- **Decision:** `data/models/temporal/<name>/<version>/{meta.json, model.txt | model.pt}`. `meta.json` holds `kind` (gbm|tcn), `target` (off_screen|speaking), `columns`, `window`, Platt `calib {a, b}`, `extra` (TCN build kwargs, feature mean/std) and `version`. The model card lives in `docs/model_cards/<name>.md` (from `TEMPLATE.md`), not beside the weights, because `data/` is not tracked. `configs` point at the directory via `scorer.models`.
- Both learned scorers must be causal so live and replay match (SPEC 7.4); the TCN test checks that changing future inputs never changes the output at t.

## ADR-011 Feature storage

`io.save_features` writes Parquet via pandas and uses CSV if the path ends with `.csv` or pyarrow is missing (the suffix then becomes `.csv` and the written path is returned); `load_features` reads either and falls back from a missing `.parquet` to its `.csv` sibling. `pyproject.toml` still lists pyarrow as a dependency; the fallback only keeps the pipeline usable on a machine without it.

## ADR-012 Label and protocol conventions

Choices SPEC 10.3 left open, recorded so annotators and evaluation agree (details in `DATA_PROTOCOL.md`):
- Ground-truth `type` is an `EVENT_TYPES` value or a `benign_*` label. Continuous off-screen attention shorter than the E4 duration is `benign_brief_away`, unless it is part of a repeated-glancing episode.
- Cue-log rows become labels with `source=cue`, `annotator_id=recorder`; annotator-corrected rows use `source=annotator`. Evaluation sessions use only annotator-corrected boundaries.
- Changing a definition after labeling starts means relabeling. Decide before the pilot ends.

## ADR-013 Third-party assumptions (SPEC Section 19): status

Not verified against live library docs in the environment this code was written in (none of the libraries were installed), so the adapters are written against the documented APIs and kept thin. **Verify before relying on them:**

| Assumption | Where it matters | Status |
|---|---|---|
| MediaPipe Tasks Face Landmarker: VIDEO mode, blendshapes, transformation matrices, `num_faces`, model URL, matrix Euler convention | `perception/landmarks.py`, ADR-008 | unverified |
| Ultralytics: nano model name (YOLO11n or newer), export to ONNX, AGPL-3.0 | `perception/objects.py`, `ml/training/train_detector.py` | unverified |
| InsightFace `buffalo_sc`: how to obtain, preprocessing, non-commercial license | `perception/identity.py` | unverified |
| Public datasets (AFLW2000, BIWI, Roboflow sets): availability and license | optional head-pose MAE, detector data | unverified |

If an assumption fails, choose the closest alternative, add a superseding ADR here, and keep the `perception/*` interfaces stable.

## ADR-014 Wording guard

Principle 1 (no verdicts) is enforced by a repo check, `test_no_accusatory_wording` in `tests/unit/test_schema.py`: no accusatory vocabulary in `src/`, `ml/`, `tools/`, `configs/`, `README.md` or `docs/` (SPEC.md is exempt because it names the ban). Events are observations; `confidence` is detector confidence that the observation is real.

## ADR-015 Evidence and review layer restored (gap plan Phase 1)

- **Spec:** SPEC 1.3 and Appendix A cut clips, keyframes, explanation templates, segment merging, review priority and
  session reports. The master-spec gap plan (Phase 1, approved) brings them back as **post-processing on replay
  output**, so the pipeline, `features.parquet` and `events.json` are unchanged and evaluation still describes them.
- **Decision:** `proctorlens report <out> [--video V]` (or `replay --report`) writes `segments.json`, `report.html`
  and, with a video, `evidence/` (per event: H.264 clip from start - 5 s to end + 3 s, capped at 30 s, plus onset /
  peak / end keyframes). Code: `explain/review.py` (segments, priority, text), `explain/evidence.py`, `explain/report.py`.
  - Clips are cut from an existing file (the recording, or the `--render` overlay video, which `replay --report`
    prefers so boxes and gaze arrows are burned in). No ring buffer: live mode writes no evidence (a ring buffer comes
    with a streaming server, gap plan Phase 2). Clips need the `ffmpeg` binary on PATH (OpenCV's `mp4v` does not play
    in browsers); without it only keyframes are written.
  - Segments (master spec 9): events overlapping or within `review.merge_gap_s` (5 s) join one segment;
    `MONITORING_DEGRADED` forms its own "blind spot" lane. `review_priority = sum(w_type * confidence *
    (1 + ln(1 + duration_s))) * (1 + multi_bonus * (distinct types - 1))`; master-spec weights; `multi_bonus` 0.25 and
    the label cut-offs (high >= 2.0, medium >= 1.0) are ours, to be tuned on validation data (recall@K) once it exists.
    It is unbounded rather than 0-1 (the master-spec formula); it orders review, it never scores a person.
  - Explanations are fixed templates over `details` (no language model); `details` gained `peak_ms` (time of the
    peak value) for the peak keyframe.
- **Consequences:** `report` writes video frames of people; it runs only when asked, and only from a video already on
  disk. Reviewer decisions (confirm / dismiss) need persistence and stay in Phase 2.
- **Tests:** `tests/unit/test_review.py`, `test_replay_report_flag_cuts_evidence_from_the_render`.

## ADR-016 Web application (gap plan Phase 2)

- **Spec:** SPEC 1.3 / Appendix A cut the web app, server, database, auth, browser telemetry and reviewer workflow.
  Gap plan Phase 2 (approved) restores them as master spec 10-11 describe, **around** the unchanged pipeline: the
  server feeds candidate frames to the same `Pipeline`, so replay, live and the web app still share one code path.
- **Decision:** `proctorlens serve` (FastAPI + Uvicorn; `src/proctorlens/server/`) serves REST + `/ws/stream` +
  `/ws/proctor` and the built React app (`frontend/`, React 18 + TypeScript + Vite + Tailwind). Binds 127.0.0.1 by
  default; cameras need localhost or HTTPS, so another machine needs HTTPS in front.
- **Deviations from master spec 10-11, and why:**
  | Master spec | Here | Why |
  |---|---|---|
  | SQLModel + migrations | stdlib `sqlite3`, `CREATE TABLE IF NOT EXISTS` (`server/db.py`) | six small tables; a schema change = a new ADR |
  | `Segment` table | recomputed from events on request | derived data cannot go stale |
  | proctor JWT | random token in an HttpOnly SameSite=Strict cookie, held in memory | one local account; a restart logs out |
  | TanStack Query, Recharts, router | a fetch hook, server-rendered SVG signal plots (`report.plot_svg`), hash routes | same result, three fewer dependencies |
  | `enroll_start/finish` messages | enrollment = the first `enroll_seconds` of the exam (ADR-009), shown as "reference picture" | one enrollment path for replay, live and web |
  | overlay toggle in the evidence viewer | clips are the raw camera frames; keyframes have a detector-overlay twin (ADR-017) | an overlay clip doubles encoding per event; the keyframes cover the check |
  | live view frames over WebSocket | `GET /sessions/{id}/frame.jpg` polled ~1/s | simpler; enough to follow |
- **Protocol** (`/ws/stream/{id}?token=`): binary = `seq uint32 LE, t float64 LE` + JPEG, 640 px wide, 10 fps; text =
  JSON control (`hello`, `calib_start/point/end`, `exam_start/end`, `browser_event`, `ping`). The client clock is
  `performance.timeOrigin + performance.now()` (epoch ms, monotonic in a page, comparable across a reload), so a
  reconnect resumes the same timeline and the gap is a frame gap (MONITORING_DEGRADED), never absence. The server acks
  every processed frame (latest frame wins; the client keeps at most 2 unacknowledged) and, when no frame arrives,
  advances the grid on its own clock so a stall is degraded while it happens.
- **Calibration** over the web: the server sends the `dot_schedule`, the client reports when each dot appears, and
  frames after `gaze.settle_s` of a dot are samples (the time-based rule of `calibrate_from_frames`, not the CLI's
  stability-gated dwell). Fitting, acceptance and the head-pose-only fallback are unchanged.
- **Evidence:** a 40 s in-memory ring buffer of JPEGs per session; when an event's clip window has passed, its clip
  (H.264 via ffmpeg) and keyframes are encoded on a background thread. No continuous recording exists on disk.
  ponytail: an event longer than ~35 s keeps only its last part, and a clip can start later than `start - 5 s`.
- **Browser telemetry** returns as `BROWSER_INTEGRITY` (`core.types.REVIEW_TYPES`; not in `EVENT_TYPES`, because no
  state machine or evaluation label exists for it): hidden..visible, blur..focus, fullscreen exit..enter,
  offline..online become intervals; paste, copy, context menu, unload are instants. Pasted text is never sent.
- **Privacy:** retention `PROCTORLENS_RETENTION_DAYS` (default 30) is applied at server start; `DELETE
  /api/sessions/{id}` removes rows and files; a session with no events leaves only features, events and summary.
- **Not built:** cue-app / annotation modes (record.py and ELAN / Label Studio cover them), PDF reports, audio.
- **Tests:** `tests/unit/test_server.py` (fake models, the whole protocol); `frontend/e2e.mjs` (real models, Edge with
  a fake camera from `tools/fake_cam.py`).

## ADR-017 Display-only tracking and annotated proctor views

- **Request:** "full advanced face tracking and object detection", with a browser prototype (face / multiple-person /
  gaze / phone detection, tab and fullscreen monitoring) as the reference. All of those already exist as events
  (FACE_ABSENT, MULTIPLE_PEOPLE, OFF_SCREEN_SUSTAINED / REPEATED_GLANCING, PROHIBITED_OBJECT, BROWSER_INTEGRITY), so
  the addition is making the tracking itself visible to the proctor and reviewer. Chosen by a design panel (four
  independent designs, two judges, one synthesis) for value at the lowest risk to measured behaviour.
- **Decision:**
  - `perception.objects.IdTracker`: IoU + Hungarian (`scipy.optimize.linear_sum_assignment`) track numbers, class-gated,
    a track ends after 1000 ms unseen (on `t_ms`, so replay is deterministic), numbers are never reused. Run in
    `Pipeline.process` for faces (every frame) and detector boxes (fresh detector frames; carried frames inherit).
    Stored on `Face.track` / `Detections.ids`.
  - `Face.mesh`: the 478 landmark xy (float32), for drawing only. `draw()` adds the MediaPipe contour mesh (eyes,
    brows, lips, face oval, irises) and the points for faces >= 40 px tall, track labels on boxes (kept below the
    banner), and no zone text when there is no zone.
  - Detector NMS is per class (it was persons only): the ONNX raw output layout had duplicate phone / book boxes.
  - Web app: `GET /api/sessions/{id}/frame.jpg?overlay=1` draws the latest exam frame with the shared `draw()` (raw
    stays the default); the Live page shows it (on by default, unmirrored) with proctor-only readouts (faces / people
    in view, phone and book detector confidence, head turn, processing ms per frame). Each evidence keyframe gets an
    `<id>_<name>_ov.jpg` twin and the review screen a "detector boxes" toggle; clips stay raw camera frames.
- **What does not change:** track numbers, ids and the mesh feed nothing in features, scores or events: `COLUMNS`,
  `SCHEMA_VERSION` (1), `MODEL_COLUMNS`, `features.parquet`, `events.json`, the DB schema and the 171-test suite
  (167 before) hold, and replay output is identical. The mesh is never written anywhere and is stripped from the
  evidence ring buffer. The candidate's status messages are unchanged (a test guards it).
- **Track numbers are not identities:** they follow boxes; a face that leaves for over a second comes back with a new
  number, and two people crossing can swap numbers. No count of "distinct people" is derived from them anywhere,
  because it would read like a people count. ponytail: no motion model; add centre-distance gating if numbers churn
  on real recordings.
- **Not built** (judged too risky or not worth it now): a second-screen class (COCO laptop / tv: false positives, ~9
  contract changes), Ultralytics `model.track` (ByteTrack: `.pt` only, installs `lap` at runtime, no faces), YOLO on
  every frame (~58 ms per call on this CPU), re-identification across exits, overlay video clips, a multi-session grid,
  a gaze heatmap, anything shown to the candidate.
- **Measured** (this laptop, CPU): landmarker 16 ms, YOLO 56-59 ms per call (every 3rd frame), whole perception
  stack 34 ms median per frame, overlay draw + JPEG 1.3 ms.
- **Tests:** `test_objects` (per-class NMS, IdTracker), `test_perceiver` (carried ids), `test_pipeline_fake` (numbers
  across an absence, determinism), `test_overlays` (labels, mesh), `test_landmarks` (mesh), `test_server` (overlay
  frame, proctor-only readouts, overlay keyframes); `frontend/e2e.mjs` checks overlay keyframes with real models.
