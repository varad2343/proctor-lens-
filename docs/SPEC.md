# ProctorLens — ML/CV Core Specification

**Project:** ProctorLens — a Python machine-learning / computer-vision pipeline that turns webcam video into time-stamped, explainable, *observable* events (gaze away from screen, phone visible, second person present, …).
**Context:** Machine Learning + Computer Vision course project (team of 1–4 students, ~10 weeks).
**Audience:** an AI coding agent (or engineer) who knows nothing about the project. Read fully before writing code.
**Status:** Design complete, no code yet. Derived from `PROCTORLENS_MASTER_SPEC.md` with scope cut to **strictly ML/CV**. Appendix A lists everything removed. Deviations go in `docs/DECISIONS.md`.

---

## 0. How to use this document

- **MUST / MUST NOT** = hard requirement. **SHOULD** = default unless documented otherwise. **MAY / STRETCH** = only after all MUST items work.
- **P0** = core, **P1** = important, **P2** = stretch.
- **Scope rule:** an item is in scope only if it is (a) a perception model, (b) a learned or rule-based temporal model over perception features, (c) data/labels for those, or (d) evaluation of those. Anything else (web UI, API, database, auth, reports, workflow) is out of scope — do not build it.
- Section 19 lists third-party-library assumptions to verify against current docs before relying on them.
- Build a **vertical slice first** (video/camera → `FACE_ABSENT` → `events.json`), then deepen.
- **Data collection is the critical path.** It starts in week 2, not after the pipeline is "finished".

---

## 1. Vision, principles, non-goals

### 1.1 What we are building
A pipeline that reads webcam video (a file, or a live camera) and outputs:
1. a **feature time series** at 10 Hz (`features.parquet`), and
2. a list of **observable events** (`events.json`), each with measured values, threshold, quality context, and optional feature-group attribution.

Research contributions: per-user calibrated gaze regression; fine-tuned phone/notes/person detector; a rules-vs-LightGBM-vs-TCN temporal comparison; a self-collected, participant-disjoint labeled dataset; honest event-level evaluation with robustness and fairness analysis.

### 1.2 Non-negotiable principles
1. **No verdicts.** The system reports observable events. It never outputs a "cheating" label or probability. No such wording in code, logs, plots, or docs — use "event", "observed", "flagged".
2. **Every event is explainable:** signal that fired, measured value vs threshold, quality context (Section 8).
3. **Honest about blindness:** when the system cannot see (dark, blur, blocked camera, unusable face, frame gap) it emits `MONITORING_DEGRADED`, never a silent miss or a false `FACE_ABSENT`.
4. **One pipeline, three entry points:** offline replay on a video file, live camera, and dataset feature extraction all call the same `Pipeline` code. Evaluation therefore describes the system that is demoed.
5. **Data governance:** local processing, consent, pseudonymous IDs, minimal storage (Section 14).
6. **Measured honestly:** participant-disjoint splits, participant-level bootstrap CIs, stated limitations.

### 1.3 Non-goals (do not build)
Web frontend; REST/WebSocket server; database; authentication; candidate/exam flow; reviewer decisions or workflow; session reports; segment merging and review-priority scoring; browser telemetry (tab/focus/fullscreen/paste); evidence video-clip encoding; natural-language explanation templates; audio/VAD; liveness/anti-spoofing; ID-card verification; deepfake detection; plagiarism/NLP; multi-camera; multi-session concurrency or scaling; cloud deployment; mobile; LLM/VLM-generated judgments.

---

## 2. Key decisions

| Area | Options considered | **Chosen** | Why |
|---|---|---|---|
| Overall approach | End-to-end video classifier ("cheat vs not"); modular perception + temporal reasoning | **Modular perception → features → temporal reasoning → events** | End-to-end needs large labeled data we lack, is unexplainable, and implies a verdict. |
| Face / landmarks / head pose | Dlib, Haar, MTCNN, RetinaFace+3DDFA, **MediaPipe Face Landmarker** | **MediaPipe Face Landmarker** (478 landmarks incl. iris, blendshapes, 4×4 facial transformation matrix, multi-face) | Real-time CPU, pose + eye + mouth in one pass. Known weakness under extreme pose, glare, low light → handled by quality gating and robustness evaluation. |
| Gaze | Pretrained gaze CNN (L2CS-Net, ETH-XGaze), iris ratios only, **landmark features + per-user calibrated regression** | **Per-user calibrated regression on head-pose + iris + blendshape features** | Cheap, personalized (camera offset, screen size), measurable (calibration error). L2CS-Net = P2 ablation. |
| Objects | Custom from scratch, Faster R-CNN, RT-DETR, **YOLO nano** | **Ultralytics YOLO nano (YOLO11n or newest nano) — COCO-pretrained, then fine-tuned** | Fast on CPU, easy fine-tune, ONNX export. AGPL-3.0 (fine for coursework; document it). |
| Identity | Train a recognizer; **pretrained ArcFace-family embedding** | **InsightFace pretrained ONNX (`buffalo_sc`/MobileFaceNet-class)**, cosine similarity to enrollment | No training; threshold τ calibrated on our data. Non-commercial research license → document. |
| Temporal reasoning | Rules only; deep model only; **rules + hysteresis state machines fed by scores from either rules or a learned model** | **Shared event state machine; two score providers (rule, learned) compared experimentally** | ML depth *and* an honest comparison; deterministic types stay rule-based. |
| Learned temporal model | LSTM/Transformer, **causal TCN**, gradient boosting | **Compare (a) thresholds+smoothing, (b) LightGBM on window stats, (c) small causal TCN; ship the best on validation, ties → simpler** | With ~15 participants a GBM may beat a NN. The experiment is the contribution. |
| Delivery | Web app, service, **Python package + CLI** | **Python package, CLI, OpenCV window for live demo; results are files (Parquet/JSON)** | Nothing outside ML/CV to build or maintain. |
| VLM/LLM "second opinion" | Use a VLM to describe/judge clips | **Rejected** | Hallucination risk; drifts toward accusatory verdicts (violates principle 1). |

---

## 3. Event taxonomy

Every event record: `type`, `start_ms`, `end_ms`, `confidence` (0–1, *detector confidence that the observation is real*, not guilt), `detector` (rule/model + version), `details` (Section 8).

| ID | Event | Observable definition | Method | Learned? | Pri |
|---|---|---|---|---|---|
| E1 | `FACE_ABSENT` | No face and no person in view ≥ `t_absent` (default 3 s), after excluding degraded-monitoring causes | Landmarker + person detector + rules | No | P0 |
| E2 | `MULTIPLE_PEOPLE` | ≥2 distinct live faces or persons persistently visible (default ≥2 s) | Landmarker (`num_faces=3`) + YOLO `person`, static-face suppression | Detector fine-tuned | P0 |
| E3 | `PROHIBITED_OBJECT` | Phone visible ≥1.5 s (and notes/book ≥3 s if policy forbids) | Fine-tuned YOLO | Yes | P0 |
| E4 | `OFF_SCREEN_SUSTAINED` | Attention off screen continuously ≥ `t_sustain` (default 4 s), with direction zone (left/right/up/down) | Calibrated gaze + head pose → temporal score → state machine | Yes (calibration + temporal model) | P0 |
| E5 | `REPEATED_GLANCING` | ≥ N (default 4) short off-screen excursions (0.3–3 s) within 60 s, ≥70% toward the same zone | Excursion counter on E4's off-screen state | Via E4 | P0 |
| E6 | `MOUTH_ACTIVITY` | Sustained mouth movement consistent with speaking (default ≥3 s active within a 10 s window) | Blendshape/landmark mouth features → temporal model | Yes | P1 |
| E7 | `IDENTITY_MISMATCH` | Embedding similarity to enrollment < τ for ≥3 consecutive high-quality checks | InsightFace embedding | Pretrained, τ calibrated | P1 |
| E8 | `MONITORING_DEGRADED` | System cannot reliably observe: dark, blurry, camera blocked, face too small/partially out of frame, frame gap/stall, calibration drift | Quality module | No | P0 |

Chosen because each is observable from one webcam, distinct enough to evaluate independently, and defensible as "worth a human look". Hands-under-desk, earbuds, and second monitors are not reliable at webcam resolution → documented limitations, not features.

**Config profile** (`configs/policy.yaml`, Pydantic-validated): thresholds above, margins, active events, and toggles `allow_notes`, `allow_looking_down`, `allow_reading_aloud`. Adjustable toggles matter for fairness (different working styles, disabilities). The config hash is written into every `events.json`.

---

## 4. Pipeline architecture

### 4.1 Diagram

```
 Source: video file (replay) | camera (live) | dataset dir (extract-features)
                     │  frames + t_ms
                     ▼
 Frame ─► Quality ─► FaceLandmarker ─► HeadPose / Gaze / Mouth features ─┐
           │            (every frame)                                   ├─► Feature row @10 Hz ─► features.parquet
           ├─► YOLO (every ~3rd frame): phone / notes / person ─────────┤
           └─► Identity (every ~10 s or on trigger) ────────────────────┘
                                                                        │
            Score providers: RuleScores │ LearnedScores (GBM / TCN) ◄───┘
                                   │
                          Event state machines (hysteresis) ─► events.json
                                   │
                          Overlay renderer (optional) ─► annotated video / live window
```

### 4.2 Core design rules
- **Pure, stateful-by-object pipeline.** `Pipeline.process(frame, t_ms) -> FrameResult`; all state in explicit objects (no hidden globals) → deterministic replay and unit-testable.
- **Time discipline.** Replay uses container timestamps; live uses the camera's monotonic capture clock. Temporal modules run on a **fixed 10 Hz grid**; jitter and variable-frame-rate video are resampled (last value, with staleness flag). Missing steps are explicit (`frame_valid=false`), never interpolated across long gaps.
- **Staggered compute.** Landmarker every processed frame; YOLO every ~3rd frame; identity every ~10 s or on trigger (e.g., return after absence); quality every frame.
- **Live backpressure.** Latest-frame-wins; record effective FPS; a stall >2 s raises `MONITORING_DEGRADED`. (Replay processes every frame; no dropping.)
- **Performance budget.** ≤100 ms per processed frame on a modern laptop CPU (target 10 fps, floor 5 fps). Export YOLO to ONNX/OpenVINO if CPU latency is too high.
- **Config over constants.** Thresholds in YAML → Pydantic models; none hard-coded in logic.

---

## 5. Perception components

### 5.1 Quality module (P0)
Per frame: mean luminance, over/under-exposure fraction, blur (variance of Laplacian on the face ROI or full frame), face size (face-box height / frame height), landmark confidence/presence, face-fully-inside-frame. Output `quality ∈ [0,1]` plus reason codes. **Gating rule:** gaze/mouth/identity scores are marked `unreliable` when quality is below threshold; persistent unreliability becomes `MONITORING_DEGRADED`. Classical CV, no training.

### 5.2 Face landmarks, head pose, eye and mouth features (P0)
- **Model:** MediaPipe Face Landmarker (Tasks API), `num_faces=3`, blendshapes + facial transformation matrices enabled, video mode with timestamps. Pretrained.
- **Head pose:** decompose the transformation matrix into yaw/pitch/roll (degrees). Document the axis/sign convention in a unit test using synthetic rotations. Cross-check (P1): OpenCV `solvePnP` with a fixed 6-point 3D model.
- **Eye features:** iris-center position relative to eye corners/lids (horizontal/vertical ratios, both eyes), eyelid openness, blendshapes `eyeLook{In,Out,Up,Down}{Left,Right}`, blink rate.
- **Mouth features:** mouth aspect ratio, blendshapes `jawOpen`, `mouthClose`, `mouthFunnel`, `mouthPucker`, `mouthSmile*`; short-window mouth-motion energy. Speaking = rhythmic small-amplitude motion, unlike a single yawn (big, slow) or smile.
- **Face-lost handling:** if the face disappears shortly after |yaw| approached the tracker limit (~60–70°), state is `TURNED_AWAY` (feeds E4), *not* `FACE_ABSENT`.
- **Extreme pose:** beyond a yaw/pitch limit, landmark-derived eye/mouth features are `unreliable`, but head-pose-only off-screen evidence still counts.

### 5.3 Gaze estimation with per-user calibration (P0 — ML/CV contribution #1)
**Goal:** a coarse, honest estimate of where attention is directed: `on_screen` or a zone `left/right/up/down/off`, plus normalized screen coordinates when on-screen.

**Calibration procedure (~25 s, run at the start of each recording by `proctorlens calibrate` / the recorder's fullscreen OpenCV window):**
1. Subject looks at screen center for 2 s → store **baseline head pose** (yaw/pitch/roll, face position and scale).
2. A dot visits 9 positions (3×3 incl. corners) for 1.5 s each (discard the first 0.5 s for saccade settling), plus 4 random *validation* dots. Dot positions and timestamps are logged to a calibration log file.
3. Feature vector per sample: `[Δyaw, Δpitch, Δroll, iris ratios (4), eye-look blendshapes (8), face center x/y, face scale]` (Δ relative to baseline).
4. Fit a **regularized regression** (ridge; optional degree-2 polynomial or tiny MLP — choose by leave-one-point-out CV) mapping features → normalized screen (x, y) ∈ [0,1]².
5. Report calibration quality: median validation-dot error (normalized units). Accept if < 0.15 of screen width; else recalibrate (max 2 tries), then fall back to **head-pose-only mode** (recorded in the session metadata as reduced accuracy).

**Off-screen decision per step:** `off_screen_score = max(gaze_outside_score, head_turn_score)`
- `gaze_outside_score`: soft function of the predicted point's distance outside the screen rectangle expanded by a margin (config, default 10–15%);
- `head_turn_score`: soft function of |Δyaw|/|Δpitch| vs limits beyond the calibrated range (regression cannot extrapolate, so large turns are caught geometrically);
- zone from the sign/dominant axis of deviation.
Smoothing happens in the temporal layer, not here.

**Drift:** if the running median of on-task gaze predictions (~2 min) drifts from screen center beyond a bound, or baseline head position shifts, emit `MONITORING_DEGRADED(calibration_drift)`.

**Honest accuracy:** webcam landmark-based gaze is coarse (several degrees). Claim zone-level attention only.

### 5.4 Object and person detection (P0 — ML/CV contribution #2)
- **Model:** Ultralytics YOLO nano, COCO-pretrained init.
- **Classes after fine-tuning:** `phone`, `notes` (book/notebook/loose paper), `person`; `headphones` (P2, only if data supports it).
- **Pretrained-vs-fine-tuned study (required):** evaluate off-the-shelf COCO (`cell phone`, `book`, `person`) vs fine-tuned on the held-out, participant-disjoint test set: per-class AP50 / AP50-95, precision/recall at the deployed threshold, and the effect on **event-level** metrics.
- **Training data:** frames from our sessions (phone held/on desk/lap-edge, notes on desk/held, second person partial/full), plus license-checked public datasets (verify license/quality; never depend on them), plus a COCO subset (phones/books/persons) against forgetting. Pre-label with the COCO model, correct in CVAT/Label Studio. Target ≈1,500–3,000 labeled images, split **by participant**; include hard negatives (remote, wallet, calculator).
- **Recipe:** imgsz 640, 50–80 epochs, pretrained init, optional backbone freeze for early epochs, deployment-like augmentations (HSV/brightness, blur, JPEG compression, mosaic), early stopping on val mAP. Colab/Kaggle GPU is fine. Export ONNX; benchmark CPU latency.
- **Temporal use:** detections are never events by themselves. They feed per-class *presence scores* (EMA of confidence or fraction of positive detections in a sliding window) into the state machine.
- **Second-person logic (E2):** fire if the count of *distinct, non-static* faces ≥2 **or** `person` detections ≥2 (after NMS) persistently. **Static-face suppression:** a face/person box with near-zero motion at an identical position over a long window (poster, photo, TV) is tagged `static_background` and excluded; also require a minimum face size to ignore tiny background faces.

### 5.5 Identity verification (P1)
- **Enrollment:** embed ~5 quality-gated frontal frames from the first `enroll_seconds` of the recording (or an explicit reference image); keep the **mean embedding in memory only**, never written to disk by default.
- **Checks:** embedding every ~10 s on the primary face crop (skip low-quality/extreme-pose frames) using InsightFace detect+align+embed.
- **Rule:** cosine similarity < τ for ≥3 consecutive high-quality checks → `IDENTITY_MISMATCH`. τ is **calibrated on our data** (same-person across conditions vs impostors from other participants; report EER and TAR@FAR=1%), not copied from a paper.
- **Limitation:** lighting/pose/glasses changes lower similarity → consecutive-check rule and quality gating.

---

## 6. Feature schema (10 Hz output of perception)

One row per time step, stored per recording as `features.parquet`; also the **training input** for the temporal models and the **replay output** for evaluation. Version it (`feature_schema_version`) and freeze it early.

| Group | Fields |
|---|---|
| Timing | `t_ms`, `frame_valid`, `frame_age_ms`, `effective_fps` |
| Face | `n_faces`, `primary_face_present`, `face_bbox`, `face_size_frac`, `landmark_conf`, `time_since_face_ms` |
| Head pose (vs baseline) | `d_yaw`, `d_pitch`, `d_roll`, `head_x`, `head_y`, `head_scale`, `head_speed` |
| Eyes / gaze | `iris_lx, iris_ly, iris_rx, iris_ry`, `eye_open_l/r`, `blink`, `eyelook_*` (8), `gaze_x`, `gaze_y`, `gaze_in_screen_prob`, `off_screen_score`, `zone` |
| Mouth | `jaw_open`, `mouth_close`, `mar`, `mouth_energy_1s` |
| Objects | `phone_conf`, `notes_conf`, `n_persons`, `static_face_flags` |
| Identity | `id_similarity` (sparse, NaN between checks), `id_quality_ok` |
| Quality | `luma`, `blur`, `overexp_frac`, `quality`, `quality_reasons` |

---

## 7. Temporal and event reasoning

### 7.1 Architecture
```
Perception features ─► Score providers ─► per-step scores s_k(t) ∈ [0,1] ─► Event state machines ─► events.json
                        ├─ RuleScores (soft-threshold functions)
                        └─ LearnedScores (GBM / TCN) for: off_screen_attention, speaking
```
Both providers emit the **same score interface**, so the state machine is identical regardless of provider. Provider choice is a config flag and an experimental factor.

### 7.2 Event state machine (shared)
Per event type: `IDLE → PENDING → ACTIVE → COOLDOWN`.
- **Onset:** score ≥ `on_thr` continuously for `t_on` (debounce) → ACTIVE; event start is back-dated to the first frame above threshold.
- **Offset (hysteresis):** score < `off_thr` (< `on_thr`) for `t_off` → event ends. Dips shorter than `t_off` do not split the event.
- **Merge:** same-type events separated by < `merge_gap` are merged.
- **Minimum duration** filter; **cooldown** so a continuing event is one record updated in place.
- **Quality gating:** if input is `unreliable`, the score is held/undefined, no new event of that type starts, and duration counters pause (configurable). Unreliable time is never counted as evidence.
- **Live vs final:** the live window shows an event as soon as it is ACTIVE ("ongoing"); `events.json` holds events finalized at offset or at end of stream (open events are closed at stream end).

### 7.3 Specific temporal logic
- **E4:** `off_screen_score` (gaze+head, optionally via the learned model) → state machine, default `t_on=4 s`. Benign patterns (brief glance up while thinking, blink-induced tracking noise, looking at keyboard if `allow_looking_down`) are absorbed by debounce and the learned model's context.
- **E5:** keep a list of off-screen *excursions* (start, end, zone) from the off-screen state (shorter than E4's threshold). Fire when ≥N excursions in a sliding 60 s window and ≥70% share a zone; `details` lists each excursion.
- **E6:** `speaking` probability → window rule (≥3 s active within 10 s) → state machine; suppressed when `allow_reading_aloud`.
- **E1/E2/E3:** presence scores from detectors/landmarker → same state machine with their own `t_on`.
- **E1 vs blindness:** `FACE_ABSENT` requires a healthy frame stream and adequate quality; otherwise emit `MONITORING_DEGRADED`.
- **Cross-event suppression:** during `MONITORING_DEGRADED`, face-derived events (E4–E7) are not emitted.

### 7.4 Learned temporal model (the ML-depth component)
- **Task:** per-time-step classification of `off_screen_attention` and `speaking` from windows of Section 6 features (excluding identity fields).
- **Inputs:** causal window T = 6 s at 10 Hz (60 steps) × ~35 features, standardized; masks for invalid steps. Per-user normalization comes from the calibration baseline (already in Δ features).
- **Models compared (same splits, same features):**
  1. **Baseline A — Rules:** soft thresholds + median filter.
  2. **Baseline B — LightGBM** on engineered window statistics (mean, std, min/max, slope, fraction-above-threshold, zero-crossing rate of mouth energy, excursion count).
  3. **Model C — causal TCN:** 4–5 dilated causal 1D-conv residual blocks, ~30–100k parameters, dropout, per-step sigmoid heads. Class-weighted BCE or focal loss. Augmentation: feature noise, time warp ±10%, random feature dropout, random invalid-step masks, synthetic calibration offset.
- **Causality:** deployed models MUST be causal (no future frames) so live and replay behave identically. Non-causal variants MAY be run as an offline upper bound.
- **Selection:** by validation event-level F1 and false-alarm rate, not frame accuracy; ties → simpler model. Report all three.
- **Output calibration:** temperature/Platt scaling on validation; reliability diagrams. A score is model confidence that the observation holds, never a guilt probability.

---

## 8. Explainability outputs

Replaces the master spec's clips/reports/UI. For every finalized event the pipeline MUST write, inside `events.json`:
1. **`details`** (structured, no free text): the signal that fired, peak/mean value, threshold(s), `t_on`, duration, zone (if any), calibration error, mean quality during the event, calibration mode (`full`/`head_pose_only`), and benign-explanation flags (low quality, glare flag, `allow_looking_down`, recent recalibration).
2. **`attribution`** (learned scorer only): feature-group occlusion — re-score with each group (head pose, eye, mouth, quality) neutralized and report the score drop. (Captum Integrated Gradients MAY be added as P2.)
3. **Provenance header:** `schema_version`, `feature_schema_version`, config hash, model file hashes, thresholds in force.

Example:
```json
{"type":"OFF_SCREEN_SUSTAINED","start_ms":751000,"end_ms":758100,"confidence":0.91,
 "detector":"rule@0.1","details":{"zone":"right","signal":"off_screen_score","peak":0.94,
 "d_yaw_deg":31.0,"yaw_limit_deg":25.0,"t_on_s":4.0,"duration_s":7.1,"calib_error":0.08,
 "quality_mean":0.82,"calib_mode":"full"}}
```

**Visual evidence** (for error analysis and the demo, not a product feature): `replay --render out.mp4` and the live window draw face box, head-pose axes, gaze arrow/zone, object boxes with class+confidence, mouth-activity indicator, quality banner, and active events via OpenCV (`cv2.VideoWriter`, any playable codec — browser compatibility is not required). Signal-vs-threshold plots are made from `features.parquet` in notebooks/eval scripts.

---

## 9. Interfaces, stack, outputs

### 9.1 CLI (the only interface)
- `proctorlens replay <video> --config configs/policy.yaml [--render out.mp4]` → `features.parquet`, `events.json`.
- `proctorlens live --camera 0 --config …` → OpenCV window with overlays and an event ticker (runs calibration first).
- `proctorlens calibrate --camera 0` / `proctorlens extract-features <dir>` (batch replay over a dataset).
- Writes no video or frames unless `--render` (or dataset recording) is used.

### 9.2 Stack
Python 3.11+, NumPy, OpenCV, MediaPipe, ONNX Runtime, Ultralytics, InsightFace (or direct ONNX), PyTorch (TCN training + inference), LightGBM, scikit-learn, pandas/pyarrow, Pydantic v2 + PyYAML (config/types), pytest, Hypothesis, ruff, mypy (core). Dependencies via `uv` (or Poetry) with a pinned lockfile.

### 9.3 Files
Per recording: `features.parquet`, `events.json`, optional rendered video. Labels: `data/labels/<recording>.csv`. No database.

---

## 10. Data strategy

### 10.1 Why self-collection
No public dataset has proctoring-style sessions with our event labels. We build **ProctorLens-Sessions**. Public data is for sanity checks/pretraining support only; verify licenses and never depend on availability:
- Head-pose sanity check: AFLW2000-3D and/or BIWI Kinect Head Pose (report MAE in yaw/pitch/roll) — P1.
- Gaze: MPIIFaceGaze / Gaze360 / ETH-XGaze only for the L2CS-Net ablation — P2.
- Objects: COCO subset + license-checked public phone/notes datasets.
- Face-verification sanity: LFW-style pairs (optional).

### 10.2 Collection protocol (scripted, cue-logged)
**Participants:** target 15–20 (minimum 10) volunteers with informed consent (Section 14), covering glasses/no glasses, facial hair, head coverings, hair styles, skin tones, and ages available. A second person ("assistant") appears in some sessions.

**Recorder:** `tools/collect/record.py` — a small OpenCV script that captures the webcam to video, runs the calibration dot display, shows on-screen cue prompts, and logs cue timestamps to CSV (ground-truth start).

**Session (~12–15 min):**
1. Calibration (Section 5.3) + enrollment frames.
2. **Scripted block** (randomized order, varied durations/directions):
   - look away left/right/up/down for 2 s, 5 s, 8 s; repeated glancing to one side (5–6 times in a minute);
   - leave the frame for 5–15 s; partially leave;
   - second person enters / stands behind / leans in (full and partial);
   - phone held up, on desk in view, at lower frame edge; notes/book held/on desk;
   - whispering/talking to the side; silent lip-reading posture;
   - camera blocked with hand; lights off/dimmed (degradation);
   - identity swap: a different participant sits in for 20–30 s (paired sessions).
3. **Natural block (5 min):** participant answers a short on-screen quiz (any static document/page; not part of the software) — realistic *honest* behavior.
4. **Honest "nuisance" block (hard negatives):** thinking (looking up/away briefly), reading aloud (when allowed), scratching head/face, drinking, stretching, sneezing/yawning, adjusting glasses, chewing gum, leaning close to the screen, looking at the keyboard, small posture shifts, hair touching face, smile/laugh, nodding.
5. **Condition variation:** ≥2 lighting conditions (normal, dim/backlit); ≥2 camera heights/angles across the cohort if possible; glasses on/off for wearers; a few sessions at reduced resolution.

### 10.3 Labels
- Event intervals `(type, start_ms, end_ms, source[cue|annotator], annotator_id)` from the cue log, then **corrected** by annotators in an off-the-shelf tool (ELAN or Label Studio) — cue labels include reaction latency, so boundary correction is required for evaluation sessions.
- Benign behaviors labeled as `benign_*` intervals (for false-alarm analysis), not left unlabeled.
- **Inter-annotator agreement:** two annotators independently label ≥20% of sessions; report frame-level Cohen's κ and interval IoU.
- Detector images sampled from sessions and labeled in CVAT/Label Studio with pre-labels. Identity: pseudonymous participant IDs.

### 10.4 Splits and leakage rules
- **All splits are participant-disjoint.** ~60/20/20 train/val/test; with small N use **GroupKFold (leave-participants-out)** for model selection plus a final untouched test set (nested CV if N < 15).
- Test data is used exactly once for final reporting; no threshold tuning on test.
- The same participant's frames MUST NOT appear in both detector train and test sets.
- `data/manifest.csv` (recording id, participant pseudonym, conditions, split, consent version) is tracked in git; no media.

---

## 11. ML training plan

| Component | Pretrained | Trained / fine-tuned | Notes |
|---|---|---|---|
| Landmarks / head pose / blendshapes | MediaPipe | — | Validate with public-set head-pose MAE (if used) + stress tests |
| Gaze | — (MediaPipe features) | **Per-user ridge/MLP regression at calibration time**; global design choices (feature set, margin) tuned on dataset | Validation-dot error, zone accuracy |
| Object/person detector | COCO YOLO nano | **Fine-tuned** on phone/notes/person | Pretrained-vs-fine-tuned comparison required |
| Identity | InsightFace embedding | Threshold τ calibrated | EER, TAR@FAR |
| Temporal scorers | — | **LightGBM and TCN trained** on our feature series | Compared with rule baseline |
| Quality, state machines | — | Rules; thresholds tuned on validation only | |

Reproducibility: fixed seeds; configs in `configs/training/*.yaml`; every run writes `metrics.json`, `config.yaml`, git hash, data-manifest hash; model cards in `docs/model_cards/` (intended use, data, metrics, failure modes, license).

---

## 12. Evaluation methodology

### 12.1 Component-level
- **Head pose** (if public set used): MAE yaw/pitch/roll.
- **Gaze/calibration:** validation-dot error (normalized units); 5-zone accuracy (on/left/right/up/down) on labeled segments; stratified by glasses and lighting.
- **Detector:** mAP50, mAP50-95, per-class PR curves, precision/recall at the deployed threshold, pretrained vs fine-tuned, CPU latency.
- **Identity:** EER, TAR@FAR=1%, genuine-vs-impostor similarity distributions, chosen τ.
- **Learned temporal scorers:** frame-level PR-AUC and per-class F1 (secondary); reliability diagram.

### 12.2 Event-level (primary)
- **Matching:** predicted event = TP if temporal IoU with a same-type GT event ≥ 0.3 (±1 s tolerance padding for events < 5 s); each GT matches at most one prediction.
- **Per event type:** precision, recall, F1; **onset latency** (median/IQR of predicted start − true start, and time from true onset to emission); **false alarms per hour of honest behavior** (natural + nuisance blocks only).
- **Flagged-time burden:** fraction of recording time covered by predicted events.
- **Uncertainty:** participant-level bootstrap 95% CIs (resample participants, not frames). State N clearly.

### 12.3 Ablations and comparisons (required)
1. Rules vs LightGBM vs TCN (E4/E6).
2. With vs without per-user calibration (head-pose-only vs calibrated gaze).
3. COCO-pretrained vs fine-tuned detector (frame-level and event-level effect).
4. With vs without quality gating (effect on false alarms and on misclassifying blindness as absence).
5. State-machine settings (`t_on`, hysteresis) — sensitivity curves.
6. Processing rate (10 vs 5 fps) — effect on recall of short events.
7. P2: L2CS-Net (or other pretrained gaze CNN) vs our calibrated features.

### 12.4 Robustness stress tests (replay on recorded test sessions)
Synthetic perturbations applied to video before the pipeline: brightness/gamma (dark), Gaussian blur, JPEG quality (30–90), downscaling (480p/360p), additive noise, horizontal flip (sanity), cropped/off-center framing. Plus natural variation (glasses, lighting). Report performance drop per perturbation and the rate at which the system correctly switches to `MONITORING_DEGRADED` instead of emitting false events.

### 12.5 System-level
Per-module and end-to-end latency (frame in → event emitted), CPU/GPU utilization, sustained FPS on the demo laptop, memory over a 60-min replay soak.

### 12.6 Acceptance targets (report honestly even if missed)
On the held-out test set: deterministic events (E1, E2, E3 phone) event recall ≥ 0.85; E4 ≥ 0.75; E5/E6 ≥ 0.60; overall false alarms ≤ 6 per hour of honest behavior (excluding degraded time); median emission latency ≤ `t_on` + 1.5 s; pipeline ≥ 8 fps on the demo laptop. Missing a target is acceptable if analyzed truthfully; over-claiming is not.

---

## 13. Testing and edge cases

### 13.1 Test layers
- **Unit (pytest):** Euler decomposition with synthetic rotations; soft-threshold functions; calibration regression on synthetic data; state machine (onset/offset/hysteresis/merge/min-duration/cooldown/gating/pause-on-unreliable); excursion counter; config validation; time-grid resampling with jitter/gaps; matching-metric implementation (hand-made cases with known answers); TCN causality (output at t unchanged when future inputs change).
- **Property-based (Hypothesis):** state machine never emits overlapping events of one type; `end ≥ start`; merging is idempotent.
- **Golden / integration:** short fixture videos (seconds; synthetic overlays or consented clips) with expected events; replay must reproduce identical `events.json` (determinism test, float tolerance).
- **Data-handling:** replay without `--render` writes only `features.parquet` and `events.json`; identity embeddings are not persisted by default.
- **Performance:** benchmark script for per-module and end-to-end latency; 60-min replay soak for memory leaks.

### 13.2 Edge-case catalog (each has a defined behavior, ideally a test)
Camera absent or unplugged mid-run (→ `MONITORING_DEGRADED`); very low FPS or frame bursts; variable-frame-rate video; duplicate/out-of-order timestamps; glasses glare hiding irises; sunglasses/masks/hijab/hair covering eyes or mouth; heavy backlight; very dark room; camera obstructed; face too small/large/partially outside frame; face lost due to extreme head turn (→ `TURNED_AWAY`); poster/photo/TV face in background (static-face suppression); reflection in window; pets; person walking past far behind (size + persistence gating); child/partner briefly entering; candidate with tics, dyskinesia, or atypical eye movement (config adjustability); partial hand occlusion while thinking; yawning/laughing/chewing vs speaking; phone-shaped objects (remote, wallet) → hard negatives; identity under lighting change; two faces swapping positions; calibration failure (→ head-pose-only mode, flagged); laptop/seat change (→ calibration drift); external monitor with off-center camera; very long recordings (memory bounds); simultaneous events; stream ending with open events (finalize).

---

## 14. Data ethics, fairness, limitations

**Requirements**
- Written **informed consent** from every dataset participant (research use, what is recorded, retention, right to withdraw/delete). Forms stored outside git. Check with the course instructor/ethics requirements before recording.
- **Minimization / local-only:** raw video exists only for consented dataset recordings; live mode writes nothing unless asked; no audio recorded; no third-party upload; dataset on an encrypted disk or restricted folder; pseudonymized participant IDs; access limited to the team.
- **Identity embeddings** are held in memory and discarded by default.
- **Fairness disclosure:** face tracking/recognition performance can vary with skin tone, lighting, facial features, head coverings, disabilities, and camera quality. Evaluation MUST report performance by available conditions (glasses, lighting, …), and the final report MUST include a fairness/limitations section stating where the sample is too small to conclude. Config toggles let thresholds be relaxed.
- Atypical gaze/movement is never treated as inherently suspicious.

**Known limitations to state in the final report:** webcam-only view cannot see hands/lap/second screens; no liveness detection (a photo/video replay could fool identity and presence); gaze accuracy is coarse; speaking detection is visual-only; results come from a small, non-representative participant pool; scripted behavior is not real cheating behavior.

---

## 15. Repository structure

```
proctorlens/
├── README.md                    # setup, run, reproduce
├── Makefile                     # make setup | test | replay | eval | train-detector | train-temporal
├── pyproject.toml / uv.lock
├── docs/
│   ├── SPEC.md                  # this file
│   ├── DECISIONS.md             # deviations + rationale
│   ├── DATA_PROTOCOL.md         # collection script, consent, labeling guide
│   ├── EVALUATION.md            # protocol + final results
│   └── model_cards/
├── configs/
│   ├── pipeline.yaml            # rates, model paths
│   ├── policy.yaml              # event thresholds and toggles (Section 3)
│   └── training/*.yaml
├── src/proctorlens/
│   ├── core/        # types (Pydantic), config, clock/time-grid
│   ├── perception/  # quality.py, landmarks.py, head_pose.py, gaze.py, objects.py, identity.py
│   ├── features/    # extractor.py, schema.py, windows.py
│   ├── temporal/    # scores_rule.py, scores_learned.py, state_machine.py, excursions.py
│   ├── explain/     # overlays.py, attribution.py
│   ├── pipeline/    # runner.py (live + replay shared), calibration.py
│   └── cli.py       # replay | live | calibrate | extract-features
├── ml/
│   ├── data/        # build_manifest.py, extract_frames.py, to_yolo_format.py, loaders
│   ├── training/    # train_detector.py, train_temporal_gbm.py, train_temporal_tcn.py, calibrate_identity.py
│   ├── evaluation/  # matching.py, metrics.py, run_eval.py, ablations.py, robustness.py
│   └── notebooks/   # EDA, error analysis
├── tools/collect/   # record.py (cue recorder + calibration dots)
├── tests/           # unit/, integration/, fixtures/
└── data/            # gitignored: raw/, recordings/, labels/, processed/, models/ ; manifest.csv tracked
```
Tooling: ruff + mypy (core), conventional commits, semantic versioning of model artifacts (`models/<name>/<version>/{weights, model_card.md, hash}`).

---

## 16. Implementation phases

| Phase | Weeks | Goals | Exit criteria |
|---|---|---|---|
| **0. Foundation** | 1 | Repo, env, config system, types, time-grid clock | `make test` green; empty pipeline runs on a video file |
| **1. Perception core (offline)** | 1–3 | Quality, MediaPipe landmarks, head pose, eye/mouth features, pretrained-YOLO wrapper, feature schema, `replay` CLI → `features.parquet` | Plausible head-pose/gaze traces on a sample video; geometry unit tests pass |
| **2. Data tooling + collection** *(starts week 2, parallel)* | 2–7 | `record.py`, consent forms, annotation guide, pilot with 2–3 people, then full collection | ≥10 participants by week 6 (target 15–20); labeled intervals + IAA on a subset; `manifest.csv` |
| **3. Calibration + rule-based events** | 3–5 | Calibration module, off-screen scores, state machines E1–E5, E8, overlay renderer, `live` window | Replay produces `events.json` with `details`; state-machine tests pass; first numbers on pilot data; live window shows ≥3 event types |
| **4. ML training** | 4–7 | Detector fine-tune; LightGBM + TCN temporal scorers; identity τ calibration; mouth/speaking model; attribution | Model cards; comparison tables; models behind config flags |
| **5. Evaluation** | 7–9 | Final untouched-test evaluation, ablations, robustness, latency/soak, error analysis (top FP/FN with rendered clips) | `docs/EVALUATION.md` with tables, plots, CIs, failure analysis |
| **6. Polish + demo** | 9–10 | Demo script, backup replay demo, final report, slides, README, cleanup | Section 17 deliverables complete |

**Sequencing rules:** (1) vertical slice first; (2) never block on data collection — use pilot data and scripted fixtures; (3) freeze the feature schema early; (4) keep the replay path green at all times.

---

## 17. Deliverables and demo

**Demo (~7 min):**
1. 30 s: framing + principle ("observable events, not verdicts").
2. Live: calibration (show accuracy score), then normal behavior (no events; honest nuisance behaviors don't trigger) → glance repeatedly right → hold up phone → second person enters → leave frame → cover camera / dim lights (→ `MONITORING_DEGRADED`, not `FACE_ABSENT`).
3. Show the rendered overlay video and `events.json` details/attribution for one event.
4. 2–3 min of evaluation highlights: rules vs learned, pretrained vs fine-tuned, false alarms/hour, robustness, limitations.
5. **Backup:** replay mode plays a pre-recorded session through the same pipeline if camera or hardware fails.

**Final deliverables:** working package runnable with documented commands; trained model artifacts + model cards; dataset manifest and (internally stored) dataset with protocol and labeling guide; evaluation report (tables, plots, CIs, ablations, error analysis); test suite; 5–8 minute demo video; slides; final written report including ethics/fairness/limitations; README with reproduction steps (`make eval` reproduces reported numbers from stored features/models).

---

## 18. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Data collection slips | No evaluation | Start week 2; pilot early; recruit in parallel; minimum N=10; recorder logs cue timestamps automatically |
| Small N overfits learned models | Weak TCN | GBM baseline; GroupKFold; augmentation; "rules win" is a valid finding |
| MediaPipe degrades (glasses, dark, pose) | False events | Quality gating, `MONITORING_DEGRADED`, head-pose-only fallback, robustness report |
| Gaze calibration unreliable | Weak E4 | Head-pose geometric term, acceptance test, retry, zone-level claims only |
| Detector false positives (remote, wallet) | Noisy E3 | Hard-negative images, persistence windows, per-class thresholds |
| CPU too slow | Laggy live demo | Staggered YOLO, ONNX/OpenVINO, lower resolution, GPU demo machine, replay backup |
| Scope creep (UI, audio, liveness, VLM) | Incomplete core | Non-goals list; P2 only after P0/P1 done |
| Privacy/ethics issues | Blocked project | Consent, minimization, local storage, early instructor approval |
| Library API drift | Build breaks | Pin versions; Section 19 verification; adapter layers around MediaPipe/YOLO/InsightFace |

---

## 19. Assumptions the implementing agent must verify before coding

1. **MediaPipe Tasks Face Landmarker** (Python): video-mode API, `output_face_blendshapes`, `output_facial_transformation_matrixes`, `num_faces`, model asset URL, Python-version support, and the actual Euler convention of the transformation matrix.
2. **Ultralytics:** latest nano model name (YOLO11n or newer), training/export API, ONNX/OpenVINO export, AGPL-3.0 implications for the README.
3. **InsightFace:** how to obtain the pretrained recognition model (`buffalo_sc` or equivalent), ONNX preprocessing, non-commercial license terms; fall back to another usable embedding model if unavailable.
4. **Public datasets** (AFLW2000, BIWI, Roboflow sets): availability, license, usage terms.
If any assumption fails, choose the closest alternative, record it in `docs/DECISIONS.md`, and keep the `perception/*` adapter interfaces stable.

---

## 20. Definition of done

Done when: (1) `replay` and `live` run the same pipeline end to end, producing `features.parquet` and `events.json`; (2) events E1–E5 and E8 (P0) work live and offline, E6–E7 (P1) work or are documented as cut with reasons; (3) every event carries `details` (and `attribution` when the learned scorer is active); (4) the rule-vs-learned, pretrained-vs-fine-tuned, calibration, and quality-gating experiments are run on participant-disjoint held-out data with CIs; (5) robustness, latency, and false-alarms-per-hour numbers are reported; (6) tests pass; (7) ethics, fairness, and limitations are documented; (8) nothing in code, outputs, or docs claims to determine that someone cheated; (9) nothing outside the ML/CV scope rule (Section 0) exists in the repo.

---

## Appendix A — Scope delta vs. `PROCTORLENS_MASTER_SPEC.md`

| Removed | Why / replacement |
|---|---|
| React/TS frontend, candidate flow (consent, environment check, enrollment, mock exam, finish), proctor dashboard, Playwright E2E | Not ML/CV. Replaced by CLI + OpenCV live window; calibration dots drawn by an OpenCV window. |
| FastAPI, REST, WebSockets, auth/JWT, SessionManager, concurrency | Not ML/CV. Replaced by a file-in/file-out pipeline; one recording at a time. |
| SQLite/SQLModel, retention/delete endpoints | Not ML/CV. Replaced by Parquet + JSON files and `data/manifest.csv`. |
| E8 `BROWSER_INTEGRITY` (tab/blur/fullscreen/paste telemetry) | Not CV. Old E9 `MONITORING_DEGRADED` is now E8. |
| Reviewer decisions, review workflow, HTML/PDF session report | Workflow, not ML/CV. |
| Segment merging, review-priority scoring, recall@K ranking evaluation | Hand-weighted product logic, not ML/CV. Replaced by flagged-time burden metric. |
| Evidence ring buffer, H.264 clip encoding, keyframe thumbnails, NL explanation templates | Product UX. Replaced by structured `details`, `attribution`, and OpenCV overlay rendering. |
| `severity` event field, exam-session policy snapshots | Product. Kept: config profile with thresholds and toggles. |
| Audio/VAD (P2), browser-API assumptions, Chromium fake-camera flags, H.264/PyAV assumptions | Out of scope / no longer needed. |
| Custom cue web app and annotation tool | Replaced by `tools/collect/record.py` and off-the-shelf ELAN/Label Studio/CVAT. |
| CI pipeline, Docker, pre-commit | Not ML/CV; optional. |
