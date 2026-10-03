# Data protocol: ProctorLens-Sessions

Distilled from SPEC Sections 10 and 14. Read together with `docs/SPEC.md`; if they differ, SPEC wins and this file gets fixed.
Scope: collecting, labeling and splitting our own recordings. Nothing here is a product feature.

Targets: **15-20 participants (minimum 10)**, 12-15 min per session, participant-disjoint splits, >= 20% of sessions double-labeled.

## 1. Before anyone is recorded

Check with the course instructor / ethics requirements first (SPEC 14). Then, per participant, **all** of these:

- [ ] Adult (18+) volunteer; not a dependent of the team or instructor.
- [ ] Read and signed the consent form **before** recording. Forms live outside git (encrypted disk or restricted folder, never in `data/`).
- [ ] Told in plain words: research use for a course project; what is recorded (video of them at a computer; **no audio**); how long it is kept and where (local, access limited to the team); that the clips are acted behavior, not real misconduct; that the software reports observable events, not verdicts; that accuracy can vary with lighting, glasses, skin tone, head coverings and camera.
- [ ] Told they can stop at any time, and can withdraw and have their data deleted afterwards, with no penalty. Given a contact for that.
- [ ] Separate optional tick-box (default **no**) for showing their face in the report or demo.
- [ ] Assistants / second persons and any identity-swap partner have signed consent too.
- [ ] Assigned a pseudonymous ID (`P01`, `P02`, ...). The name-to-ID table is stored with the consent forms, never next to the media.
- [ ] Consent form version noted (goes in `manifest.csv` as `consent_version`).

Withdrawal: delete that participant's recordings, calibration logs, cue logs, labels, extracted features, sampled detector images and manifest rows. Do not reshuffle the other participants' splits; retrain if the participant was in a training split.

Storage rules: raw media only under `data/recordings/` on an encrypted disk or restricted folder; `data/` is git-ignored except `manifest.csv`; no third-party upload (this includes cloud annotation tools, unless the instructor approves and the consent form says so); live mode and `replay` write no frames.

## 2. Recording a session

Tool: `python tools/collect/record.py` (see `--help`). It records the webcam to video **without audio**, shows the calibration dots, shows the cue prompts, and logs every cue timestamp to CSV.

Order (randomize the order inside blocks 2 and 4):

1. **Calibration + enrollment (~30 s).** Look at screen center 2 s (baseline head pose), follow the dot through the 3x3 grid (1.5 s per dot, first 0.5 s discarded) and 4 random validation dots. Face the camera frontally for the first 5 s so identity enrollment has frames. The dot log is saved as `<recording_id>.calib_log.csv`.
2. **Scripted block.** Cues from section 3. Include, with varied durations and directions:
   - looking away left / right / up / down for 2 s, 5 s, 8 s; repeated glancing to one side 5-6 times in a minute;
   - leaving the frame for 5-15 s; partially leaving;
   - a second person entering, standing behind, leaning in (full and partial);
   - phone held up, phone on the desk in view, phone at the lower frame edge; notes or a book held, on the desk;
   - whispering / talking to the side; silent lip-reading posture;
   - camera blocked with a hand; lights off or dimmed;
   - identity swap: a different participant sits in for 20-30 s (paired sessions only).
3. **Natural block (5 min).** The participant answers a short on-screen quiz (any static page; it is not part of the software). This is realistic honest behavior and feeds the false-alarms-per-hour number.
4. **Nuisance block (hard negatives).** Thinking (looking up or away briefly), reading aloud (when the policy allows), scratching head or face, drinking, stretching, sneezing or yawning, adjusting glasses, chewing gum, leaning close to the screen, looking at the keyboard, small posture shifts, hair touching the face, smiling or laughing, nodding.
5. **Condition variation across the cohort:** at least two lighting conditions (normal, dim/backlit); at least two camera heights or angles if possible; glasses on/off for wearers; a few sessions at reduced resolution. Cover glasses, facial hair, head coverings, hair styles, skin tones and ages that are available.

Afterwards: open the video once to check it plays and has a sensible length, then add the `manifest.csv` row.

## 3. Cue script and cue log

The recorder reads a cue script and writes a cue log. Both are plain CSV.

**Cue script** (input, one row per cue; the recorder shuffles rows within a block):

```
block,type,zone,duration_s,prompt
scripted,OFF_SCREEN_SUSTAINED,left,5,Look to the left of the screen until the prompt clears
scripted,REPEATED_GLANCING,right,60,Glance to the right 5-6 times during the next minute
scripted,FACE_ABSENT,,10,Leave the frame
nuisance,benign_thinking,,4,Look up briefly as if thinking
```

- `block`: `scripted` | `natural` | `nuisance`.
- `type`: an event type (`core.types.EVENT_TYPES`) or a `benign_*` label (section 5).
- `zone`: `left|right|up|down` for gaze cues, empty otherwise.
- `duration_s`: how long the prompt is shown. `prompt` is the on-screen instruction (neutral wording; it describes an action, not a motive).

**Cue log** (output, `<recording_id>.cues.csv`, one row per shown cue):

```
cue_id,block,type,zone,t_cue_ms,t_end_ms
```

`t_cue_ms` / `t_end_ms` are on the video timeline: milliseconds from the first recorded frame, the same clock `replay` uses as `t_ms`. They are **when the prompt was shown**, so they include the participant's reaction delay. That is why they are only a starting point (section 4).

## 4. Labels

### 4.1 File and schema

One CSV per recording: `data/labels/<recording_id>.csv`, columns exactly

```
type,start_ms,end_ms,source,annotator_id
```

| Column | Rule |
|---|---|
| `type` | one of the eight `EVENT_TYPES` (`FACE_ABSENT`, `MULTIPLE_PEOPLE`, `PROHIBITED_OBJECT`, `OFF_SCREEN_SUSTAINED`, `REPEATED_GLANCING`, `MOUTH_ACTIVITY`, `IDENTITY_MISMATCH`, `MONITORING_DEGRADED`) or a `benign_*` label |
| `start_ms`, `end_ms` | integers on the video timeline, `start_ms < end_ms` |
| `source` | `cue` (straight from the cue log) or `annotator` (a person checked or set the boundaries) |
| `annotator_id` | pseudonymous (`A1`, `A2`, ...); `recorder` for `source=cue` |

Rules: intervals of different types may overlap (simultaneous events); intervals of the same type must not overlap (merge them); every second of a session is either labeled or honest background; benign behaviors are labeled, never left blank.

### 4.2 Workflow

1. Convert the cue log to label rows with `source=cue`.
2. An annotator opens the video in ELAN or Label Studio (used locally), and **corrects every boundary** to what is actually visible. Cue times include reaction latency, so evaluation sessions must have `source=annotator` boundaries. Add anything that happened but was not cued (including unplanned degraded intervals).
3. Export to the CSV schema above.

### 4.3 Definitions (label what is observable, not why)

Start = the first frame where the condition is observable; end = the last frame where it still is.

| Type | Label when | Notes |
|---|---|---|
| `FACE_ABSENT` | no face and no person in view for >= 3 s | not while `MONITORING_DEGRADED` applies |
| `MULTIPLE_PEOPLE` | >= 2 distinct live faces or persons visible for >= 2 s | posters, photos and screens showing faces do not count |
| `PROHIBITED_OBJECT` | phone visible >= 1.5 s, or notes / book visible >= 3 s | use `benign_notes_allowed` when the policy allows notes |
| `OFF_SCREEN_SUSTAINED` | attention clearly off the screen, continuously >= 4 s | head turn or eyes away; the direction comes from the cue (`zone`), not from this CSV |
| `REPEATED_GLANCING` | >= 4 short off-screen looks (0.3-3 s) within 60 s, >= 70% toward one side | one interval from the start of the first look to the end of the last |
| `MOUTH_ACTIVITY` | rhythmic, small-amplitude mouth movement consistent with speaking, >= 3 s within 10 s | yawns, smiles, chewing are benign |
| `IDENTITY_MISMATCH` | a different person is at the screen | |
| `MONITORING_DEGRADED` | dark, blurred, blocked, face too small or cut off, or the stream stalls | |

Benign vocabulary (extend freely; the only rule is the `benign_` prefix): `benign_brief_away` (off-screen look shorter than 4 s that is not part of a glancing episode), `benign_thinking`, `benign_reading_aloud`, `benign_scratching`, `benign_drinking`, `benign_stretching`, `benign_yawn_sneeze`, `benign_adjust_glasses`, `benign_chewing`, `benign_lean_close`, `benign_look_keyboard`, `benign_posture_shift`, `benign_hair_touch`, `benign_smile_laugh`, `benign_nod`, `benign_notes_allowed`.

If a definition has to change, decide it before the pilot ends and record it in `docs/DECISIONS.md` (ADR-012). Changing it later means relabeling everything.

### 4.4 Inter-annotator agreement (IAA)

1. Pick >= 20% of the sessions (random with a fixed seed, stratified by lighting and glasses so each appears); list them in `data/labels/iaa_sessions.txt`.
2. Two annotators label those sessions **independently**, from the video only (no cue log, not looking at each other's files). Annotator IDs `A1`, `A2`.
3. Report, per type, **before** any reconciliation:
   - frame-level Cohen's kappa on the 10 Hz grid (`ml.data.labels.frame_labels`),
   - interval IoU (`ml.evaluation.matching.match_events`, one annotator as prediction, the other as ground truth).
4. For any type with kappa < 0.6, tighten the definition in section 4.3, relabel that type, and repeat.
5. Reconcile disagreements together; the reconciled rows are the ground truth used for evaluation. Report the pre-reconciliation numbers in `docs/EVALUATION.md`.

## 5. Detector images

Sample frames from sessions (phone held / on desk / at frame edge, notes held / on desk, second person partial / full, plus hard negatives: remote, wallet, calculator). Pre-label with the COCO model, correct in CVAT or Label Studio run locally. Target about 1,500-3,000 labeled images, **split by participant** with the same train/val/test assignment as the sessions. Add a COCO subset (phones, books, persons) against forgetting, and license-check any public dataset before using it.

## 6. Naming and layout

`recording_id = <participant>_<session>`, e.g. `P07_s01`. Only ASCII, no real names.

```
data/
  manifest.csv                      tracked in git; everything else under data/ is not
  recordings/P07_s01.mp4
  recordings/P07_s01.calib_log.csv  calibration dot schedule + timestamps
  recordings/P07_s01.cues.csv       cue log
  recordings/P07_s01.calib.json     fitted calibration (read by extract-features as <stem>.calib.json)
  labels/P07_s01.csv
  processed/P07_s01/                features.parquet, events.json
  models/                           face_landmarker.task, detector weights, buffalo_sc/, temporal/<name>/<version>/
```

## 7. Manifest and splits

`data/manifest.csv` (tracked; no media): `recording_id,participant,split,conditions,consent_version`.

- `split`: `train`, `val` or `test`, assigned **per participant**; every recording of a participant has the same split. About 60/20/20. With fewer than 15 participants use GroupKFold (leave-participants-out) for model selection plus a final untouched test set (nested CV if N < 15).
- `conditions`: semicolon-separated tags, e.g. `glasses;light_dim;cam_low;res_reduced`. Suggested vocabulary: `glasses`/`no_glasses`, `light_normal`/`light_dim`/`light_backlit`, `cam_low`/`cam_eye`/`cam_high`, `res_full`/`res_reduced`, `head_covering`, `facial_hair`. Evaluation reports results by these tags.
- `consent_version`: the consent form version the participant signed.

Leakage rules:
- The test split is used **once**, for final reporting. No threshold or model-selection decisions on it.
- Frames of one participant never appear in both detector-train and detector-test images.
- Identity tau is calibrated on train/val participants only.
