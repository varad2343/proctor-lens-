# Model card: <name> <version>

Copy to `docs/model_cards/<name>.md`, fill in every field, delete hints. One card per shipped artifact (detector weights, each temporal scorer, identity threshold). Never write "n/a" without a reason.

## 1. Model details

| Field | Value |
|---|---|
| Name / version | |
| Kind | detector (YOLO) / temporal scorer (gbm, tcn) / identity threshold / gaze regression |
| Target | e.g. phone, notes, person / off_screen / speaking |
| Artifact path | e.g. `data/models/temporal/<name>/<version>/` |
| Artifact hash (SHA-256) | |
| Base model and its license | e.g. COCO-pretrained YOLO nano, AGPL-3.0; MediaPipe asset; InsightFace pretrained (non-commercial research) |
| Trained on (date, machine, seed) | |
| Code: git commit, training config (`configs/training/*.yaml`), `metrics.json` | |
| `feature_schema_version` / `config_hash` it was evaluated with | |
| Input and output | features and window length / image size; output is a score in [0, 1] = model confidence that the observation holds |

## 2. Intended use

- **For:** producing time-stamped, explainable, observable events from one webcam, to be looked at by a person.
- **Not for:** any automatic decision about a person. The score is a confidence that an observation is real, not a judgment of intent. Do not use outside the conditions in section 4.
- Policy toggles that change behavior (`allow_notes`, `allow_looking_down`, `allow_reading_aloud`) and their defaults:

## 3. Training and validation data

| Item | Value |
|---|---|
| Source | ProctorLens-Sessions (+ public sets, with license) |
| Participants / recordings / hours (train, val) | |
| Labels and how they were made | `docs/DATA_PROTOCOL.md`; inter-annotator agreement (kappa, IoU) |
| Split rule | participant-disjoint; list the `manifest.csv` hash |
| Conditions covered / missing | lighting, glasses, head coverings, camera heights, skin tones, ages |
| Pre-processing, augmentation, class weighting | |

## 4. Evaluation (test split, used once)

Fill from `docs/EVALUATION.md`; state N participants for every number and include participant-level bootstrap 95% CIs.

| Metric | Rules baseline | This model | Comparison note |
|---|---|---|---|
| Frame-level PR-AUC / F1 (if learned scorer) | | | |
| Event precision / recall / F1 | | | |
| False alarms per honest hour | | | |
| Onset / emission latency | | | |
| Calibration (Platt a, b; ECE; reliability diagram) | | | |
| CPU latency (ms per frame or per window) | | | |

By condition (glasses, lighting, ...): link the table in `docs/EVALUATION.md` section 4 and state where N is too small to conclude.

## 5. Failure modes and limitations

List observed failures with examples (recording id, time range): e.g. glare on glasses, dim or backlit scenes, extreme head pose, partial occlusion, look-alike objects (remote, wallet), posters or screens showing faces, short events near the threshold. State what the model cannot see (hands, lap, second screens) and what happens when quality is low (`MONITORING_DEGRADED`, scores `None`).

## 6. Fairness

Where performance differs across recorded conditions, give the numbers; where the sample is too small to tell, say so. Atypical gaze, movement or speech patterns are not treated as inherently notable; say which toggles or thresholds can be relaxed and how.

## 7. Ethics and data handling

Consent version(s), retention, where the data lives, what is not stored (audio, identity embeddings).

## 8. License and provenance

License of the weights and of every upstream model or dataset used, and what that implies (for example AGPL-3.0 for Ultralytics, non-commercial research terms for the InsightFace pretrained models).

## 9. Reproduce

Exact commands, in order, from a clean checkout to this artifact and to the numbers in section 4.
