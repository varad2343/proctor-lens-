# Evaluation protocol and results

Protocol from SPEC Section 12, with the tables the final report fills in. **No results exist yet:** every table cell below is empty on purpose. Do not fill a cell with a number that was not produced by the commands in section 1 on the split named in the table.

Conventions for every table: `N` = number of participants (and recordings) behind it, stated in the caption; `[lo, hi]` = participant-level bootstrap 95% CI (resample participants, not frames, 1,000 draws, seed 0); a dash means "not run". If a target is missed, say so and analyze it (SPEC 12.6); over-claiming is not acceptable.

## 0. Run header (fill once per reported run)

| Field | Value |
|---|---|
| Date | |
| Git commit | |
| `config_hash` | |
| `feature_schema_version` | |
| Manifest hash | |
| Test split: N participants / recordings / hours | |
| Honest-behavior hours (natural + nuisance blocks) | |
| Demo laptop (CPU, RAM, OS) | |

## 1. Protocol

**Splits.** Participant-disjoint (`data/manifest.csv`, `docs/DATA_PROTOCOL.md` section 7). Model selection and all threshold tuning use train/val only (GroupKFold when N is small). The test split is evaluated **once**, at the end, with frozen thresholds and models.

**Reproduce.** (Commands are the intended entry points; check `--help` of each script for the exact arguments.)

```
proctorlens extract-features data/recordings --config configs/pipeline.yaml --config configs/policy.yaml
python -m ml.evaluation.run_eval          # event-level tables from stored features + labels
python -m ml.evaluation.ablations         # section 7
python -m ml.evaluation.robustness        # section 8
```

**Ground truth.** `data/labels/<recording>.csv` (`type,start_ms,end_ms,source,annotator_id`), annotator-corrected boundaries only. Benign intervals (`benign_*`) are not scored as events; they define the "honest behavior" time used for false alarms per hour.

**Event matching** (`ml/evaluation/matching.py`, `match_events`). A predicted event is a true positive if its temporal IoU with a **same-type** ground-truth event is >= 0.3. Ground-truth events shorter than 5 s are padded by 1 s on each side before the IoU. Greedy by IoU; each ground-truth event matches at most one prediction. Unmatched predictions are false positives, unmatched ground truth are false negatives.

**Metrics** (`ml/evaluation/metrics.py`).

| Metric | Definition |
|---|---|
| Precision / recall / F1 | per event type, from the matching above |
| Onset latency | predicted start minus true start (median, IQR) for true positives; and emission latency = time from true onset to the event becoming ACTIVE |
| False alarms per hour | predicted events with no match, per hour of honest behavior, excluding time under `MONITORING_DEGRADED` |
| Flagged-time fraction | share of recording time covered by predicted events |
| CI | participant-level bootstrap, 95% |

Frame-level metrics (PR-AUC, F1) are secondary and only used for the learned scorers.

## 2. Component level

### 2.1 Head pose (only if AFLW2000-3D / BIWI is used; P1)

| Dataset | N images | MAE yaw (deg) | MAE pitch (deg) | MAE roll (deg) |
|---|---|---|---|---|
| | | | | |

### 2.2 Gaze and calibration

| Stratum | N participants | Median validation-dot error [lo, hi] | Accepted at first try (%) | Head-pose-only fallback (%) | 5-zone accuracy |
|---|---|---|---|---|---|
| All | | | | | |
| Glasses | | | | | |
| No glasses | | | | | |
| Normal light | | | | | |
| Dim / backlit | | | | | |

### 2.3 Object and person detector (test split)

| Model | Class | AP50 | AP50-95 | Precision @ deployed thr | Recall @ deployed thr | CPU latency (ms) |
|---|---|---|---|---|---|---|
| COCO-pretrained | phone (`cell phone`) | | | | | |
| COCO-pretrained | notes (`book`) | | | | | |
| COCO-pretrained | person | | | | | |
| Fine-tuned | phone | | | | | |
| Fine-tuned | notes | | | | | |
| Fine-tuned | person | | | | | |

### 2.4 Identity (P1)

| N genuine pairs | N impostor pairs | EER | TAR @ FAR = 1% | Chosen tau |
|---|---|---|---|---|
| | | | | |

Also plot genuine vs impostor similarity distributions and save the figure next to this file.

### 2.5 Learned temporal scorers (frame level, secondary)

| Target | Scorer | PR-AUC | F1 @ chosen thr | Calibration (ECE) |
|---|---|---|---|---|
| off_screen | rules | | | |
| off_screen | LightGBM | | | |
| off_screen | TCN | | | |
| speaking | rules | | | |
| speaking | LightGBM | | | |
| speaking | TCN | | | |

Reliability diagrams for each learned scorer go next to this file.

## 3. Event level (primary)

### 3.1 Per event type, shipped configuration (test split)

| Event type | GT events | Pred events | TP | FP | FN | Precision [lo, hi] | Recall [lo, hi] | F1 [lo, hi] | Onset latency median (IQR) s | Emission latency median s |
|---|---|---|---|---|---|---|---|---|---|---|
| FACE_ABSENT | | | | | | | | | | |
| MULTIPLE_PEOPLE | | | | | | | | | | |
| PROHIBITED_OBJECT | | | | | | | | | | |
| OFF_SCREEN_SUSTAINED | | | | | | | | | | |
| REPEATED_GLANCING | | | | | | | | | | |
| MOUTH_ACTIVITY | | | | | | | | | | |
| IDENTITY_MISMATCH | | | | | | | | | | |
| MONITORING_DEGRADED | | | | | | | | | | |

### 3.2 Burden

| False alarms per honest hour [lo, hi] | Flagged-time fraction [lo, hi] | Honest hours |
|---|---|---|
| | | |

### 3.3 Acceptance targets (SPEC 12.6)

| Target | Required | Measured | Met? |
|---|---|---|---|
| Event recall: FACE_ABSENT, MULTIPLE_PEOPLE, PROHIBITED_OBJECT (phone) | >= 0.85 | | |
| Event recall: OFF_SCREEN_SUSTAINED | >= 0.75 | | |
| Event recall: REPEATED_GLANCING, MOUTH_ACTIVITY | >= 0.60 | | |
| False alarms per honest hour (excluding degraded time) | <= 6 | | |
| Median emission latency | <= t_on + 1.5 s | | |
| Pipeline speed on the demo laptop | >= 8 fps | | |

## 4. Fairness and conditions

Event-level F1 and false alarms per honest hour by recorded condition (from `manifest.csv` `conditions`). State where N is too small to conclude.

| Condition | N participants | N recordings | F1 (macro over types) | False alarms / honest hour | Degraded-time fraction | Comment |
|---|---|---|---|---|---|---|
| glasses | | | | | | |
| no_glasses | | | | | | |
| light_normal | | | | | | |
| light_dim / backlit | | | | | | |
| head_covering | | | | | | |
| cam_low / cam_high | | | | | | |
| res_reduced | | | | | | |

## 5. Ablations (required, SPEC 12.3)

Same splits, same features unless stated.

| # | Comparison | Metric | Variant A | Variant B | Variant C |
|---|---|---|---|---|---|
| 1 | Rules / LightGBM / TCN (E4, E6) | event F1, false alarms per honest hour | | | |
| 2 | Head-pose-only / calibrated gaze | OFF_SCREEN_SUSTAINED F1, false alarms per honest hour | | | n/a |
| 3 | COCO-pretrained / fine-tuned detector | phone and notes: frame AP50 and event F1 | | | n/a |
| 4 | Without / with quality gating | false alarms per honest hour; blindness reported as FACE_ABSENT (count) | | | n/a |
| 5 | State-machine settings (`t_on`, hysteresis) | sensitivity curve: F1 vs `t_on` | see figure | | |
| 6 | Processing rate 10 / 5 fps | recall of events under 5 s | | | n/a |
| 7 | (P2) pretrained gaze CNN / calibrated features | 5-zone accuracy | | | n/a |

## 6. Robustness (SPEC 12.4)

Perturbations are applied to the test videos before the pipeline runs. "Correct degraded switch" = fraction of perturbed seconds where `MONITORING_DEGRADED` is active instead of a face-derived event.

| Perturbation | Setting | Event F1 (macro) | Drop vs clean | False alarms / honest hour | Correct degraded switch (%) |
|---|---|---|---|---|---|
| none (clean) | | | 0 | | |
| Brightness / gamma (dark) | | | | | |
| Gaussian blur | | | | | |
| JPEG quality | 90 / 60 / 30 | | | | |
| Downscale | 480p / 360p | | | | |
| Additive noise | | | | | |
| Horizontal flip (sanity) | | | | | |
| Cropped / off-center framing | | | | | |

## 7. System level (SPEC 12.5)

| Measure | Value |
|---|---|
| Per-module latency (ms): quality / landmarks / detector / identity / features / scorer / state machines | |
| End-to-end latency, frame in to event emitted (ms) | |
| Sustained FPS on the demo laptop | |
| CPU / GPU utilization | |
| Memory over a 60-min replay soak (start, end) | |
| Determinism: two replays of the same video give the same `events.json` | |

## 8. Error analysis

Top false positives and false negatives per event type, each with recording id, time range, rendered clip (`replay --render`), and a one-line observed cause (quality reason, occlusion, pose, lighting).

| Type | FP / FN | Recording | Start-end (ms) | Observed cause | Clip |
|---|---|---|---|---|---|
| | | | | | |

## 9. Limitations (keep honest; edit with the results)

State at least: webcam-only view cannot see hands, lap or second screens; no liveness detection; gaze is coarse and only zone-level claims are made; speaking detection is visual-only; a small, non-representative participant pool, so conditions with few participants support no conclusion; scripted behavior is not real-world behavior; the system reports observable events and makes no judgment about intent.
