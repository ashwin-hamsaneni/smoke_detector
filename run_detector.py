"""
Industrial Early-Stage Smoke Detection Monitor (v3 - Stability Edition)
=========================================================================
Fixes two concrete failure modes reported in the field:

  A) "Smoke still shown as present after it visually cleared."
     Root cause: v1/v2 alarms were one-shot triggers with no un-trigger logic --
     once fired, nothing ever said "this is over." v3 adds HYSTERESIS:
       - a track must cross ALARM_SCORE_THRESHOLD (using its full window) to
         first raise an alarm, same as before.
       - once alarmed, it only STAYS alarmed if a smaller, RECENT sub-window
         of observations keeps scoring above a lower SUSTAIN_SCORE_THRESHOLD.
       - if no qualifying evidence arrives for CLEAR_AFTER_FRAMES_NO_QUALIFYING
         real video frames, the system explicitly prints "ALARM CLEARED" and
         the canonical id becomes eligible to alarm again if smoke returns.

  B) "A person was identified as smoke for a second."
     Root cause: the custom smoke model is single-class, so it has no concept
     of "person" to rule things out. v3 adds two independent defenses:
       1. A lightweight general-purpose person detector (COCO YOLOv8n, class 0)
          runs alongside the smoke model. Any smoke detection that spatially
          overlaps a detected person (IoU) is vetoed outright.
       2. A texture/edge-density evidence term: real smoke is diffuse and
          soft-edged; people (clothing, limbs, faces) are high-frequency/
          textured. This acts as a second line of defense even in frames
          where the person detector misses (occlusion, motion blur, etc).

Everything from v2 (motion confirmation, color/desaturation check, ROI
exclusion zones, diagnostic CSV + audit crops) is retained.
"""

import csv
import math
import os
import sys
from collections import deque, namedtuple
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class Config:
    # --- Detection / tracking ---
    DETECTION_CONF = 0.05
    CONFIRM_CONF = 0.25
    IOU = 0.30
    IMG_SIZE = 960
    TRACKER = "bytetrack.yaml"

    # --- Temporal evidence window ---
    WINDOW_FRAMES = 45
    MIN_FRAMES_FOR_ALARM = 10

    # --- Re-ID stitching ---
    REID_MAX_GAP_FRAMES = 15
    REID_MAX_CENTER_DIST_NORM = 0.06
    REID_MAX_AREA_RATIO = 2.5

    # --- Motion confirmation (background subtraction) ---
    USE_MOTION_CONFIRMATION = True
    BG_HISTORY = 500
    BG_VAR_THRESHOLD = 16
    BG_MORPH_KERNEL = 5
    MOTION_OVERLAP_TARGET = 0.15

    # --- Color/desaturation heuristic ---
    USE_COLOR_DESATURATION_CHECK = True
    MAX_SMOKE_SATURATION = 60

    # --- Texture/edge-density heuristic (smoke = diffuse/soft; people/objects = textured) ---
    USE_TEXTURE_CHECK = True
    EDGE_VARIANCE_MAX = 500.0   # Laplacian variance above this looks "solid/textured", not smoke-like
                                 # NOTE: calibrate this against your own footage using the diagnostic CSV.

    # --- Person-detection veto ---
    USE_PERSON_FILTER = True
    PERSON_FILTER_MODEL = "yolov8n.pt"       # auto-downloads a standard COCO-pretrained model
    PERSON_FILTER_CONF = 0.35
    PERSON_FILTER_EVERY_N_FRAMES = 2         # run every N frames, cache result in between (perf)
    PERSON_IOU_VETO_THRESHOLD = 0.20         # smoke bbox overlapping a person by more than this is vetoed

    # --- Known false-positive zones: normalized (0-1) polygons [(x1,y1),(x2,y2),...] ---
    ROI_EXCLUSION_ZONES = []

    # --- Evidence weights (sum to 1.0 here, but doesn't have to; tune ALARM_SCORE_THRESHOLD too) ---
    W_PERSISTENCE = 0.15
    W_CONF_ESCALATION = 0.15
    W_AREA_GROWTH = 0.10
    W_UPWARD_DRIFT = 0.10
    W_MOTION_CONFIRM = 0.15
    W_COLOR_DESAT = 0.15
    W_TEXTURE = 0.20
    ALARM_SCORE_THRESHOLD = 0.60

    # --- Growth/drift sanity bounds ---
    MIN_AREA_GROWTH_RATIO = 1.15
    MIN_UPWARD_DRIFT_NORM = 0.01

    # --- Alarm hysteresis (fixes "still shows smoke after it's gone") ---
    RECENT_SUBWINDOW_FRAMES = 12             # only the most recent N observations decide "is it STILL there"
    SUSTAIN_SCORE_THRESHOLD = 0.45           # lower bar than ALARM_SCORE_THRESHOLD -- easier to sustain than to raise
    CLEAR_AFTER_FRAMES_NO_QUALIFYING = 60    # real elapsed video frames (~2s @30fps) of no qualifying evidence -> clear

    # --- Diagnostics ---
    ENABLE_DIAGNOSTIC_LOG = True
    DIAGNOSTIC_LOG_FILENAME = "smoke_diagnostics.csv"
    SAVE_AUDIT_CROPS = True
    AUDIT_CROPS_DIRNAME = "audit_crops"
    AUDIT_CROP_EVERY_N_FRAMES = 5
    NEAR_MISS_SCORE_MIN = 0.35


Obs = namedtuple("Obs", "frame_idx cx cy area conf motion color texture x1 y1 x2 y2")


def point_in_polygon(x, y, polygon):
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def in_excluded_zone(cx_norm, cy_norm, zones):
    return any(point_in_polygon(cx_norm, cy_norm, zone) for zone in zones)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------
class PlumeCandidate:
    __slots__ = ("canonical_id", "observations", "last_frame_idx")

    def __init__(self, canonical_id, obs: Obs, window):
        self.canonical_id = canonical_id
        self.observations = deque(maxlen=window)
        self.observations.append(obs)
        self.last_frame_idx = obs.frame_idx

    def add(self, obs: Obs):
        self.observations.append(obs)
        self.last_frame_idx = obs.frame_idx


class PlumeTrackManager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.active = {}
        self.recently_dead = {}

    @staticmethod
    def diagonal(shape):
        h, w = shape[:2]
        return math.hypot(w, h)

    def _try_stitch(self, cx, cy, area, frame_idx, diagonal):
        best_id, best_dist = None, None
        for tid, cand in self.recently_dead.items():
            gap = frame_idx - cand.last_frame_idx
            if gap <= 0 or gap > self.cfg.REID_MAX_GAP_FRAMES:
                continue
            prev = cand.observations[-1]
            dist_norm = math.hypot(cx - prev.cx, cy - prev.cy) / diagonal
            area_ratio = max(area, prev.area) / max(min(area, prev.area), 1e-6)
            if dist_norm <= self.cfg.REID_MAX_CENTER_DIST_NORM and area_ratio <= self.cfg.REID_MAX_AREA_RATIO:
                if best_dist is None or dist_norm < best_dist:
                    best_dist, best_id = dist_norm, tid
        return best_id

    def update(self, track_id, obs: Obs, diagonal):
        if track_id in self.active:
            cand = self.active[track_id]
            cand.add(obs)
            return cand

        stitch_id = self._try_stitch(obs.cx, obs.cy, obs.area, obs.frame_idx, diagonal)
        if stitch_id is not None:
            cand = self.recently_dead.pop(stitch_id)
            cand.add(obs)
            self.active[track_id] = cand
            return cand

        cand = PlumeCandidate(track_id, obs, self.cfg.WINDOW_FRAMES)
        self.active[track_id] = cand
        return cand

    def retire_stale(self, seen_ids, frame_idx, on_expire=None):
        for tid in list(self.active.keys()):
            if tid not in seen_ids:
                self.recently_dead[tid] = self.active.pop(tid)

        for tid in list(self.recently_dead.keys()):
            cand = self.recently_dead[tid]
            if frame_idx - cand.last_frame_idx > self.cfg.REID_MAX_GAP_FRAMES:
                if on_expire is not None:
                    on_expire(cand)
                del self.recently_dead[tid]


@dataclass
class Evidence:
    n: int
    max_conf: float
    area_growth_ratio: float
    upward_drift_norm: float
    persistence_score: float
    conf_score: float
    growth_score: float
    drift_score: float
    motion_score: float
    color_score: float
    texture_score: float
    score: float


def evaluate_evidence_from_obs(obs_list, cfg: Config, diagonal: float):
    n = len(obs_list)
    if n < 2:
        return None

    confs = [o.conf for o in obs_list]
    max_conf = max(confs)

    third = max(1, n // 3)
    first_chunk = obs_list[:third]
    last_chunk = obs_list[-third:]

    avg_area_first = sum(o.area for o in first_chunk) / len(first_chunk)
    avg_area_last = sum(o.area for o in last_chunk) / len(last_chunk)
    area_growth_ratio = avg_area_last / max(avg_area_first, 1e-6)

    avg_cy_first = sum(o.cy for o in first_chunk) / len(first_chunk)
    avg_cy_last = sum(o.cy for o in last_chunk) / len(last_chunk)
    upward_drift_norm = (avg_cy_first - avg_cy_last) / diagonal

    avg_motion = sum(o.motion for o in obs_list) / n
    avg_color = sum(o.color for o in obs_list) / n
    avg_texture = sum(o.texture for o in obs_list) / n

    persistence_score = min(1.0, n / cfg.MIN_FRAMES_FOR_ALARM)
    conf_score = 1.0 if max_conf >= cfg.CONFIRM_CONF else max_conf / cfg.CONFIRM_CONF
    growth_score = min(1.0, max(0.0, (area_growth_ratio - 1.0) / (cfg.MIN_AREA_GROWTH_RATIO - 1.0)))
    drift_score = min(1.0, max(0.0, upward_drift_norm / cfg.MIN_UPWARD_DRIFT_NORM))
    motion_score = min(1.0, avg_motion) if cfg.USE_MOTION_CONFIRMATION else 0.0
    color_score = min(1.0, avg_color) if cfg.USE_COLOR_DESATURATION_CHECK else 0.0
    texture_score = min(1.0, avg_texture) if cfg.USE_TEXTURE_CHECK else 0.0

    score = (
        cfg.W_PERSISTENCE * persistence_score
        + cfg.W_CONF_ESCALATION * conf_score
        + cfg.W_AREA_GROWTH * growth_score
        + cfg.W_UPWARD_DRIFT * drift_score
        + cfg.W_MOTION_CONFIRM * motion_score
        + cfg.W_COLOR_DESAT * color_score
        + cfg.W_TEXTURE * texture_score
    )

    return Evidence(
        n=n, max_conf=max_conf, area_growth_ratio=area_growth_ratio,
        upward_drift_norm=upward_drift_norm, persistence_score=persistence_score,
        conf_score=conf_score, growth_score=growth_score, drift_score=drift_score,
        motion_score=motion_score, color_score=color_score, texture_score=texture_score,
        score=score,
    )


def evaluate_evidence(cand: PlumeCandidate, cfg: Config, diagonal: float):
    return evaluate_evidence_from_obs(list(cand.observations), cfg, diagonal)


def evaluate_recent(cand: PlumeCandidate, cfg: Config, diagonal: float):
    obs_list = list(cand.observations)[-cfg.RECENT_SUBWINDOW_FRAMES:]
    return evaluate_evidence_from_obs(obs_list, cfg, diagonal)


def compute_shape_signals(orig_img, fg_mask, x1, y1, x2, y2, cfg: Config):
    """Returns (motion_score, color_score, texture_score) for a bbox region."""
    h, w = orig_img.shape[:2]
    xi1, yi1 = int(max(0, x1)), int(max(0, y1))
    xi2, yi2 = int(min(w, x2)), int(min(h, y2))
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0, 0.0, 0.0

    motion_score = 0.0
    if cfg.USE_MOTION_CONFIRMATION and fg_mask is not None:
        roi_mask = fg_mask[yi1:yi2, xi1:xi2]
        if roi_mask.size > 0:
            fg_ratio = float((roi_mask > 0).sum()) / roi_mask.size
            motion_score = min(1.0, fg_ratio / cfg.MOTION_OVERLAP_TARGET)

    roi_bgr = orig_img[yi1:yi2, xi1:xi2]
    color_score = 0.0
    texture_score = 0.0
    if roi_bgr.size > 0:
        if cfg.USE_COLOR_DESATURATION_CHECK:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            mean_sat = float(hsv[:, :, 1].mean())
            color_score = max(0.0, min(1.0, (cfg.MAX_SMOKE_SATURATION - mean_sat) / cfg.MAX_SMOKE_SATURATION))

        if cfg.USE_TEXTURE_CHECK:
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            edge_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            texture_score = max(0.0, min(1.0, (cfg.EDGE_VARIANCE_MAX - edge_var) / cfg.EDGE_VARIANCE_MAX))

    return motion_score, color_score, texture_score


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
class DiagnosticLogger:
    def __init__(self, output_dir, cfg: Config):
        self.cfg = cfg
        self.enabled = cfg.ENABLE_DIAGNOSTIC_LOG
        self.audit_enabled = cfg.SAVE_AUDIT_CROPS
        self._writer = None
        self._file = None

        if self.enabled:
            log_path = os.path.join(output_dir, cfg.DIAGNOSTIC_LOG_FILENAME)
            self._file = open(log_path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow([
                "frame_idx", "canonical_id", "track_id", "event", "n", "max_conf",
                "area_growth_ratio", "upward_drift_norm", "persistence_score",
                "conf_score", "growth_score", "drift_score", "motion_score",
                "color_score", "texture_score", "score", "fired",
            ])

        if self.audit_enabled:
            self.audit_dir = os.path.join(output_dir, cfg.AUDIT_CROPS_DIRNAME)
            os.makedirs(self.audit_dir, exist_ok=True)

    def log_evaluation(self, frame_idx, track_id, canonical_id, evidence: Evidence, fired, event="eval"):
        if self.enabled and self._writer is not None:
            self._writer.writerow([
                frame_idx, canonical_id, track_id, event, evidence.n, round(evidence.max_conf, 3),
                round(evidence.area_growth_ratio, 3), round(evidence.upward_drift_norm, 4),
                round(evidence.persistence_score, 3), round(evidence.conf_score, 3),
                round(evidence.growth_score, 3), round(evidence.drift_score, 3),
                round(evidence.motion_score, 3), round(evidence.color_score, 3),
                round(evidence.texture_score, 3), round(evidence.score, 3), int(fired),
            ])

    def log_event_only(self, frame_idx, canonical_id, track_id, event):
        if self.enabled and self._writer is not None:
            self._writer.writerow([frame_idx, canonical_id, track_id, event] + [""] * 12)

    def log_expiry(self, cand: PlumeCandidate, cfg: Config, diagonal: float):
        evidence = evaluate_evidence(cand, cfg, diagonal)
        if evidence is not None:
            self.log_evaluation(cand.last_frame_idx, cand.canonical_id, "expired", evidence, fired=False, event="expired_no_alarm")

    def save_crop(self, frame_bgr, x1, y1, x2, y2, tag, score):
        if not self.audit_enabled:
            return
        h, w = frame_bgr.shape[:2]
        pad = 20
        xi1, yi1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        xi2, yi2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        if xi2 <= xi1 or yi2 <= yi1:
            return
        crop = frame_bgr[yi1:yi2, xi1:xi2].copy()
        cv2.rectangle(crop, (int(x1 - xi1), int(y1 - yi1)), (int(x2 - xi1), int(y2 - yi1)), (0, 0, 255), 2)
        filename = f"{tag}_score{score:.2f}.jpg"
        cv2.imwrite(os.path.join(self.audit_dir, filename), crop)

    def close(self):
        if self._file is not None:
            self._file.close()


def download_weights():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        import subprocess
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    print("Verifying model weights...")
    return hf_hub_download(repo_id="kittendev/YOLOv8m-smoke-detection", filename="best.pt")


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------
def run_industrial_monitor(source_video, output_dir, cfg: Config = Config()):
    weights_path = download_weights()
    model = YOLO(weights_path)

    person_model = None
    if cfg.USE_PERSON_FILTER:
        print("[SYSTEM] Loading person-detection filter model...")
        person_model = YOLO(cfg.PERSON_FILTER_MODEL)

    print(f"\n[SYSTEM] Starting continuous monitoring on: {source_video}")
    print(f"[SYSTEM] Entry conf={cfg.DETECTION_CONF}, confirm conf={cfg.CONFIRM_CONF}, "
          f"raise threshold={cfg.ALARM_SCORE_THRESHOLD}, sustain threshold={cfg.SUSTAIN_SCORE_THRESHOLD}.")
    print(f"[SYSTEM] Motion={cfg.USE_MOTION_CONFIRMATION}, color={cfg.USE_COLOR_DESATURATION_CHECK}, "
          f"texture={cfg.USE_TEXTURE_CHECK}, person-filter={cfg.USE_PERSON_FILTER}, "
          f"ROI exclusions={len(cfg.ROI_EXCLUSION_ZONES)} zone(s).\n")

    results = model.track(
        source=source_video,
        conf=cfg.DETECTION_CONF,
        iou=cfg.IOU,
        imgsz=cfg.IMG_SIZE,
        tracker=cfg.TRACKER,
        stream=True,
        show=False,
        save=True,
        project=output_dir,
        name="industrial_smoke_alerts",
        exist_ok=True,
    )

    manager = PlumeTrackManager(cfg)
    logger = DiagnosticLogger(output_dir, cfg)

    bg_subtractor = None
    if cfg.USE_MOTION_CONFIRMATION:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=cfg.BG_HISTORY, varThreshold=cfg.BG_VAR_THRESHOLD, detectShadows=False
        )
    morph_kernel = np.ones((cfg.BG_MORPH_KERNEL, cfg.BG_MORPH_KERNEL), np.uint8)

    diagonal = None
    cached_person_boxes = []          # list of (x1,y1,x2,y2), refreshed every N frames
    alarm_active_since_frame = {}     # canonical_id -> frame_idx when alarm was first raised
    alarm_last_qualifying_frame = {}  # canonical_id -> frame_idx of last sustain-qualifying evidence

    def on_expire(cand):
        if diagonal is not None:
            logger.log_expiry(cand, cfg, diagonal)

    try:
        for frame_idx, result in enumerate(results):
            orig_img = result.orig_img
            if diagonal is None:
                diagonal = manager.diagonal(result.orig_shape)

            fg_mask = None
            if cfg.USE_MOTION_CONFIRMATION:
                fg_mask = bg_subtractor.apply(orig_img)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)

            if cfg.USE_PERSON_FILTER and frame_idx % cfg.PERSON_FILTER_EVERY_N_FRAMES == 0:
                person_results = person_model.predict(
                    orig_img, classes=[0], conf=cfg.PERSON_FILTER_CONF, verbose=False
                )
                cached_person_boxes = []
                if person_results and person_results[0].boxes is not None:
                    for pbox in person_results[0].boxes:
                        cached_person_boxes.append(tuple(pbox.xyxy[0].tolist()))

            boxes = result.boxes
            seen_ids = set()

            if boxes is not None and len(boxes) > 0:
                h, w = orig_img.shape[:2]

                for box in boxes:
                    if box.id is None:
                        continue

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    if "smoke" not in class_name.lower():
                        continue

                    track_id = int(box.id[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

                    if cfg.ROI_EXCLUSION_ZONES and in_excluded_zone(cx / w, cy / h, cfg.ROI_EXCLUSION_ZONES):
                        continue

                    if cfg.USE_PERSON_FILTER and cached_person_boxes:
                        max_person_iou = max(
                            (iou_xyxy((x1, y1, x2, y2), pbox) for pbox in cached_person_boxes),
                            default=0.0,
                        )
                        if max_person_iou > cfg.PERSON_IOU_VETO_THRESHOLD:
                            logger.log_event_only(frame_idx, track_id, track_id, "person_veto")
                            continue

                    motion_score, color_score, texture_score = compute_shape_signals(
                        orig_img, fg_mask, x1, y1, x2, y2, cfg
                    )

                    seen_ids.add(track_id)
                    obs = Obs(frame_idx, cx, cy, area, confidence, motion_score, color_score,
                              texture_score, x1, y1, x2, y2)
                    cand = manager.update(track_id, obs, diagonal)

                    evidence = evaluate_evidence(cand, cfg, diagonal)
                    if evidence is None:
                        continue

                    canonical_id = cand.canonical_id
                    is_active_alarm = canonical_id in alarm_active_since_frame

                    if not is_active_alarm:
                        # --- RAISE check: needs the full window to clear the higher bar ---
                        fired = evidence.n >= cfg.MIN_FRAMES_FOR_ALARM and evidence.score >= cfg.ALARM_SCORE_THRESHOLD
                        logger.log_evaluation(frame_idx, track_id, canonical_id, evidence, fired)

                        if fired:
                            coords_formatted = [round(c, 1) for c in (x1, y1, x2, y2)]
                            print(f"🚨 [ALARM RAISED] Early Smoke verified!")
                            print(f"   ┣━ Plume ID (canonical): #{canonical_id}  (current track #{track_id})")
                            print(f"   ┣━ Score: {evidence.score:.2f}  "
                                  f"(persist={evidence.persistence_score:.2f}, conf={evidence.conf_score:.2f}, "
                                  f"growth={evidence.growth_score:.2f}, drift={evidence.drift_score:.2f}, "
                                  f"motion={evidence.motion_score:.2f}, color={evidence.color_score:.2f}, "
                                  f"texture={evidence.texture_score:.2f})")
                            print(f"   ┣━ Location: {coords_formatted}")
                            print(f"   ┗━ Verified over {evidence.n} observed frames.\n")

                            alarm_active_since_frame[canonical_id] = frame_idx
                            alarm_last_qualifying_frame[canonical_id] = frame_idx
                            logger.save_crop(orig_img, x1, y1, x2, y2, f"ALARM_RAISED_f{frame_idx}_id{canonical_id}", evidence.score)

                        elif (
                            cfg.SAVE_AUDIT_CROPS
                            and evidence.score >= cfg.NEAR_MISS_SCORE_MIN
                            and frame_idx % cfg.AUDIT_CROP_EVERY_N_FRAMES == 0
                        ):
                            logger.save_crop(orig_img, x1, y1, x2, y2, f"nearmiss_f{frame_idx}_id{canonical_id}", evidence.score)

                    else:
                        # --- SUSTAIN check: use only the recent sub-window and a lower bar ---
                        recent_evidence = evaluate_recent(cand, cfg, diagonal)
                        still_qualifies = recent_evidence is not None and recent_evidence.score >= cfg.SUSTAIN_SCORE_THRESHOLD
                        logger.log_evaluation(frame_idx, track_id, canonical_id, evidence, fired=still_qualifies, event="sustain_check")
                        if still_qualifies:
                            alarm_last_qualifying_frame[canonical_id] = frame_idx

            manager.retire_stale(seen_ids, frame_idx, on_expire=on_expire)

            # --- Auto-clear check: runs every frame regardless of track lifecycle ---
            for canonical_id in list(alarm_active_since_frame.keys()):
                last_ok = alarm_last_qualifying_frame.get(canonical_id, alarm_active_since_frame[canonical_id])
                if frame_idx - last_ok > cfg.CLEAR_AFTER_FRAMES_NO_QUALIFYING:
                    print(f"✅ [ALARM CLEARED] Plume #{canonical_id} no longer shows sustained evidence "
                          f"(no qualifying frames for {cfg.CLEAR_AFTER_FRAMES_NO_QUALIFYING}+ frames).\n")
                    logger.log_event_only(frame_idx, canonical_id, canonical_id, "alarm_cleared")
                    del alarm_active_since_frame[canonical_id]
                    del alarm_last_qualifying_frame[canonical_id]
    finally:
        logger.close()

    print(f"\n[SYSTEM] Video processing complete.")
    print(f"[SYSTEM] Review {cfg.DIAGNOSTIC_LOG_FILENAME} and {cfg.AUDIT_CROPS_DIRNAME}/ in your output folder.")


if __name__ == "__main__":
    TEST_VIDEO_PATH = r"C:\Users\HP\Downloads\6185928-hd_1920_1080_30fps.mp4"
    OUTPUT_DIRECTORY = r"C:\Users\HP\Downloads"
    run_industrial_monitor(TEST_VIDEO_PATH, OUTPUT_DIRECTORY)