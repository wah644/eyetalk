"""
eyetrax-train-swipe
====================
Interactive tool for recording swipe-typing templates used by the DTW
word-matching engine.

Workflow (per sample)
---------------------
1. The target word is shown on screen with the correct START and END key
   rows highlighted.
2. Dwell on the START row (first letter's key group) for 0.5 s — recording
   begins.  If you dwell on the wrong row an error is shown and you retry.
3. Sweep your gaze through the keyboard as you would when typing the word.
4. Dwell on the END row (last letter's key group) for 0.6 s — recording
   stops.  If you dwell on the wrong row the sample is discarded and you
   restart from step 2.

Only words with 4 or more characters are trained (shorter words are not
worth swiping).

Controls: ESC = skip word | Q = quit & save

Usage
-----
  # Calibrate then train in one shot (recommended):
  eyetrax-train-swipe \\
      --calibration vertical --camera 0 --landmark-alpha 0.7 \\
      --multi-position --save-calibration my_model.pkl \\
      --swipe-output swipe_templates.json

  # Use a previously saved gaze model:
  eyetrax-train-swipe \\
      --model-file my_model.pkl --camera 0 \\
      --landmark-alpha 0.7 --multi-position \\
      --swipe-output swipe_templates.json

  # Train specific words only:
  eyetrax-train-swipe \\
      --model-file my_model.pkl --camera 0 \\
      --landmark-alpha 0.7 --multi-position \\
      --swipe-output swipe_templates.json \\
      --words hello water weather food
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from eyetrax.app.swipe_dtw import SwipeTemplateDB, SWIPE_ARM_DWELL, SWIPE_END_DWELL
from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
    run_multi_position_calibration,
    run_vertical_enhanced_calibration,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, iter_frames


# ---------------------------------------------------------------------------
# Keyboard layout (must match keyboard_demo.py)
# ---------------------------------------------------------------------------

KEYBOARD_KEYS = ["ABCD", "EFGH", "IJKL", "MNOP", "QRSTU", "VWXYZ"]  # no SPACE in swipe mode
KEY_WIDTH     = 300
KEY_GAP       = 10
MIN_WORD_LEN  = 1       # single-letter words supported (dwell same row twice)

LETTER_TO_ROW: dict[str, int] = {}
for _idx, _grp in enumerate(KEYBOARD_KEYS):
    for _ch in _grp.lower():
        LETTER_TO_ROW[_ch] = _idx

# Dwell thresholds imported from swipe_dtw — same values used in keyboard_demo
MIN_RECORD_SECS = 0.5   # don't check for end-dwell until this long after start


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record swipe-typing templates for the DTW keyboard"
    )
    p.add_argument("--model-file",       default=None,
                   help="Pre-trained gaze model (.pkl). If omitted, calibrates first.")
    p.add_argument("--save-calibration", default=None, metavar="PATH",
                   help="Save freshly calibrated model here.")
    p.add_argument("--calibration",      default="vertical",
                   choices=["9p", "5p", "lissajous", "vertical"])
    p.add_argument("--camera",           type=int, default=0)
    p.add_argument("--landmark-alpha",   type=float, default=0.7)
    p.add_argument("--multi-position",   action="store_true")
    p.add_argument("--swipe-output",     default="swipe_templates.json",
                   help="File to save/append swipe templates.")
    p.add_argument("--samples",          type=int, default=3,
                   help="Recordings per word (default: 3).")
    p.add_argument("--words",            nargs="*", default=None,
                   help="Words to train (default: all 4+ char words from words.txt).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_word_list(words_arg: list[str] | None) -> list[str]:
    if words_arg:
        words = [w.lower() for w in words_arg]
    else:
        path = Path(__file__).parent / "words.txt"
        try:
            with open(path) as f:
                words = [l.strip().lower() for l in f if l.strip()]
        except FileNotFoundError:
            print(f"[train_swipe] Warning: words.txt not found, using short fallback")
            words = ["hello", "help", "water", "food", "pain", "cold", "warm",
                     "tired", "hungry", "please", "thank", "sleep", "move"]

    filtered = [w for w in words if len(w) >= MIN_WORD_LEN
                and all(c in LETTER_TO_ROW for c in w)]
    skipped = len(words) - len(filtered)
    if skipped:
        print(f"[train_swipe] Skipped {skipped} word(s) with unmapped letters")
    return filtered


def _word_key_rows(word: str) -> tuple[int, int]:
    """Return (start_row, end_row) for the first and last letter of *word*."""
    return LETTER_TO_ROW[word[0]], LETTER_TO_ROW[word[-1]]


def _key_bounds(row: int, screen_w: int, screen_h: int) -> tuple[int, int, int, int]:
    n       = len(KEYBOARD_KEYS)
    avail_h = screen_h - (n - 1) * KEY_GAP
    key_h   = avail_h // n
    y1 = row * (key_h + KEY_GAP)
    y2 = y1 + key_h
    x1 = (screen_w - KEY_WIDTH) // 2
    x2 = x1 + KEY_WIDTH
    return x1, y1, x2, y2


def _get_hovered_row(y_pred: int | None, screen_w: int, screen_h: int) -> int | None:
    """Return the key row index the gaze is currently on, or None."""
    if y_pred is None:
        return None
    for ki in range(len(KEYBOARD_KEYS)):
        _, y1, _, y2 = _key_bounds(ki, screen_w, screen_h)
        if y1 <= y_pred <= y2:
            return ki
    return None


def _draw_keyboard(canvas: np.ndarray, screen_w: int, screen_h: int,
                   start_row: int, end_row: int,
                   hovered_row: int | None,
                   dwell_row: int | None, dwell_progress: float,
                   is_recording: bool) -> None:
    for i, label in enumerate(KEYBOARD_KEYS):
        x1, y1, x2, y2 = _key_bounds(i, screen_w, screen_h)

        if i == start_row and not is_recording:
            bg, border, thick = (20, 60, 20), (0, 255, 100), 4   # green = start
        elif i == end_row and is_recording:
            bg, border, thick = (20, 40, 60), (0, 180, 255), 4   # cyan  = end
        elif i == hovered_row:
            bg, border, thick = (50, 50, 50), (200, 200, 200), 3
        else:
            bg, border, thick = (30, 30, 30), (90, 90, 90), 2

        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, thick)

        fs = 1.0 if len(label) <= 5 else 0.8
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.putText(canvas, label,
                    (x1 + (KEY_WIDTH - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (220, 220, 220), 2, cv2.LINE_AA)

        # Dwell arc on the active row
        if i == dwell_row and dwell_progress > 0:
            cx, cy_k = (x1 + x2) // 2, (y1 + y2) // 2
            col = (0, 255, 100) if not is_recording else (0, 180, 255)
            cv2.ellipse(canvas, (cx, cy_k), (30, 30), -90,
                        0, int(360 * dwell_progress), col, 5)



def _flash_error(cap, gaze, screen_w, screen_h, message: str, duration: float = 1.2):
    """Display a red error banner for *duration* seconds, still reading frames."""
    deadline = time.time() + duration
    for frame in iter_frames(cap):
        gaze.extract_features(frame)   # keep mediapipe warm
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = (30, 10, 10)
        (tw, _), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.putText(canvas, message,
                    ((screen_w - tw) // 2, screen_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 60, 220), 3, cv2.LINE_AA)
        cv2.imshow("Swipe Training", canvas)
        cv2.waitKey(1)
        if time.time() >= deadline:
            break


# ---------------------------------------------------------------------------
# Per-sample recording  (returns trajectory or None if skipped)
# ---------------------------------------------------------------------------

def _record_sample(cap, gaze, screen_w: int, screen_h: int,
                   word: str, sample_num: int, total_samples: int,
                   start_row: int, end_row: int) -> list[float] | None:
    """
    Run the dwell-start → swipe → dwell-end flow for one sample of *word*.

    Returns the trajectory (list of normalised y positions) or None if the
    user pressed ESC.  Raises SystemExit on Q.
    """
    same_row = (start_row == end_row)

    # State: "waiting_start" | "recording"
    state          = "waiting_start"
    trajectory: list[float] = []
    record_start_t: float | None = None

    # Dwell tracking
    dwell_row:    int   | None = None
    dwell_start:  float | None = None

    for frame in iter_frames(cap):
        features, blink = gaze.extract_features(frame)
        x_pred = y_pred = None
        if features is not None and not blink:
            gp = gaze.predict(np.array([features]))[0]
            x_pred, y_pred = int(gp[0]), int(gp[1])

        now     = time.time()
        hovered = _get_hovered_row(y_pred, screen_w, screen_h)

        # ---- update dwell tracker ----
        if hovered != dwell_row:
            dwell_row   = hovered
            dwell_start = now if hovered is not None else None
        dwell_elapsed = (now - dwell_start) if dwell_start is not None else 0.0

        threshold = SWIPE_ARM_DWELL if state == "waiting_start" else SWIPE_END_DWELL

        # ---- state transitions ----
        if state == "waiting_start":
            if dwell_row is not None and dwell_elapsed >= threshold:
                if dwell_row == start_row:
                    state          = "recording"
                    trajectory     = []
                    record_start_t = now
                    dwell_row      = None
                    dwell_start    = None
                else:
                    # Wrong start key — flash error and reset
                    _flash_error(cap, gaze, screen_w, screen_h,
                                 f"Wrong key!  Start on  {KEYBOARD_KEYS[start_row]}")
                    dwell_row   = None
                    dwell_start = None

        elif state == "recording":
            if y_pred is not None:
                trajectory.append(y_pred / screen_h)

            # Don't check for end-dwell until minimum record time has passed
            # (prevents same-row words from ending immediately after starting)
            min_elapsed = (now - record_start_t) >= MIN_RECORD_SECS if record_start_t else False

            # Stop only when the CORRECT end key is dwelled — ignore any other key dwell
            if min_elapsed and dwell_row == end_row and dwell_elapsed >= threshold:
                return trajectory

        # ---- draw (no gaze cursor — reduces visible jitter) ----
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        canvas[:] = (22, 22, 32)

        is_recording = (state == "recording")
        dp = min(dwell_elapsed / threshold, 1.0) if dwell_row is not None else 0.0
        _draw_keyboard(canvas, screen_w, screen_h,
                       start_row, end_row, hovered,
                       dwell_row, dp, is_recording)

        # HUD
        word_col = (0, 255, 150) if is_recording else (255, 255, 255)
        cv2.putText(canvas, word.upper(),
                    (screen_w // 2 - 200, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, word_col, 4, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"Sample {sample_num}/{total_samples}",
                    (screen_w // 2 - 100, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2, cv2.LINE_AA)

        if is_recording:
            rec_col = (0, 0, 200) if int(now * 2) % 2 == 0 else (0, 0, 100)
            cv2.circle(canvas, (screen_w - 45, 45), 18, rec_col, -1)
            cv2.putText(canvas, "REC",
                        (screen_w - 95, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
            cv2.putText(canvas,
                        f"Now dwell on END key:  {KEYBOARD_KEYS[end_row]}",
                        (screen_w // 2 - 240, screen_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas,
                        f"Points: {len(trajectory)}",
                        (50, screen_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas,
                        f"Dwell on START key:  {KEYBOARD_KEYS[start_row]}",
                        (screen_w // 2 - 240, screen_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2, cv2.LINE_AA)

        if same_row:
            hint = ("(single-letter word — dwell the same row twice)"
                    if len(word) == 1 else
                    "(start & end same row — you may sweep away and return)")
            cv2.putText(canvas, hint,
                        (screen_w // 2 - 340, screen_h - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 180, 100), 1, cv2.LINE_AA)

        cv2.putText(canvas, "ESC = skip word   Q = quit & save",
                    (screen_w // 2 - 220, screen_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1, cv2.LINE_AA)

        cv2.imshow("Swipe Training", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            raise SystemExit(0)
        if key == 27:   # ESC
            return None

    return None   # camera exhausted


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_training() -> None:
    args = _parse_args()

    # ---- gaze estimator ----
    gaze = GazeEstimator(
        model_name="ridge",
        landmark_alpha=args.landmark_alpha,
        include_face_position=args.multi_position,
    )

    if args.model_file and os.path.isfile(args.model_file):
        gaze.load_model(args.model_file)
        print(f"[train_swipe] Loaded gaze model from {args.model_file}")
    elif args.multi_position:
        run_multi_position_calibration(gaze, camera_index=args.camera,
                                       calibration_method=args.calibration)
    else:
        if args.calibration == "vertical":
            run_vertical_enhanced_calibration(gaze, camera_index=args.camera)
        elif args.calibration == "9p":
            run_9_point_calibration(gaze, camera_index=args.camera)
        elif args.calibration == "5p":
            run_5_point_calibration(gaze, camera_index=args.camera)
        else:
            run_lissajous_calibration(gaze, camera_index=args.camera)

    if args.save_calibration and not (args.model_file and os.path.isfile(args.model_file)):
        gaze.save_model(args.save_calibration)
        print(f"[train_swipe] Gaze model saved to {args.save_calibration}")

    screen_w, screen_h = get_screen_size()

    # ---- template DB ----
    db = SwipeTemplateDB()
    out_path = Path(args.swipe_output)
    if out_path.exists():
        db.load(out_path)
        print(f"[train_swipe] Appending to existing template file")

    words = _load_word_list(args.words)
    print(f"[train_swipe] {len(words)} words to train, {args.samples} samples each")

    cv2.namedWindow("Swipe Training", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Swipe Training", 0, 0)
    cv2.setWindowProperty("Swipe Training", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    try:
        with camera(args.camera) as cap:
            for word in words:
                already = db.sample_count(word)
                needed  = max(0, args.samples - already)
                if needed == 0:
                    print(f"[train_swipe] '{word}' already complete, skipping")
                    continue

                start_row, end_row = _word_key_rows(word)
                print(f"[train_swipe] '{word}'  start={KEYBOARD_KEYS[start_row]}  "
                      f"end={KEYBOARD_KEYS[end_row]}  "
                      f"({already} existing, need {needed} more)")

                collected = 0
                while collected < needed:
                    traj = _record_sample(
                        cap, gaze, screen_w, screen_h,
                        word,
                        sample_num=already + collected + 1,
                        total_samples=args.samples,
                        start_row=start_row,
                        end_row=end_row,
                    )

                    if traj is None:
                        # ESC pressed — skip remaining samples for this word
                        print(f"[train_swipe] '{word}' skipped")
                        break

                    if len(traj) < 3:
                        print(f"[train_swipe] '{word}' trajectory too short, retrying")
                        continue

                    db.add_template(word, traj)
                    collected += 1
                    print(f"[train_swipe] '{word}' sample {already + collected} saved "
                          f"({len(traj)} points)")
                    db.save(out_path)

                    # Brief green flash
                    flash = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                    flash[:] = (10, 40, 10)
                    cv2.putText(flash,
                                f"Saved!  '{word}'  "
                                f"({db.sample_count(word)}/{args.samples})",
                                (screen_w // 2 - 320, screen_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                (0, 255, 100), 3, cv2.LINE_AA)
                    cv2.imshow("Swipe Training", flash)
                    cv2.waitKey(700)

    except SystemExit:
        pass

    db.save(out_path)
    cv2.destroyAllWindows()
    print(f"[train_swipe] Done — {len(db.words())} words in {out_path}")

if __name__ == '__main__':
    run_training()
