"""
Stability Benchmark for eyeTrax gaze estimation.

Two-phase test protocol:
  Phase 1 (still-head):  fixate on targets with head still  --> baseline
  Phase 2 (moving-head): fixate on targets while gently moving head --> stress

Metrics (per-point and aggregate):
  - Accuracy:        mean Euclidean error (px and degrees of visual angle)
  - Precision S2S-RMS: RMS of successive gaze-point distances (jitter)
  - Precision STD:   std of gaze points around centroid
  - Systematic bias: mean signed error in X and Y
  - Head pose range: observed yaw/pitch/roll span
  - Robustness ratio: accuracy_moving / accuracy_still

Usage:
  python -m eyetrax.app.stability_benchmark --calibration 9p
  python -m eyetrax.app.stability_benchmark --compare before.json after.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size

AVERAGE_IPD_MM = 63.0
WEBCAM_HFOV_DEG = 60.0

GRID_ORDER_3x3 = [
    (1, 1),  # center
    (0, 0), (0, 2), (2, 0), (2, 2),  # corners
    (0, 1), (1, 0), (1, 2), (2, 1),  # edges
]


def _estimate_viewing_distance_mm(ipd_pixels: float, image_width: int) -> float:
    focal_px = (image_width / 2) / np.tan(np.radians(WEBCAM_HFOV_DEG / 2))
    return (AVERAGE_IPD_MM * focal_px) / max(ipd_pixels, 1.0)


def _pixels_to_degrees(px_error: float, pixel_pitch_mm: float,
                       viewing_distance_mm: float) -> float:
    return float(np.degrees(np.arctan(px_error * pixel_pitch_mm / viewing_distance_mm)))


def _compute_point_metrics(predictions: np.ndarray, target: np.ndarray,
                           poses: np.ndarray) -> dict:
    """Compute all metrics for a single target point."""
    if len(predictions) < 2:
        return None

    errors = predictions - target
    dists = np.linalg.norm(errors, axis=1)

    diffs = np.diff(predictions, axis=0)
    s2s_dists = np.linalg.norm(diffs, axis=1)

    centroid = predictions.mean(axis=0)
    deviations = np.linalg.norm(predictions - centroid, axis=1)

    poses_deg = np.degrees(poses)

    return {
        "target": target.tolist(),
        "n_samples": len(predictions),
        "predictions": predictions.tolist(),
        "poses_rad": poses.tolist(),
        "accuracy_px": float(dists.mean()),
        "precision_s2s_rms_px": float(np.sqrt(np.mean(s2s_dists ** 2))),
        "precision_std_px": float(deviations.std()),
        "bias_x_px": float(errors[:, 0].mean()),
        "bias_y_px": float(errors[:, 1].mean()),
        "pose_range_deg": {
            "yaw": float(poses_deg[:, 0].ptp()),
            "pitch": float(poses_deg[:, 1].ptp()),
            "roll": float(poses_deg[:, 2].ptp()),
        },
    }


def _aggregate_metrics(point_metrics: list[dict]) -> dict:
    valid = [m for m in point_metrics if m is not None]
    if not valid:
        return {}

    return {
        "accuracy_px": float(np.mean([m["accuracy_px"] for m in valid])),
        "precision_s2s_rms_px": float(np.mean([m["precision_s2s_rms_px"] for m in valid])),
        "precision_std_px": float(np.mean([m["precision_std_px"] for m in valid])),
        "bias_x_px": float(np.mean([m["bias_x_px"] for m in valid])),
        "bias_y_px": float(np.mean([m["bias_y_px"] for m in valid])),
        "mean_pose_range_deg": {
            "yaw": float(np.mean([m["pose_range_deg"]["yaw"] for m in valid])),
            "pitch": float(np.mean([m["pose_range_deg"]["pitch"] for m in valid])),
            "roll": float(np.mean([m["pose_range_deg"]["roll"] for m in valid])),
        },
    }


def _run_test_phase(
    gaze_estimator: GazeEstimator,
    smoother,
    cap: cv2.VideoCapture,
    points: list[tuple[int, int]],
    sw: int,
    sh: int,
    phase_name: str,
    pulse_d: float = 1.0,
    capture_d: float = 2.5,
) -> list[dict] | None:
    """Run one phase of the benchmark (still or moving). Returns per-point metrics."""
    win = "Stability Benchmark"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if phase_name == "moving":
        instruction = "GENTLY MOVE YOUR HEAD while looking at each dot"
    else:
        instruction = "Keep your head STILL while looking at each dot"

    _show_phase_intro(win, sw, sh, phase_name, instruction, cap)

    all_metrics = []

    for idx, (tx, ty) in enumerate(points):
        # -- pulse phase --
        ps = time.time()
        final_r = 20
        while True:
            e = time.time() - ps
            if e > pulse_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            r = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
            final_r = r
            cv2.circle(canvas, (tx, ty), r, (0, 255, 0), -1)
            cv2.circle(canvas, (tx, ty), 5, (0, 0, 0), -1)
            _draw_phase_hud(canvas, phase_name, idx, len(points), sw)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) == 27:
                return None

        # -- capture phase --
        cs = time.time()
        preds, poses = [], []
        while True:
            e = time.time() - cs
            if e > capture_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue

            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), final_r, (0, 255, 0), -1)
            cv2.circle(canvas, (tx, ty), 5, (0, 0, 0), -1)
            t = e / capture_d
            ease = t * t * (3 - 2 * t)
            ang = 360 * (1 - ease)
            cv2.ellipse(canvas, (tx, ty), (40, 40), 0, -90, -90 + ang,
                        (255, 255, 255), 4)
            _draw_phase_hud(canvas, phase_name, idx, len(points), sw)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) == 27:
                return None

            features, blink = gaze_estimator.extract_features(frame)
            if features is not None and not blink:
                gaze = gaze_estimator.predict(np.array([features]))[0]
                gx, gy = smoother.step(int(gaze[0]), int(gaze[1]))
                preds.append([gx, gy])
                if gaze_estimator._last_pose is not None:
                    poses.append(gaze_estimator._last_pose.copy())

        preds_arr = np.array(preds) if preds else np.empty((0, 2))
        poses_arr = np.array(poses) if poses else np.empty((0, 3))
        target_arr = np.array([tx, ty], dtype=float)

        metrics = _compute_point_metrics(preds_arr, target_arr, poses_arr)
        all_metrics.append(metrics)

    return all_metrics


def _show_phase_intro(win, sw, sh, phase_name, instruction, cap, duration=3.0):
    title = f"Phase: {phase_name.upper()}"
    start = time.time()
    while time.time() - start < duration:
        ret, _ = cap.read()
        if not ret:
            continue
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

        sz, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        cv2.putText(canvas, title,
                    ((sw - sz[0]) // 2, sh // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3, cv2.LINE_AA)

        sz2, _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(canvas, instruction,
                    ((sw - sz2[0]) // 2, sh // 3 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        remaining = max(0, duration - (time.time() - start))
        count_txt = f"Starting in {remaining:.0f}s..."
        sz3, _ = cv2.getTextSize(count_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(canvas, count_txt,
                    ((sw - sz3[0]) // 2, sh // 3 + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        if cv2.waitKey(1) == 27:
            return


def _draw_phase_hud(canvas, phase_name, point_idx, total_points, sw):
    label = f"{phase_name.upper()} | Point {point_idx + 1}/{total_points}"
    cv2.putText(canvas, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)


def _display_results(report: dict, sw: int, sh: int):
    win = "Benchmark Results"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    still_agg = report["phases"]["still"]["aggregate"]
    moving_agg = report["phases"]["moving"]["aggregate"]
    ratio = report["robustness_ratio"]

    lines = [
        ("STABILITY BENCHMARK RESULTS", (0, 255, 255), 1.4),
        ("", (0, 0, 0), 0.8),
        ("--- STILL HEAD ---", (0, 255, 0), 1.0),
        (f"  Accuracy:       {still_agg['accuracy_px']:.1f} px  |  {still_agg.get('accuracy_deg', 0):.2f} deg", (255, 255, 255), 0.8),
        (f"  Precision S2S:  {still_agg['precision_s2s_rms_px']:.1f} px", (255, 255, 255), 0.8),
        (f"  Precision STD:  {still_agg['precision_std_px']:.1f} px", (255, 255, 255), 0.8),
        (f"  Bias (X, Y):    ({still_agg['bias_x_px']:.1f}, {still_agg['bias_y_px']:.1f}) px", (255, 255, 255), 0.8),
        ("", (0, 0, 0), 0.8),
        ("--- MOVING HEAD ---", (0, 100, 255), 1.0),
        (f"  Accuracy:       {moving_agg['accuracy_px']:.1f} px  |  {moving_agg.get('accuracy_deg', 0):.2f} deg", (255, 255, 255), 0.8),
        (f"  Precision S2S:  {moving_agg['precision_s2s_rms_px']:.1f} px", (255, 255, 255), 0.8),
        (f"  Precision STD:  {moving_agg['precision_std_px']:.1f} px", (255, 255, 255), 0.8),
        (f"  Bias (X, Y):    ({moving_agg['bias_x_px']:.1f}, {moving_agg['bias_y_px']:.1f}) px", (255, 255, 255), 0.8),
        ("", (0, 0, 0), 0.8),
        (f"ROBUSTNESS RATIO: {ratio:.2f}  (1.0 = perfect)", (0, 255, 255), 1.1),
        ("", (0, 0, 0), 0.8),
        ("Press ESC to close", (150, 150, 150), 0.7),
    ]

    while True:
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        y = 80
        for text, color, scale in lines:
            if text:
                cv2.putText(canvas, text, (60, y),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
            y += int(45 * scale + 10)

        cv2.imshow(win, canvas)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyWindow(win)


def _print_comparison(before: dict, after: dict):
    """Print side-by-side comparison of two benchmark runs."""
    phases = ["still", "moving"]
    keys = [
        ("accuracy_px", "Accuracy (px)"),
        ("accuracy_deg", "Accuracy (deg)"),
        ("precision_s2s_rms_px", "Precision S2S-RMS (px)"),
        ("precision_std_px", "Precision STD (px)"),
        ("bias_x_px", "Bias X (px)"),
        ("bias_y_px", "Bias Y (px)"),
    ]

    print()
    print("=" * 72)
    print("  STABILITY BENCHMARK COMPARISON")
    print("=" * 72)
    print(f"  Before: {before['metadata']['timestamp']}")
    print(f"  After:  {after['metadata']['timestamp']}")
    print("=" * 72)

    for phase in phases:
        print(f"\n  --- {phase.upper()} HEAD ---")
        print(f"  {'Metric':<26s} {'Before':>10s} {'After':>10s} {'Change':>10s}")
        print(f"  {'-'*26} {'-'*10} {'-'*10} {'-'*10}")

        b_agg = before["phases"][phase]["aggregate"]
        a_agg = after["phases"][phase]["aggregate"]

        for key, label in keys:
            bv = b_agg.get(key, 0)
            av = a_agg.get(key, 0)
            if abs(bv) > 1e-9:
                pct = (av - bv) / abs(bv) * 100
                change = f"{pct:+.1f}%"
            else:
                change = "N/A"
            print(f"  {label:<26s} {bv:>10.2f} {av:>10.2f} {change:>10s}")

    br = before.get("robustness_ratio", 0)
    ar = after.get("robustness_ratio", 0)
    if abs(br) > 1e-9:
        pct = (ar - br) / abs(br) * 100
        change = f"{pct:+.1f}%"
    else:
        change = "N/A"

    print()
    print(f"  {'Robustness Ratio':<26s} {br:>10.2f} {ar:>10.2f} {change:>10s}")
    print(f"  (1.0 = head movement has no effect; lower is better)")
    print("=" * 72)
    print()


def run_benchmark():
    parser = argparse.ArgumentParser(
        description="Stability benchmark for eyeTrax gaze estimation"
    )
    parser.add_argument("--filter", choices=["kalman", "kde", "none"], default="none")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration", choices=["9p", "5p", "lissajous"], default="9p")
    parser.add_argument("--model", default="ridge")
    parser.add_argument("--model-file", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-timestamped)")
    parser.add_argument("--multi-pose", action="store_true", default=False,
                        help="Enable multi-pose calibration for head-movement robustness")
    parser.add_argument("--landmark-alpha", type=float, default=0.7,
                        help="EMA alpha for raw landmark smoothing (1.0=off, default: 0.7)")
    parser.add_argument("--feature-alpha", type=float, default=None,
                        help="EMA alpha for feature vector smoothing (None=off)")
    parser.add_argument("--pose-damping", type=float, default=0.0,
                        help="Head pose drift compensation (0.0=off, 1.0=full clamp)")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Compare two benchmark JSON files instead of running a test")
    args = parser.parse_args()

    if args.compare:
        with open(args.compare[0]) as f:
            before = json.load(f)
        with open(args.compare[1]) as f:
            after = json.load(f)
        _print_comparison(before, after)
        return

    sw, sh = get_screen_size()
    gaze_estimator = GazeEstimator(
        model_name=args.model,
        landmark_alpha=args.landmark_alpha,
        feature_alpha=args.feature_alpha,
    )
    gaze_estimator._pose_damping = args.pose_damping

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[benchmark] Loaded model from {args.model_file}")
    else:
        print("[benchmark] Running calibration...")
        mp = args.multi_pose
        if args.calibration == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=args.camera,
                                    multi_pose=mp)
        elif args.calibration == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=args.camera,
                                    multi_pose=mp)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=args.camera)
        print("[benchmark] Calibration complete.")

    if args.filter == "kalman":
        kf = make_kalman()
        smoother = KalmanSmoother(kf)
        smoother.tune(gaze_estimator, camera_index=args.camera)
    elif args.filter == "kde":
        smoother = KDESmoother(sw, sh, confidence=args.confidence)
    else:
        smoother = NoSmoother()

    points = compute_grid_points(GRID_ORDER_3x3, sw, sh, margin_ratio=0.10)

    cap = cv2.VideoCapture(args.camera)

    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, dur=2):
        cap.release()
        cv2.destroyAllWindows()
        print("[benchmark] Cancelled.")
        return

    print("[benchmark] Phase 1: STILL head")
    still_metrics = _run_test_phase(
        gaze_estimator, smoother, cap, points, sw, sh, phase_name="still"
    )
    if still_metrics is None:
        cap.release()
        cv2.destroyAllWindows()
        print("[benchmark] Cancelled during still phase.")
        return

    print("[benchmark] Phase 2: MOVING head")
    moving_metrics = _run_test_phase(
        gaze_estimator, smoother, cap, points, sw, sh, phase_name="moving"
    )
    if moving_metrics is None:
        cap.release()
        cv2.destroyAllWindows()
        print("[benchmark] Cancelled during moving phase.")
        return

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640

    ipd_samples = []
    for _ in range(30):
        ok, frame = cap.read()
        if not ok:
            continue
        gaze_estimator.extract_features(frame)
        if gaze_estimator._last_ipd_pixels is not None:
            ipd_samples.append(gaze_estimator._last_ipd_pixels)
    cap.release()

    ipd_px = float(np.median(ipd_samples)) if ipd_samples else 80.0

    viewing_dist_mm = _estimate_viewing_distance_mm(ipd_px, img_w)

    from screeninfo import get_monitors
    try:
        mon = get_monitors()[0]
        phys_w_mm = mon.width_mm or 344
    except Exception:
        phys_w_mm = 344
    pixel_pitch_mm = phys_w_mm / sw

    still_agg = _aggregate_metrics(still_metrics)
    moving_agg = _aggregate_metrics(moving_metrics)

    if still_agg:
        still_agg["accuracy_deg"] = _pixels_to_degrees(
            still_agg["accuracy_px"], pixel_pitch_mm, viewing_dist_mm
        )
    if moving_agg:
        moving_agg["accuracy_deg"] = _pixels_to_degrees(
            moving_agg["accuracy_px"], pixel_pitch_mm, viewing_dist_mm
        )

    robustness = (
        moving_agg["accuracy_px"] / still_agg["accuracy_px"]
        if still_agg and still_agg["accuracy_px"] > 1e-9
        else float("inf")
    )

    def _strip_predictions(metrics_list):
        """Return a copy without bulky per-frame arrays for the JSON summary."""
        stripped = []
        for m in metrics_list:
            if m is None:
                stripped.append(None)
                continue
            s = {k: v for k, v in m.items() if k not in ("predictions", "poses_rad")}
            stripped.append(s)
        return stripped

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "calibration_method": args.calibration,
            "model_type": args.model,
            "filter": args.filter,
            "screen_width": sw,
            "screen_height": sh,
            "estimated_viewing_distance_mm": round(viewing_dist_mm, 1),
            "estimated_ipd_pixels": round(ipd_px, 1),
            "pixel_pitch_mm": round(pixel_pitch_mm, 4),
            "multi_pose": args.multi_pose,
            "landmark_alpha": args.landmark_alpha,
            "feature_alpha": args.feature_alpha,
            "pose_damping": args.pose_damping,
        },
        "phases": {
            "still": {
                "per_point": _strip_predictions(still_metrics),
                "aggregate": still_agg,
            },
            "moving": {
                "per_point": _strip_predictions(moving_metrics),
                "aggregate": moving_agg,
            },
        },
        "robustness_ratio": round(robustness, 3),
    }

    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_path = f"benchmark_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[benchmark] Results saved to {out_path}")

    _print_summary(report)
    _display_results(report, sw, sh)
    cv2.destroyAllWindows()


def _print_summary(report: dict):
    still = report["phases"]["still"]["aggregate"]
    moving = report["phases"]["moving"]["aggregate"]
    ratio = report["robustness_ratio"]

    print()
    print("=" * 55)
    print("  STABILITY BENCHMARK RESULTS")
    print("=" * 55)
    print(f"  {'':26s} {'Still':>10s} {'Moving':>10s}")
    print(f"  {'-'*26} {'-'*10} {'-'*10}")
    print(f"  {'Accuracy (px)':<26s} {still['accuracy_px']:>10.1f} {moving['accuracy_px']:>10.1f}")
    print(f"  {'Accuracy (deg)':<26s} {still.get('accuracy_deg',0):>10.2f} {moving.get('accuracy_deg',0):>10.2f}")
    print(f"  {'Precision S2S-RMS (px)':<26s} {still['precision_s2s_rms_px']:>10.1f} {moving['precision_s2s_rms_px']:>10.1f}")
    print(f"  {'Precision STD (px)':<26s} {still['precision_std_px']:>10.1f} {moving['precision_std_px']:>10.1f}")
    print(f"  {'Bias X (px)':<26s} {still['bias_x_px']:>10.1f} {moving['bias_x_px']:>10.1f}")
    print(f"  {'Bias Y (px)':<26s} {still['bias_y_px']:>10.1f} {moving['bias_y_px']:>10.1f}")
    print(f"  {'-'*26} {'-'*10} {'-'*10}")
    print(f"  Robustness Ratio: {ratio:.2f}  (1.0 = perfect)")
    print("=" * 55)
    print()


if __name__ == "__main__":
    run_benchmark()
