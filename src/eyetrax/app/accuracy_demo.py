import argparse
import os
import time

import cv2
import numpy as np

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
    run_multi_position_calibration,
    run_vertical_enhanced_calibration,
    run_vertical_only_calibration,
    run_vertical_single_calibration,
)
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.screen import get_screen_size


def parse_args():
    parser = argparse.ArgumentParser(description="Eye-tracking accuracy demo")
    parser.add_argument(
        "--calibration",
        choices=["9p", "5p", "vertical", "vertical_single", "vertical-only", "lissajous"],
        default="9p",
        help="Calibration method (default: 9p)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--model",
        default="ridge",
        help="ML model for gaze estimation (default: ridge)",
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=None,
        help="Path to a previously-trained gaze model",
    )
    parser.add_argument(
        "--filter",
        choices=["kalman", "kde", "none"],
        default="none",
        help="Post-prediction filter (default: none)",
    )
    parser.add_argument(
        "--landmark-alpha",
        type=float,
        default=1.0,
        help="EMA alpha for landmark smoothing (1.0 = no smoothing, default: 1.0)",
    )
    parser.add_argument(
        "--multi-position",
        action="store_true",
        default=False,
        help="Add face anchor position features to the estimator (include_face_position=True)",
    )
    parser.add_argument(
        "--multi-pose",
        action="store_true",
        default=False,
        help="Add a head-rotation phase after each still-head capture during calibration",
    )
    parser.add_argument(
        "--capture-duration",
        type=float,
        default=1.0,
        help="Still-head capture duration per calibration point in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--multi-pose-duration",
        type=float,
        default=4.0,
        help="Head-rotation capture duration per calibration point in seconds (default: 4.0)",
    )
    return parser.parse_args()


def _run_calibration(args, gaze_estimator):
    """Dispatch to the appropriate calibration function."""
    cam = args.camera
    cd = args.capture_duration
    mp = args.multi_pose
    mpd = args.multi_pose_duration

    if args.multi_position:
        run_multi_position_calibration(
            gaze_estimator, camera_index=cam,
            calibration_method=args.calibration,
        )
        return

    if args.calibration == "9p":
        run_9_point_calibration(
            gaze_estimator, camera_index=cam, multi_pose=mp, multi_pose_d=mpd
        )
    elif args.calibration == "5p":
        run_5_point_calibration(
            gaze_estimator, camera_index=cam, multi_pose=mp, multi_pose_d=mpd
        )
    elif args.calibration == "lissajous":
        run_lissajous_calibration(gaze_estimator, camera_index=cam)
    elif args.calibration == "vertical":
        run_vertical_enhanced_calibration(
            gaze_estimator, camera_index=cam, multi_pose=mp, multi_pose_d=mpd
        )
    elif args.calibration == "vertical_single":
        run_vertical_single_calibration(
            gaze_estimator, camera_index=cam, cd_d=cd, multi_pose=mp, multi_pose_d=mpd
        )
    elif args.calibration == "vertical-only":
        run_vertical_only_calibration(
            gaze_estimator, camera_index=cam, multi_pose=mp, multi_pose_d=mpd
        )


def _show_test_instructions(cap, sw, sh, duration=3):
    win = "Accuracy Test"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start = time.time()
    while time.time() - start < duration:
        ret, _ = cap.read()
        if not ret:
            continue
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        lines = [
            "ACCURACY TEST",
            "",
            "7 points will appear along the vertical center",
            "Look at each GREEN circle as it appears",
            "Keep your gaze steady during capture",
            "",
            "Starting soon...",
        ]
        y_off = sh // 4
        for i, text in enumerate(lines):
            if i == 0:
                fs, thick, color = 1.5, 3, (0, 255, 255)
            elif text == "":
                continue
            else:
                fs, thick, color = 1.0, 2, (255, 255, 255)
            sz, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            x = (sw - sz[0]) // 2
            y = y_off + i * 50
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, color, thick, cv2.LINE_AA)
        cv2.imshow(win, canvas)
        if cv2.waitKey(1) == 27:
            return


def _draw_progress_label(canvas, idx, total, sw):
    label = f"Point {idx + 1} / {total}"
    cv2.putText(canvas, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv2.LINE_AA)


def _run_accuracy_test(gaze_estimator, camera_index, capture_duration):
    """
    7-point vertical accuracy test.

    Points sit at x = screen_width // 2, y evenly spaced from 10% to 90%.
    Each point gets 1.0 s pulse then *capture_duration* seconds of capture.
    Returns a results dict, or None on user abort / no data.
    """
    sw, sh = get_screen_size()

    num_points = 7
    cx = sw // 2
    ys = [int(sh * (0.10 + i * (0.80 / (num_points - 1)))) for i in range(num_points)]
    test_points = [(cx, y) for y in ys]

    cap = cv2.VideoCapture(camera_index)
    _show_test_instructions(cap, sw, sh)

    win = "Accuracy Test"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    all_avg_preds = []
    all_targets = []

    for idx, (tx, ty) in enumerate(test_points):
        # --- Pulse phase (1.0 s) ---
        pulse_start = time.time()
        final_radius = 20
        while True:
            elapsed = time.time() - pulse_start
            if elapsed > 1.0:
                break
            ret, _ = cap.read()
            if not ret:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            radius = 15 + int(15 * abs(np.sin(2 * np.pi * elapsed)))
            final_radius = radius
            cv2.circle(canvas, (tx, ty), radius, (0, 255, 0), -1)
            cv2.circle(canvas, (tx, ty), 5, (0, 0, 0), -1)
            _draw_progress_label(canvas, idx, num_points, sw)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return None

        # --- Capture phase ---
        capture_start = time.time()
        point_preds = []
        while True:
            elapsed = time.time() - capture_start
            if elapsed > capture_duration:
                break
            ret, frame = cap.read()
            if not ret:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), final_radius, (0, 255, 0), -1)
            cv2.circle(canvas, (tx, ty), 5, (0, 0, 0), -1)
            t = elapsed / capture_duration
            ease = t * t * (3 - 2 * t)
            angle = 360 * (1 - ease)
            cv2.ellipse(canvas, (tx, ty), (40, 40), 0, -90, -90 + angle,
                        (255, 255, 255), 4)
            _draw_progress_label(canvas, idx, num_points, sw)
            cv2.imshow(win, canvas)
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return None

            features, blink = gaze_estimator.extract_features(frame)
            if features is not None and not blink:
                raw = gaze_estimator.predict(np.array([features]))[0]
                if getattr(gaze_estimator, "vertical_only", False):
                    pred = [gaze_estimator.vertical_center_x, float(raw)]
                else:
                    pred = raw
                point_preds.append(pred)

        if point_preds:
            all_avg_preds.append(np.mean(point_preds, axis=0))
            all_targets.append([tx, ty])

    cap.release()
    cv2.destroyAllWindows()

    if not all_avg_preds:
        return None

    predictions = np.array(all_avg_preds)
    targets = np.array(all_targets)
    errors = predictions - targets
    vertical_errors = errors[:, 1]
    horizontal_errors = errors[:, 0]

    return {
        "predictions": predictions,
        "targets": targets,
        "vertical_errors": vertical_errors,
        "horizontal_errors": horizontal_errors,
        "mean_vertical_error": float(np.mean(np.abs(vertical_errors))),
        "rmse_vertical": float(np.sqrt(np.mean(vertical_errors ** 2))),
        "std_vertical_error": float(np.std(vertical_errors)),
        "mean_horizontal_error": float(np.mean(np.abs(horizontal_errors))),
    }


def _print_results(results):
    print()
    print("=" * 50)
    print("ACCURACY TEST RESULTS")
    print("=" * 50)
    print(f"Mean Vertical Error:   {results['mean_vertical_error']:.1f} px")
    print(f"Vertical RMSE:         {results['rmse_vertical']:.1f} px")
    print(f"Std Dev Vertical:      {results['std_vertical_error']:.1f} px")
    print(f"Mean Horizontal Error: {results['mean_horizontal_error']:.1f} px")
    print("=" * 50)
    print()


def _display_results(results, sw, sh):
    """Visual: green target circles, red prediction circles, magenta connecting lines."""
    win = "Accuracy Results"
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    targets = results["targets"]
    predictions = results["predictions"]

    stats_lines = [
        f"Mean Vertical Error:   {results['mean_vertical_error']:.1f} px",
        f"Vertical RMSE:         {results['rmse_vertical']:.1f} px",
        f"Std Dev Vertical:      {results['std_vertical_error']:.1f} px",
        f"Mean Horizontal Error: {results['mean_horizontal_error']:.1f} px",
    ]

    while True:
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)

        # Title
        title = "ACCURACY TEST RESULTS"
        sz, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(canvas, title, ((sw - sz[0]) // 2, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)

        # Vertical center guide
        cv2.line(canvas, (sw // 2, 100), (sw // 2, sh - 100),
                 (80, 80, 80), 1, cv2.LINE_AA)

        # Draw target → prediction pairs
        for target, pred in zip(targets, predictions):
            tpt = tuple(target.astype(int))
            ppt = tuple(pred.astype(int))
            cv2.line(canvas, tpt, ppt, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, tpt, 14, (0, 255, 0), -1)
            cv2.circle(canvas, tpt, 17, (255, 255, 255), 2)
            cv2.circle(canvas, ppt, 8, (0, 0, 255), -1)

        # Stats (top-right)
        stats_x = sw - 560
        for i, line in enumerate(stats_lines):
            cv2.putText(canvas, line, (stats_x, 120 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Legend (bottom-right)
        legend = [
            ("Green  = target", (0, 255, 0)),
            ("Red    = prediction", (0, 0, 255)),
            ("Magenta = error", (255, 0, 255)),
        ]
        for i, (text, color) in enumerate(legend):
            cv2.putText(canvas, text, (stats_x, sh - 150 + i * 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

        # ESC hint
        hint = "Press ESC to close"
        sz, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, hint, ((sw - sz[0]) // 2, sh - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


def run_accuracy_demo():
    args = parse_args()

    gaze_estimator = GazeEstimator(
        model_name=args.model,
        landmark_alpha=args.landmark_alpha,
        include_face_position=args.multi_position,
    )

    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[accuracy_demo] Loaded model from {args.model_file}")
    else:
        _run_calibration(args, gaze_estimator)

    print("[accuracy_demo] Running 7-point accuracy test...")
    results = _run_accuracy_test(gaze_estimator, args.camera, args.capture_duration)

    if results is None:
        print("[accuracy_demo] Test aborted or no predictions collected.")
        return

    _print_results(results)

    sw, sh = get_screen_size()
    _display_results(results, sw, sh)


if __name__ == "__main__":
    run_accuracy_demo()
