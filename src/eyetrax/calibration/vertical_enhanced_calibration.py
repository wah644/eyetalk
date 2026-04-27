import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size


def run_vertical_enhanced_calibration(gaze_estimator, camera_index: int = 0,
                                      multi_pose: bool = False, multi_pose_d: float = 1.0,
                                      train: bool = True, single_column: bool = False):
    """
    Enhanced calibration with vertical points:
    - Original 5 points (center + 4 corners)
    - If single_column=False (default): 18 additional vertical points (9 × 2 lines near center) = 23 total
    - If single_column=True: 9 additional vertical points along the center line = 14 total
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return None

    # Define margin for the grid
    margin_ratio = 0.10
    mx, my = int(sw * margin_ratio), int(sh * margin_ratio)
    gw, gh = sw - 2 * mx, sh - 2 * my

    # Original 5-point grid positions (in grid coordinates)
    original_5_points = [
        (1, 1),    # center
        (0, 0),    # top-left
        (2, 0),    # top-right
        (0, 2),    # bottom-left
        (2, 2),    # bottom-right
    ]
    
    # 9 vertical points × 2 lines (left and right of center)
    # Lines separated by ~150 pixels (about half keyboard width)
    vertical_margin_ratio = 0.05  # Tighter margins for more coverage
    v_my = int(sh * vertical_margin_ratio)
    v_gh = sh - 2 * v_my
    
    # Create 9 evenly spaced Y positions
    num_vertical_points = 9
    step_y = v_gh / (num_vertical_points - 1)
    
    # Convert original 5 points to pixels using the standard grid system
    original_5_pixels = compute_grid_points(original_5_points, sw, sh)

    if single_column:
        # Single center line: 9 points at x = center
        vertical_points = [(sw // 2, v_my + int(i * step_y)) for i in range(num_vertical_points)]
        # Combine: original 5 + center line 9 = 14 total points
        all_points = original_5_pixels + vertical_points
    else:
        # Horizontal offset from center (half keyboard width ≈ 75 pixels on each side)
        h_offset = 75

        # Left line (center - offset)
        left_line_points = [(sw // 2 - h_offset, v_my + int(i * step_y)) for i in range(num_vertical_points)]

        # Right line (center + offset)
        right_line_points = [(sw // 2 + h_offset, v_my + int(i * step_y)) for i in range(num_vertical_points)]

        # Combine: original 5 + left line 9 + right line 9 = 23 total points
        all_points = original_5_pixels + left_line_points + right_line_points
    
    # Show instruction screen
    _show_enhanced_instructions(cap, sw, sh, len(all_points))
    
    # Collect data for all 23 points
    res = _pulse_and_capture(gaze_estimator, cap, all_points, sw, sh,
                             multi_pose=multi_pose, multi_pose_d=multi_pose_d)
    cap.release()
    cv2.destroyAllWindows()
    
    if res is None:
        return None
    
    feats, targs = res
    if not feats:
        return None

    X, y = np.array(feats), np.array(targs)
    print(f"[Vertical Enhanced Calibration] Collected {len(feats)} samples from {len(all_points)} points")
    if train:
        gaze_estimator.train(X, y)
        print("[Vertical Enhanced Calibration] Training complete!")
        print("[Calibration] Look ABOVE the screen for BACKSPACE (2 sec)")
        print("[Calibration] Look BELOW the screen for HOME (5 sec)")
        return None
    return X, y


def run_vertical_single_calibration(gaze_estimator, camera_index: int = 0,
                                    cd_d: float = 1.0,
                                    multi_pose: bool = False, multi_pose_d: float = 1.0,
                                    train: bool = True):
    """
    Single vertical line calibration with 14 points:
    - Standard 5 corner/center points at 10% margin
    - 9 points along a single vertical center line (x = screen_width // 2),
      evenly spaced from 5% to 95% of screen height
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return None

    original_5_points = [
        (1, 1),  # center
        (0, 0),  # top-left
        (2, 0),  # top-right
        (0, 2),  # bottom-left
        (2, 2),  # bottom-right
    ]
    corner_pixels = compute_grid_points(original_5_points, sw, sh, margin_ratio=0.10)

    num_vertical = 9
    center_x = sw // 2
    ys = [int(sh * (0.05 + i * (0.90 / (num_vertical - 1)))) for i in range(num_vertical)]
    center_line_points = [(center_x, y) for y in ys]

    all_points = corner_pixels + center_line_points

    _show_single_instructions(cap, sw, sh, len(all_points))

    res = _pulse_and_capture(gaze_estimator, cap, all_points, sw, sh,
                             cd_d=cd_d,
                             multi_pose=multi_pose, multi_pose_d=multi_pose_d)
    cap.release()
    cv2.destroyAllWindows()

    if res is None:
        return None

    feats, targs = res
    if not feats:
        return None

    X, y = np.array(feats), np.array(targs)
    print(f"[Vertical Single Calibration] Collected {len(feats)} samples from {len(all_points)} points")
    if train:
        gaze_estimator.train(X, y)
        print("[Vertical Single Calibration] Training complete!")
        return None
    return X, y


def _capture_offscreen_point(gaze_estimator, cap, sw, sh, direction: str,
                              target_y: int, cd_d: float = 1.5,
                              multi_pose: bool = False, multi_pose_d: float = 1.0):
    """Capture gaze samples for a point that is beyond the screen edge.

    A large arrow and instruction text are shown at the top or bottom of the
    screen.  The dot is placed at the nearest visible edge but the training
    TARGET is the off-screen *target_y* so the model learns what looking
    beyond the screen looks like.

    direction: "above" or "below"
    """
    import time as _time
    center_x = sw // 2
    is_above  = (direction == "above")
    dot_y_vis = 18 if is_above else sh - 18   # dot at screen edge (visible)
    arrow_tip_y   = 10  if is_above else sh - 10
    arrow_base_y  = 80  if is_above else sh - 80
    instr = "LOOK ABOVE THE SCREEN" if is_above else "LOOK BELOW THE SCREEN"
    arrow_col = (0, 220, 255)
    dot_col_pulse   = (0, 255, 0)
    dot_col_capture = (0, 0, 255)

    feats, targs = [], []

    def _draw_frame(dot_col):
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        # Arrow pointing off-screen
        pts_arr = np.array([
            [center_x,          arrow_tip_y],
            [center_x - 40,     arrow_base_y],
            [center_x - 15,     arrow_base_y],
            [center_x - 15,     arrow_base_y + (40 if not is_above else -40)],
            [center_x + 15,     arrow_base_y + (40 if not is_above else -40)],
            [center_x + 15,     arrow_base_y],
            [center_x + 40,     arrow_base_y],
        ], np.int32)
        cv2.fillPoly(canvas, [pts_arr], arrow_col)
        # Instruction text
        fs, thick = 1.0, 2
        (tw, th), _ = cv2.getTextSize(instr, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
        ty = (sh // 2 - 20) if is_above else (sh // 2 + 30)
        cv2.putText(canvas, instr, ((sw - tw) // 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thick, cv2.LINE_AA)
        # Edge dot
        cv2.circle(canvas, (center_x, dot_y_vis), 8, dot_col, -1)
        return canvas

    # Pulse phase (no data)
    ps = _time.time()
    while _time.time() - ps < 1.0:
        ok, _ = cap.read()
        if not ok:
            continue
        cv2.imshow("Calibration", _draw_frame(dot_col_pulse))
        if cv2.waitKey(1) == 27:
            return None

    # Capture phase
    cs = _time.time()
    while _time.time() - cs < cd_d:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("Calibration", _draw_frame(dot_col_capture))
        if cv2.waitKey(1) == 27:
            return None
        ft, blink = gaze_estimator.extract_features(frame)
        if ft is not None and not blink:
            feats.append(ft)
            targs.append(target_y)

    # Multi-pose phase
    if multi_pose:
        ms = _time.time()
        orange = (0, 165, 255)
        while _time.time() - ms < multi_pose_d:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow("Calibration", _draw_frame(orange))
            if cv2.waitKey(1) == 27:
                return None
            ft, blink = gaze_estimator.extract_features(frame)
            if ft is not None and not blink:
                feats.append(ft)
                targs.append(target_y)

    return feats, targs


def run_vertical_only_calibration(gaze_estimator, camera_index: int = 0,
                                  cd_d: float = 1.0,
                                  multi_pose: bool = False, multi_pose_d: float = 1.0,
                                  train: bool = True):
    """
    Vertical-only calibration: 9 points along the center vertical line +
    2 off-screen anchor points (above and below) to teach the model what
    extreme gaze positions look like for trie/bigram/backspace gestures.
    Trains a 1-D model that predicts Y (screen height) exclusively.
    gaze_estimator.vertical_only is set to True so the runtime substitutes
    screen_width // 2 for the X coordinate at prediction time.
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return None

    num_vertical = 9
    center_x = sw // 2
    ys = [int(sh * (0.05 + i * (0.90 / (num_vertical - 1)))) for i in range(num_vertical)]
    points = [(center_x, y) for y in ys]

    _show_vertical_only_instructions(cap, sw, sh, len(points) + 2)

    # Regular 9-point capture
    res = _pulse_and_capture(gaze_estimator, cap, points, sw, sh,
                             cd_d=cd_d,
                             multi_pose=multi_pose, multi_pose_d=multi_pose_d)
    if res is None:
        cap.release()
        cv2.destroyAllWindows()
        return None

    feats, targs_2d = res

    # Off-screen above: target Y = -12% of screen height
    above_target = -int(sh * 0.12)
    res_above = _capture_offscreen_point(gaze_estimator, cap, sw, sh,
                                         direction="above", target_y=above_target,
                                         cd_d=cd_d, multi_pose=multi_pose,
                                         multi_pose_d=multi_pose_d)
    if res_above is None:
        cap.release()
        cv2.destroyAllWindows()
        return None
    feats.extend(res_above[0])
    targs_2d.extend([[center_x, t] for t in res_above[1]])

    # Off-screen below: target Y = screen_height + 12%
    below_target = sh + int(sh * 0.12)
    res_below = _capture_offscreen_point(gaze_estimator, cap, sw, sh,
                                         direction="below", target_y=below_target,
                                         cd_d=cd_d, multi_pose=multi_pose,
                                         multi_pose_d=multi_pose_d)
    if res_below is None:
        cap.release()
        cv2.destroyAllWindows()
        return None
    feats.extend(res_below[0])
    targs_2d.extend([[center_x, t] for t in res_below[1]])

    cap.release()
    cv2.destroyAllWindows()

    if not feats:
        return None

    X = np.array(feats)
    y = np.array([t[1] for t in targs_2d])   # Y-only targets (1-D)
    print(f"[Vertical Only Calibration] Collected {len(feats)} samples from {len(points) + 2} points")
    if train:
        gaze_estimator.train(X, y)
        gaze_estimator.vertical_only = True
        gaze_estimator.vertical_center_x = center_x
        print("[Vertical Only Calibration] Training complete (Y-axis only)!")
        return None
    return X, y


def run_vertical_center_calibration(gaze_estimator, camera_index: int = 0,
                                    cd_d: float = 1.0,
                                    multi_pose: bool = False, multi_pose_d: float = 1.0,
                                    train: bool = True):
    """
    9-point center vertical calibration — all points on-screen.
    Points are evenly spaced from 5% to 95% of screen height along the
    centre x column. No off-screen anchors. Trains the standard 2-D model.
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return None

    num_points = 9
    center_x = sw // 2
    ys = [int(sh * (0.05 + i * (0.90 / (num_points - 1)))) for i in range(num_points)]
    points = [(center_x, y) for y in ys]

    _show_vertical_center_instructions(cap, sw, sh, num_points)

    res = _pulse_and_capture(gaze_estimator, cap, points, sw, sh,
                             cd_d=cd_d,
                             multi_pose=multi_pose, multi_pose_d=multi_pose_d)
    cap.release()
    cv2.destroyAllWindows()

    if res is None:
        return None

    feats, targs = res
    if not feats:
        return None

    X, y = np.array(feats), np.array(targs)
    print(f"[Vertical Center Calibration] Collected {len(feats)} samples from {num_points} points")
    if train:
        gaze_estimator.train(X, y)
        print("[Vertical Center Calibration] Training complete!")
        return None
    return X, y


def _show_vertical_center_instructions(cap, sw, sh, num_points, duration=3):
    """Show instruction screen before starting the vertical-center calibration."""
    import time
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, _ = cap.read()
        if not ret:
            continue

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        instructions = [
            "VERTICAL CENTER CALIBRATION",
            "",
            f"You will see {num_points} calibration points",
            "All points are on the centre vertical column",
            "All points are within the screen",
            "Look at each GREEN circle as it appears",
            "Keep your gaze steady during data collection",
            "",
            "Starting soon...",
        ]

        y_offset = sh // 4
        for i, text in enumerate(instructions):
            if i == 0:
                font_scale, thickness, color = 1.5, 3, (0, 255, 255)
            elif text == "":
                continue
            else:
                font_scale, thickness, color = 1.0, 2, (255, 255, 255)

            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (sw - size[0]) // 2
            y = y_offset + i * 50
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False

    return True


def _show_vertical_only_instructions(cap, sw, sh, num_points, duration=3):
    """Show instruction screen before starting the vertical-only calibration."""
    import time
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, _ = cap.read()
        if not ret:
            continue

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        instructions = [
            "VERTICAL-ONLY CALIBRATION",
            "",
            f"You will see {num_points} calibration points",
            "All points are on the centre vertical line",
            "Look at each GREEN circle as it appears",
            "Keep your gaze steady during data collection",
            "",
            "Predicts vertical position only (Y-axis)",
            "Starting soon...",
        ]

        y_offset = sh // 4
        for i, text in enumerate(instructions):
            if i == 0:
                font_scale, thickness, color = 1.5, 3, (0, 255, 255)
            elif text == "":
                continue
            else:
                font_scale, thickness, color = 1.0, 2, (255, 255, 255)

            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (sw - size[0]) // 2
            y = y_offset + i * 50
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False

    return True


def _show_single_instructions(cap, sw, sh, num_points, duration=3):
    """Show instruction screen before starting the single vertical line calibration"""
    import time
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, _ = cap.read()
        if not ret:
            continue

        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        instructions = [
            "VERTICAL SINGLE LINE CALIBRATION",
            "",
            f"You will see {num_points} calibration points",
            "Look at each GREEN circle as it appears",
            "Keep your gaze steady during data collection",
            "",
            "Includes center vertical line for accuracy",
            "Starting soon...",
        ]

        y_offset = sh // 4
        for i, text in enumerate(instructions):
            if i == 0:
                font_scale, thickness, color = 1.5, 3, (0, 255, 255)
            elif text == "":
                continue
            else:
                font_scale, thickness, color = 1.0, 2, (255, 255, 255)

            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (sw - size[0]) // 2
            y = y_offset + i * 50
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False

    return True


def _show_enhanced_instructions(cap, sw, sh, num_points, duration=3):
    """Show instruction screen before starting the enhanced calibration"""
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    import time
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, _ = cap.read()
        if not ret:
            continue
            
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        
        # Instructions text
        instructions = [
            "ENHANCED VERTICAL CALIBRATION",
            "",
            f"You will see {num_points} calibration points",
            "Look at each GREEN circle as it appears",
            "Keep your gaze steady during data collection",
            "",
            "Includes dual vertical lines for better accuracy",
            "Starting soon...",
        ]
        
        y_offset = sh // 4
        for i, text in enumerate(instructions):
            if i == 0:
                font_scale = 1.5
                thickness = 3
                color = (0, 255, 255)
            elif text == "":
                continue
            else:
                font_scale = 1.0
                thickness = 2
                color = (255, 255, 255)
            
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (sw - size[0]) // 2
            y = y_offset + i * 50
            
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow("Calibration", canvas)
        if cv2.waitKey(1) == 27:
            return False
    
    return True