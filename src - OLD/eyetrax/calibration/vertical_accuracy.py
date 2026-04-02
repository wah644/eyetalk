import time

import cv2
import numpy as np

from eyetrax.calibration.common import wait_for_face_and_countdown
from eyetrax.utils.screen import get_screen_size


def run_vertical_accuracy_test(gaze_estimator, camera_index: int = 0):
    """
    Tests vertical accuracy by displaying 7 points along the vertical center
    and measuring the error between predicted gaze and actual target positions.

    Returns:
        dict: Results containing errors, predictions, and targets
    """
    sw, sh = get_screen_size()
    
    # Create 7 points along vertical center with margins
    margin_ratio = 0.10
    my = int(sh * margin_ratio)
    gh = sh - 2 * my
    
    # 7 evenly spaced points along vertical axis
    num_points = 7
    step_y = gh / (num_points - 1)
    
    vertical_points = [(sw // 2, my + int(i * step_y)) for i in range(num_points)]
    
    cap = cv2.VideoCapture(camera_index)
    
    # Wait for face detection and countdown
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    # Show instructions
    _show_instructions(cap, sw, sh)
    
    # Collect gaze predictions for each point
    results = _measure_accuracy(gaze_estimator, cap, vertical_points, sw, sh)
    
    cap.release()
    
    if results is None:
        cv2.destroyAllWindows()
        return None
    
    # Display results
    _display_results(results, sw, sh)
    
    cv2.destroyAllWindows()
    return results


def _show_instructions(cap, sw, sh, duration=3):
    """Show instruction screen before starting the test"""
    cv2.namedWindow("Vertical Accuracy Test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Vertical Accuracy Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, _ = cap.read()
        if not ret:
            continue
            
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        
        # Instructions text
        instructions = [
            "VERTICAL ACCURACY TEST",
            "",
            "Look at each GREEN circle as it appears",
            "Keep your gaze steady on the circle",
            "The test will begin in a moment...",
        ]
        
        y_offset = sh // 3
        for i, text in enumerate(instructions):
            font_scale = 1.5 if i == 0 else 1.0
            thickness = 3 if i == 0 else 2
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            
            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            x = (sw - size[0]) // 2
            y = y_offset + i * 60
            
            cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, color, thickness, cv2.LINE_AA)
        
        cv2.imshow("Vertical Accuracy Test", canvas)
        if cv2.waitKey(1) == 27:
            return False
    
    return True


def _measure_accuracy(gaze_estimator, cap, points, sw, sh, 
                      pulse_duration=1.0, capture_duration=2.0):
    """
    Display each point and collect gaze predictions
    """
    cv2.namedWindow("Vertical Accuracy Test", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Vertical Accuracy Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    all_predictions = []
    all_targets = []
    
    for point_idx, (x, y) in enumerate(points):
        # Pulse phase - get attention
        pulse_start = time.time()
        final_radius = 20
        
        while True:
            elapsed = time.time() - pulse_start
            if elapsed > pulse_duration:
                break
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            
            # Pulsing circle
            radius = 15 + int(15 * abs(np.sin(2 * np.pi * elapsed)))
            final_radius = radius
            cv2.circle(canvas, (x, y), radius, (0, 255, 0), -1)
            # Add center dot
            cv2.circle(canvas, (x, y), 5, (0, 0, 0), -1)
            
            # Show progress
            progress_text = f"Point {point_idx + 1} / {len(points)}"
            cv2.putText(canvas, progress_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.imshow("Vertical Accuracy Test", canvas)
            if cv2.waitKey(1) == 27:
                return None
        
        # Capture phase - collect gaze data
        capture_start = time.time()
        point_predictions = []
        
        while True:
            elapsed = time.time() - capture_start
            if elapsed > capture_duration:
                break
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            
            # Static circle
            cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
            # Add center dot
            cv2.circle(canvas, (x, y), 5, (0, 0, 0), -1)
            
            # Progress indicator (countdown ring)
            t = elapsed / capture_duration
            ease = t * t * (3 - 2 * t)
            angle = 360 * (1 - ease)
            cv2.ellipse(canvas, (x, y), (40, 40), 0, -90, -90 + angle, 
                       (255, 255, 255), 4)
            
            # Show progress
            progress_text = f"Point {point_idx + 1} / {len(points)}"
            cv2.putText(canvas, progress_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            cv2.imshow("Vertical Accuracy Test", canvas)
            if cv2.waitKey(1) == 27:
                return None
            
            # Extract features and predict
            features, blink = gaze_estimator.extract_features(frame)
            if features is not None and not blink:
                prediction = gaze_estimator.predict(np.array([features]))[0]
                point_predictions.append(prediction)
        
        # Average predictions for this point
        if point_predictions:
            avg_prediction = np.mean(point_predictions, axis=0)
            all_predictions.append(avg_prediction)
            all_targets.append([x, y])
    
    if not all_predictions:
        return None
    
    # Calculate errors
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    errors = predictions - targets
    vertical_errors = errors[:, 1]  # Only Y-axis errors
    horizontal_errors = errors[:, 0]  # X-axis errors for reference
    
    results = {
        'predictions': predictions,
        'targets': targets,
        'vertical_errors': vertical_errors,
        'horizontal_errors': horizontal_errors,
        'mean_vertical_error': np.mean(np.abs(vertical_errors)),
        'std_vertical_error': np.std(vertical_errors),
        'mean_horizontal_error': np.mean(np.abs(horizontal_errors)),
        'rmse_vertical': np.sqrt(np.mean(vertical_errors ** 2)),
    }
    
    return results


def _display_results(results, sw, sh):
    """
    Display accuracy test results
    """
    cv2.namedWindow("Vertical Accuracy Results", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Vertical Accuracy Results", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
        
        # Title
        title = "VERTICAL ACCURACY TEST RESULTS"
        size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        x = (sw - size[0]) // 2
        cv2.putText(canvas, title, (x, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 255, 255), 3, cv2.LINE_AA)
        
        # Draw visualization
        margin = 200
        viz_width = sw - 2 * margin
        viz_height = sh - 2 * margin
        viz_x = margin
        viz_y = margin
        
        # Draw reference line (vertical center)
        center_x = sw // 2
        cv2.line(canvas, (center_x, viz_y), (center_x, viz_y + viz_height), 
                (100, 100, 100), 2, cv2.LINE_AA)
        
        # Draw targets and predictions
        targets = results['targets']
        predictions = results['predictions']
        
        for i, (target, pred) in enumerate(zip(targets, predictions)):
            # Target point (green)
            cv2.circle(canvas, tuple(target.astype(int)), 12, (0, 255, 0), -1)
            cv2.circle(canvas, tuple(target.astype(int)), 15, (255, 255, 255), 2)
            
            # Prediction point (red)
            cv2.circle(canvas, tuple(pred.astype(int)), 8, (0, 0, 255), -1)
            
            # Line connecting target to prediction
            cv2.line(canvas, tuple(target.astype(int)), tuple(pred.astype(int)), 
                    (255, 0, 255), 2, cv2.LINE_AA)
        
        # Statistics
        stats_y = 100
        stats_x = 50
        line_height = 40
        
        stats = [
            f"Mean Vertical Error: {results['mean_vertical_error']:.1f} pixels",
            f"Vertical RMSE: {results['rmse_vertical']:.1f} pixels",
            f"Std Dev (Vertical): {results['std_vertical_error']:.1f} pixels",
            f"Mean Horizontal Error: {results['mean_horizontal_error']:.1f} pixels (reference)",
            "",
            "Green circles = Target positions",
            "Red circles = Your gaze predictions",
            "Lines show the error",
        ]
        
        for i, stat in enumerate(stats):
            color = (255, 255, 255) if i < 4 else (150, 150, 150)
            font_scale = 0.9 if i < 4 else 0.7
            cv2.putText(canvas, stat, (stats_x, stats_y + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
        
        # Instructions to close
        close_text = "Press ESC to close"
        size, _ = cv2.getTextSize(close_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x = (sw - size[0]) // 2
        cv2.putText(canvas, close_text, (x, sh - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("Vertical Accuracy Results", canvas)
        if cv2.waitKey(1) == 27:
            break
    
    # Print to console as well
    print("\n" + "="*50)
    print("VERTICAL ACCURACY TEST RESULTS")
    print("="*50)
    for stat in stats[:4]:
        print(stat)
    print("="*50 + "\n")