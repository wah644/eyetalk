import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
    run_vertical_enhanced_calibration,  # NEW - Added vertical calibration
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor, draw_scan_path, make_thumbnail
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, fullscreen, iter_frames


# ============================================================
#                   KEYBOARD CONFIGURATION
# ============================================================

KEYBOARD_KEYS = [
    "ABCD",
    "EFGH",
    "IJKL",
    "MNOP",
    "QRSTU",
    "VWXYZ",
    "SPACE"
]

# Special actions (can be added as separate buttons if needed)
SPECIAL_KEYS = ["DELETE", "ACCEPT"]

# Map letters to key indices
LETTER_TO_KEY = {}
for idx, key in enumerate(KEYBOARD_KEYS[:-1]):  # Exclude SPACE key
    for letter in key.lower():
        LETTER_TO_KEY[letter] = idx

DWELL_TIME = 1.2  # seconds to dwell on a key to select it
DWELL_FEEDBACK_CIRCLE_MAX = 30  # max radius for dwell feedback circle
OUTPUT_FILE = "keyboard_output.txt"

# Load dictionary from words.txt file
def load_dictionary(filepath="words.txt"):
    """Load words from a text file (one word per line)"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = [line.strip().lower() for line in f if line.strip()]
        print(f"[Dictionary] Loaded {len(words)} words from {filepath}")
        return words
    except FileNotFoundError:
        print(f"[Dictionary] Warning: {filepath} not found, using fallback dictionary")
        # Fallback to small dictionary if file not found
        return [
            "hello", "hi", "hey", "help", "how", "are", "you", "i", "am",
            "the", "this", "that", "what", "when", "where", "who", "why",
            "yes", "no", "ok", "thanks", "please", "sorry"
        ]

DICTIONARY = load_dictionary()


# ============================================================
#                   TRIE IMPLEMENTATION
# ============================================================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.word = None


class T9Trie:
    """Trie structure for T9-style word prediction"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        """Insert a word into the Trie"""
        node = self.root
        word = word.lower()
        
        # Convert word to key sequence
        key_sequence = []
        for char in word:
            if char in LETTER_TO_KEY:
                key_sequence.append(LETTER_TO_KEY[char])
        
        # Build trie path
        for key_idx in key_sequence:
            if key_idx not in node.children:
                node.children[key_idx] = TrieNode()
            node = node.children[key_idx]
        
        node.is_end_of_word = True
        node.word = word
    
    def search_predictions(self, key_sequence, max_results=5):
        """Find all words matching the key sequence"""
        if not key_sequence:
            return []
        
        # Navigate to the node for this sequence
        node = self.root
        for key_idx in key_sequence:
            if key_idx not in node.children:
                return []
            node = node.children[key_idx]
        
        # Collect all words from this node
        predictions = []
        self._collect_words(node, predictions, max_results)
        return predictions
    
    def _collect_words(self, node, predictions, max_results):
        """Recursively collect all words from a node"""
        if len(predictions) >= max_results:
            return
        
        if node.is_end_of_word:
            predictions.append(node.word)
        
        for child in node.children.values():
            self._collect_words(child, predictions, max_results)
            if len(predictions) >= max_results:
                return


# ============================================================
#                   KEYBOARD CONTROLLER
# ============================================================

class KeyboardController:
    """Handles vertical keyboard layout and dwell-based selection with T9 prediction"""
    
    def __init__(self, screen_width, screen_height, keys=KEYBOARD_KEYS, dwell_time=DWELL_TIME):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.keys = keys
        self.num_keys = len(keys)
        self.dwell_time = dwell_time
        
        # Keyboard layout with gaps
        self.keyboard_width = 300  # pixels
        self.keyboard_x = (screen_width - self.keyboard_width) // 2  # Center horizontally
        self.key_gap = 10  # Gap between keys in pixels
        self.available_height = screen_height - (self.num_keys - 1) * self.key_gap
        self.key_height = self.available_height // self.num_keys
        
        # Dwell state
        self.current_key = None
        self.dwell_start_time = None
        self.typed_text = ""
        
        # T9 prediction
        self.trie = T9Trie()
        for word in DICTIONARY:
            self.trie.insert(word)
        
        self.current_key_sequence = []
        self.current_predictions = []
        self.selected_prediction_idx = 0
        
        # Initialize output file
        Path(OUTPUT_FILE).write_text("")
    
    def get_key_bounds(self, key_index):
        """Get the bounding box for a key"""
        y1 = key_index * (self.key_height + self.key_gap)
        y2 = y1 + self.key_height
        x1 = self.keyboard_x
        x2 = self.keyboard_x + self.keyboard_width
        return x1, y1, x2, y2
    
    def get_hovered_key(self, x, y):
        """Determine which key (if any) is being hovered over"""
        # Check if x is within keyboard bounds
        if x < self.keyboard_x or x > self.keyboard_x + self.keyboard_width:
            return None
        
        # Account for gaps when determining key
        for key_index in range(self.num_keys):
            x1, y1, x2, y2 = self.get_key_bounds(key_index)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return key_index
        return None
    
    def update(self, gaze_x, gaze_y):
        """Update dwell state based on gaze position"""
        if gaze_x is None or gaze_y is None:
            self.current_key = None
            self.dwell_start_time = None
            return None
        
        hovered_key = self.get_hovered_key(gaze_x, gaze_y)
        
        # Reset if moved to different key or no key
        if hovered_key != self.current_key:
            self.current_key = hovered_key
            self.dwell_start_time = time.time() if hovered_key is not None else None
            return None
        
        # Check if dwelling long enough
        if self.current_key is not None and self.dwell_start_time is not None:
            dwell_duration = time.time() - self.dwell_start_time
            if dwell_duration >= self.dwell_time:
                # Key selected!
                selected_key = self.current_key
                self.process_key_selection(selected_key)
                # Reset dwell state
                self.current_key = None
                self.dwell_start_time = None
                return selected_key
        
        return None
    
    def process_key_selection(self, key_index):
        """Process the selection of a key"""
        key_label = self.keys[key_index]
        
        if key_label == "SPACE":
            # Accept top prediction if available (incomplete word), otherwise add space
            if self.current_predictions:
                word = self.current_predictions[0]  # Always use top prediction
                self.typed_text += word + " "
                with open(OUTPUT_FILE, "a") as f:
                    f.write(word + " ")
            else:
                self.typed_text += " "
                with open(OUTPUT_FILE, "a") as f:
                    f.write(" ")
            
            # Reset prediction state
            self.current_key_sequence = []
            self.current_predictions = []
            self.selected_prediction_idx = 0
            
        elif key_label == "DELETE":
            if self.current_key_sequence:
                # Delete last key in sequence
                self.current_key_sequence.pop()
                self._update_predictions()
            elif self.typed_text:
                # Delete last character
                self.typed_text = self.typed_text[:-1]
                with open(OUTPUT_FILE, "w") as f:
                    f.write(self.typed_text)
        
        elif key_label == "ACCEPT":
            # Accept current prediction
            if self.current_predictions:
                word = self.current_predictions[self.selected_prediction_idx]
                self.typed_text += word + " "
                with open(OUTPUT_FILE, "a") as f:
                    f.write(word + " ")
                
                # Reset prediction state
                self.current_key_sequence = []
                self.current_predictions = []
                self.selected_prediction_idx = 0
        
        else:
            # Letter key - add to sequence
            if key_index < len(KEYBOARD_KEYS) - 1:  # Only letter keys (exclude SPACE)
                self.current_key_sequence.append(key_index)
                self._update_predictions()
    
    def _update_predictions(self):
        """Update word predictions based on current key sequence"""
        self.current_predictions = self.trie.search_predictions(
            self.current_key_sequence, 
            max_results=5
        )
        self.selected_prediction_idx = 0
    
    def handle_backspace(self):
        """Handle backspace gesture (looking above screen)"""
        if self.current_key_sequence:
            # Delete last key in sequence
            self.current_key_sequence.pop()
            self._update_predictions()
        elif self.typed_text:
            # Delete last character
            self.typed_text = self.typed_text[:-1]
            with open(OUTPUT_FILE, "w") as f:
                f.write(self.typed_text)
    
    def get_dwell_progress(self):
        """Get current dwell progress (0.0 to 1.0)"""
        if self.current_key is None or self.dwell_start_time is None:
            return 0.0
        
        dwell_duration = time.time() - self.dwell_start_time
        return min(dwell_duration / self.dwell_time, 1.0)
    
    def draw(self, canvas, gaze_x=None, gaze_y=None):
        """Draw the vertical keyboard on the canvas"""
        dwell_progress = self.get_dwell_progress()
        
        for i, key_label in enumerate(self.keys):
            x1, y1, x2, y2 = self.get_key_bounds(i)
            
            # Determine color and thickness
            color = (255, 255, 255)  # White default
            thickness = 2
            
            # Highlight if hovering
            if i == self.current_key:
                color = (0, 255, 0)  # Green
                thickness = 4
            
            # Draw key rectangle with semi-transparent background
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
            
            # Draw key border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            
            # Draw key label
            font_scale = 1.2 if len(key_label) <= 5 else 0.9
            (text_w, text_h), _ = cv2.getTextSize(
                key_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            text_x = x1 + (x2 - x1 - text_w) // 2
            text_y = y1 + (y2 - y1 + text_h) // 2
            cv2.putText(
                canvas, key_label, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA
            )
            
            # Draw dwell progress indicator
            if i == self.current_key and dwell_progress > 0:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = int(DWELL_FEEDBACK_CIRCLE_MAX * dwell_progress)
                
                # Pulsing circle
                cv2.circle(canvas, (center_x, center_y), radius, (0, 255, 255), 3)
                
                # Progress arc
                angle = int(360 * dwell_progress)
                cv2.ellipse(
                    canvas, (center_x, center_y), (radius + 10, radius + 10),
                    -90, 0, angle, (0, 255, 0), 5
                )
        
        # Draw predictions panel (left side)
        if self.current_predictions:
            panel_width = 400
            panel_x = 50
            panel_y = 200
            
            # Draw panel background
            overlay = canvas.copy()
            cv2.rectangle(
                overlay, 
                (panel_x, panel_y), 
                (panel_x + panel_width, panel_y + 60 + len(self.current_predictions) * 50),
                (30, 30, 30), 
                -1
            )
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
            
            # Draw title
            cv2.putText(
                canvas, "Predictions:",
                (panel_x + 10, panel_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA
            )
            
            # Draw predictions
            for idx, word in enumerate(self.current_predictions):
                y = panel_y + 60 + idx * 50
                
                # Highlight selected prediction
                if idx == self.selected_prediction_idx:
                    cv2.rectangle(
                        canvas,
                        (panel_x + 5, y - 30),
                        (panel_x + panel_width - 5, y + 10),
                        (0, 255, 0),
                        2
                    )
                
                cv2.putText(
                    canvas, f"{idx + 1}. {word}",
                    (panel_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
                )
        
        # Draw current key sequence
        if self.current_key_sequence:
            seq_text = "Keys: " + "-".join([self.keys[k] for k in self.current_key_sequence])
            cv2.putText(
                canvas, seq_text,
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA
            )
        
        # Draw typed text on the left side
        text_x = 50
        text_y = self.screen_height - 100
        
        # Background box for typed text
        box_height = 80
        box_y = text_y - 60
        cv2.rectangle(canvas, (0, box_y), (self.keyboard_x - 20, box_y + box_height), (0, 0, 0), -1)
        cv2.rectangle(canvas, (0, box_y), (self.keyboard_x - 20, box_y + box_height), (100, 100, 100), 2)
        
        # Show current prediction inline with top prediction
        display_text = self.typed_text
        if self.current_predictions:
            display_text += "[" + self.current_predictions[0] + "]"  # Show top prediction
        
        # Word wrap if text is too long
        max_width = self.keyboard_x - 70
        cv2.putText(
            canvas, display_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
        )


# ============================================================
#                   MENU SYSTEM
# ============================================================

# Menu state tracking
menu_options = ["keyboard", "fixed_phrases", "home_iot"]
menu_dwell_times = [None, None, None]
menu_dwell_start = [None, None, None]
MENU_DWELL_TIME = 1.5
MENU_BUTTON_WIDTH = 400  # Width of menu buttons (centered)
MENU_BUTTON_GAP = 20  # Gap between menu buttons

def get_menu_button_bounds(option_index, screen_width, screen_height):
    """Get the bounding box for a menu button"""
    available_height = screen_height - (2 * MENU_BUTTON_GAP)  # 2 gaps between 3 buttons
    button_height = available_height // 3
    
    y1 = option_index * (button_height + MENU_BUTTON_GAP)
    y2 = y1 + button_height
    
    # Center buttons horizontally
    x1 = (screen_width - MENU_BUTTON_WIDTH) // 2
    x2 = x1 + MENU_BUTTON_WIDTH
    
    return x1, y1, x2, y2

def get_hovered_menu_option(x, y, screen_width, screen_height):
    """Determine which menu option is being hovered over"""
    if x is None or y is None:
        return None
    
    # Check each button's bounds
    for i in range(3):
        x1, y1, x2, y2 = get_menu_button_bounds(i, screen_width, screen_height)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

def update_menu(gaze_x, gaze_y, screen_width, screen_height):
    """Update menu dwell state and return selected option if any"""
    global menu_dwell_start
    
    hovered = get_hovered_menu_option(gaze_x, gaze_y, screen_width, screen_height)
    
    # Reset non-hovered options
    for i in range(3):
        if i != hovered:
            menu_dwell_start[i] = None
    
    if hovered is None:
        return None
    
    # Start or continue dwelling
    if menu_dwell_start[hovered] is None:
        menu_dwell_start[hovered] = time.time()
    
    dwell_duration = time.time() - menu_dwell_start[hovered]
    if dwell_duration >= MENU_DWELL_TIME:
        menu_dwell_start[hovered] = None  # Reset
        return menu_options[hovered]
    
    return None

def draw_menu(canvas, screen_width, screen_height, gaze_x=None, gaze_y=None):
    """Draw the main menu with three centered buttons with gaps"""
    hovered = get_hovered_menu_option(gaze_x, gaze_y, screen_width, screen_height)
    
    menu_labels = ["KEYBOARD", "FIXED PHRASES", "HOME IoT"]
    menu_colors = [(0, 200, 0), (100, 100, 100), (100, 100, 100)]  # Green for keyboard, gray for others
    
    for i, label in enumerate(menu_labels):
        x1, y1, x2, y2 = get_menu_button_bounds(i, screen_width, screen_height)
        
        # Button background with rounded effect
        is_hovered = (i == hovered)
        bg_color = (80, 80, 80) if is_hovered else (40, 40, 40)
        
        # Draw semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)
        
        # Button border
        border_color = (255, 255, 255) if is_hovered else (100, 100, 100)
        border_thickness = 4 if is_hovered else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, border_thickness)
        
        # Dwell progress indicator
        if is_hovered and menu_dwell_start[i] is not None:
            dwell_duration = time.time() - menu_dwell_start[i]
            progress = min(dwell_duration / MENU_DWELL_TIME, 1.0)
            
            # Draw progress circle in center of button
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            radius = 50
            angle = int(360 * progress)
            cv2.ellipse(
                canvas,
                (center_x, center_y - 60),
                (radius, radius),
                0,
                -90,
                -90 + angle,
                (0, 255, 0),
                8
            )
        
        # Button text
        text_color = menu_colors[i]
        font_scale = 1.5
        thickness = 3
        
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = (x1 + x2 - size[0]) // 2
        text_y = (y1 + y2) // 2 + size[1] // 2
        
        cv2.putText(
            canvas, label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA
        )
        
        # Status text for disabled options
        if i in [1, 2]:  # Fixed phrases and Home IoT
            status_text = "(Coming Soon)"
            status_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            status_x = (x1 + x2 - status_size[0]) // 2
            status_y = text_y + 50
            cv2.putText(
                canvas, status_text,
                (status_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (150, 150, 150),
                2,
                cv2.LINE_AA
            )
    
    # Instructions at bottom
    instructions = "Look ABOVE screen for BACKSPACE | Look BELOW screen for HOME"
    inst_size, _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    inst_x = (screen_width - inst_size[0]) // 2
    cv2.putText(
        canvas, instructions,
        (inst_x, screen_height - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA
    )


def run_demo():
    """Main demo function with menu and integrated keyboard"""
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence
    scan_path_enabled = args.scan_path
    scan_path_max = args.scan_path_max
    scan_path_log = args.scan_path_log

    gaze_estimator = GazeEstimator(model_name=args.model)

    # Load or calibrate model
    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[demo] Loaded gaze model from {args.model_file}")
    else:
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index)
        elif calibration_method == "vertical":  # NEW - Added vertical calibration
            run_vertical_enhanced_calibration(gaze_estimator, camera_index=camera_index)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    screen_width, screen_height = get_screen_size()

    # Initialize smoother
    if filter_method == "kalman":
        kalman = make_kalman()
        smoother = KalmanSmoother(kalman)
        smoother.tune(gaze_estimator, camera_index=camera_index)
    elif filter_method == "kde":
        kalman = None
        smoother = KDESmoother(screen_width, screen_height, confidence=confidence_level)
    else:
        kalman = None
        smoother = NoSmoother()

    # Initialize keyboard controller
    keyboard = KeyboardController(screen_width, screen_height)
    
    # Mode tracking: "menu", "keyboard", "fixed_phrases", "home_iot"
    current_mode = "menu"

    # Setup background
    if background_path and os.path.isfile(background_path):
        background = cv2.imread(background_path)
        background = cv2.resize(background, (screen_width, screen_height))
    else:
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        background[:] = (50, 50, 50)

    cam_width, cam_height = 320, 240
    BORDER = 2
    MARGIN = 20
    cursor_alpha = 0.0
    cursor_step = 0.05

    # Initialize scan path tracking
    scan_path_points = deque(maxlen=scan_path_max if scan_path_enabled else 0)
    scan_path_timestamps = deque(maxlen=scan_path_max if scan_path_enabled else 0)
    
    # Gesture detection state
    gesture_above_start = None  # Track when started looking above
    gesture_below_start = None  # Track when started looking below
    GESTURE_BACKSPACE_TIME = 2.0  # 2 seconds to trigger backspace
    GESTURE_HOME_TIME = 5.0  # 5 seconds to trigger home

    def save_scan_path_log():
        """Save scan path to CSV file"""
        if not scan_path_enabled or not scan_path_points:
            return
        
        if scan_path_log:
            log_path = Path(scan_path_log)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(f"scan_path_{timestamp}.csv")
        
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, "w") as f:
            f.write("timestamp,x,y\n")
            for (x, y), ts in zip(scan_path_points, scan_path_timestamps):
                f.write(f"{ts:.6f},{x},{y}\n")
        
        print(f"[demo] Scan path saved to {log_path} ({len(scan_path_points)} points)")

    # Create window and position it on the second display (portrait monitor to the right)
    cv2.namedWindow("Gaze Keyboard", cv2.WINDOW_NORMAL)
    
    # Position window on second display
    # Assuming main display width is around 1920-2560 pixels, position at x=2000
    # Adjust this value if your main display has a different width
    cv2.moveWindow("Gaze Keyboard", 2000, 0)
    
    # Set to fullscreen
    cv2.setWindowProperty("Gaze Keyboard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    with camera(camera_index) as cap:
        prev_time = time.time()
        start_time = time.time()

        for frame in iter_frames(cap):
            features, blink_detected = gaze_estimator.extract_features(frame)

            if features is not None and not blink_detected:
                gaze_point = gaze_estimator.predict(np.array([features]))[0]
                x, y = map(int, gaze_point)
                x_pred, y_pred = smoother.step(x, y)
                contours = smoother.debug.get("contours", [])
                cursor_alpha = min(cursor_alpha + cursor_step, 1.0)
                
                # Add point to scan path if enabled
                if scan_path_enabled and x_pred is not None and y_pred is not None:
                    scan_path_points.append((int(x_pred), int(y_pred)))
                    scan_path_timestamps.append(time.time() - start_time)
            else:
                x_pred = y_pred = None
                blink_detected = True
                contours = []
                cursor_alpha = max(cursor_alpha - cursor_step, 0.0)

            # Check for gesture zones with dwell time (backspace = above, home = below)
            if x_pred is not None and y_pred is not None:
                if y_pred < -40:  # Above screen - backspace gesture
                    if gesture_above_start is None:
                        gesture_above_start = time.time()
                    else:
                        dwell_duration = time.time() - gesture_above_start
                        if dwell_duration >= GESTURE_BACKSPACE_TIME:
                            if current_mode == "keyboard":
                                keyboard.handle_backspace()
                                print("[gesture] Backspace triggered")
                            gesture_above_start = None  # Reset
                    gesture_below_start = None  # Reset other gesture
                    
                elif y_pred > screen_height + 40:  # Below screen - home gesture
                    if gesture_below_start is None:
                        gesture_below_start = time.time()
                    else:
                        dwell_duration = time.time() - gesture_below_start
                        if dwell_duration >= GESTURE_HOME_TIME:
                            current_mode = "menu"
                            print("[gesture] Home triggered")
                            gesture_below_start = None  # Reset
                    gesture_above_start = None  # Reset other gesture
                    
                else:
                    # Not in any gesture zone - reset both
                    gesture_above_start = None
                    gesture_below_start = None
            else:
                # No gaze detected - reset gestures
                gesture_above_start = None
                gesture_below_start = None

            # Update based on current mode
            if current_mode == "menu":
                selected_option = update_menu(x_pred, y_pred, screen_width, screen_height)
                if selected_option:
                    current_mode = selected_option
                    print(f"[menu] Selected: {selected_option}")
            elif current_mode == "keyboard":
                selected_key = keyboard.update(x_pred, y_pred)
                if selected_key is not None:
                    print(f"[keyboard] Key selected: {keyboard.keys[selected_key]}")

            # Draw on canvas
            canvas = background.copy()

            if filter_method == "kde" and contours:
                cv2.drawContours(canvas, contours, -1, (15, 182, 242), 5)

            # Draw scan path if enabled
            if scan_path_enabled and scan_path_points:
                draw_scan_path(
                    canvas,
                    list(scan_path_points),
                    color=(0, 255, 0),
                    thickness=2,
                    fade_alpha=True,
                    max_points=scan_path_max,
                )

            # Draw based on mode
            if current_mode == "menu":
                draw_menu(canvas, screen_width, screen_height, x_pred, y_pred)
            elif current_mode == "keyboard":
                keyboard.draw(canvas, x_pred, y_pred)
            
            # Draw gesture progress indicators
            if gesture_above_start is not None:
                # Backspace gesture progress
                progress = (time.time() - gesture_above_start) / GESTURE_BACKSPACE_TIME
                progress = min(progress, 1.0)
                
                # Progress bar at top
                bar_width = 300
                bar_height = 20
                bar_x = (screen_width - bar_width) // 2
                bar_y = 20
                
                # Background
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # Progress
                progress_width = int(bar_width * progress)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (255, 100, 0), -1)
                # Border
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                
                # Label
                cv2.putText(canvas, "BACKSPACE", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2, cv2.LINE_AA)
            
            if gesture_below_start is not None:
                # Home gesture progress
                progress = (time.time() - gesture_below_start) / GESTURE_HOME_TIME
                progress = min(progress, 1.0)
                
                # Progress bar at bottom
                bar_width = 300
                bar_height = 20
                bar_x = (screen_width - bar_width) // 2
                bar_y = screen_height - 40
                
                # Background
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # Progress
                progress_width = int(bar_width * progress)
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 150, 255), -1)
                # Border
                cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                
                # Label
                cv2.putText(canvas, "HOME", (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2, cv2.LINE_AA)

            # Cursor is disabled (invisible)
            # if x_pred is not None and y_pred is not None and cursor_alpha > 0:
            #     draw_cursor(canvas, x_pred, y_pred, cursor_alpha)

            # Draw camera thumbnail
            thumb = make_thumbnail(frame, size=(cam_width, cam_height), border=BORDER)
            h, w = thumb.shape[:2]
            canvas[-h - MARGIN : -MARGIN, -w - MARGIN : -MARGIN] = thumb

            # Draw FPS and status
            now = time.time()
            fps = 1 / (now - prev_time)
            prev_time = now

            cv2.putText(
                canvas,
                f"FPS: {int(fps)}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            
            status_text = "Blinking" if blink_detected else "Tracking"
            status_color = (0, 0, 255) if blink_detected else (0, 255, 0)
            cv2.putText(
                canvas,
                status_text,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Gaze Keyboard", canvas)
            if cv2.waitKey(1) == 27:  # ESC key
                break
        
        # Save scan path log on exit
        save_scan_path_log()
        
        print(f"\n[demo] Final typed text: {keyboard.typed_text}")
        print(f"[demo] Output saved to: {OUTPUT_FILE}")
    
    # Cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()