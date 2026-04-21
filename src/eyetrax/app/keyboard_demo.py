import sys
import os
import time
import shutil
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import cv2
import numpy as np

from eyetrax.app.keyword_to_sentence import (
    generate_sentence_async,
    get_pending_result,
    is_generating,
)
from eyetrax.integrations.adb_emergency import (
    EmergencyCallConfig,
    ensure_device,
    get_phone_screen_size,
    open_url,
    tap,
    trigger_emergency_call,
)
from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
    run_multi_position_calibration,
    run_vertical_enhanced_calibration,
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

# Word selection panel constants
SELECTION_ZONE_DWELL_TIME = 2.0   # Seconds looking left/right to trigger selection mode
LEFT_ZONE_RATIO  = 0.20           # Left 20% of screen → trie selection panel
RIGHT_ZONE_RATIO = 0.80           # Right 20% of screen → ngram selection panel
WORD_PANEL_DWELL_TIME = 1.5       # Seconds to dwell at center to select a word
# Display order top-to-bottom: 5th-best, 3rd-best, 1st-best (center), 2nd-best, 4th-best
WORD_PANEL_ORDER = [4, 2, 0, 1, 3]

# Load dictionary from words.txt file
def load_dictionary(filepath=None):
    """Load words from a text file (one word per line)"""
    if filepath is None:
        filepath = Path(__file__).parent / "words.txt"
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
        """Find all words matching the key sequence (limited to top 5)"""
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
#                   BIGRAM MODEL
# ============================================================

def load_bigrams(filepath=None):
    """Load bigrams from a text file with format: 'word1 word2    frequency'"""
    if filepath is None:
        filepath = Path(__file__).parent / "Bigram.txt"
    bigrams: dict[str, list[tuple[str, int]]] = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                word1, word2 = parts[0].lower(), parts[1].lower()
                try:
                    freq = int(parts[2])
                except ValueError:
                    continue
                bigrams.setdefault(word1, []).append((word2, freq))
        for word in bigrams:
            bigrams[word].sort(key=lambda x: x[1], reverse=True)
        count = sum(len(v) for v in bigrams.values())
        print(f"[Bigram] Loaded {count} bigram entries from {filepath}")
    except FileNotFoundError:
        print(f"[Bigram] Warning: {filepath} not found, bigram predictions unavailable")
    return bigrams


class BigramModel:
    """Bigram model for next-word prediction loaded from bigram.txt."""

    def __init__(self):
        self.bigrams = load_bigrams()

    def get_predictions(self, current_word: str, max_results: int = 5) -> list[str]:
        """Return top next-word predictions for *current_word*."""
        if not current_word:
            return []
        word = current_word.lower().strip()
        if word not in self.bigrams:
            return []
        return [w for w, _ in self.bigrams[word][:max_results]]


# ============================================================
#               WORD SELECTION PANEL
# ============================================================

class WordSelectionPanel:
    """Scrollable vertical word-selection panel (replaces keyboard in selection mode).

    Words are arranged top-to-bottom in the order defined by WORD_PANEL_ORDER
    (5th-best, 3rd-best, 1st-best at centre, 2nd-best, 4th-best).  The user
    scrolls by looking above/below screen-centre; the further from centre, the
    faster the scroll.  Dwelling at screen-centre for WORD_PANEL_DWELL_TIME
    seconds selects the word currently nearest the centre.
    """

    def __init__(self, screen_width: int, screen_height: int, panel_type: str = "trie"):
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.panel_type    = panel_type  # "trie" or "ngram"

        # Panel geometry (centred horizontally)
        self.panel_width = 360
        self.panel_x     = (screen_width - self.panel_width) // 2

        # Word layout
        self.word_spacing    = 155   # px between word-button centres
        self.max_word_height = 115   # button height when at screen centre
        self.min_word_height = 28    # floor for very distant words

        # Scroll state  (0.0 → best word at screen centre)
        self.scroll_offset: float = 0.0
        self.words: list[str] = []

        # Dwell / selection
        self.dwell_start:      float | None = None
        self.dwell_target_idx: int   | None = None
        self.dwell_time = WORD_PANEL_DWELL_TIME

        # Scroll parameters – 6-section model:
        #   top 2 sections    (y < H/3)        → scroll up
        #   middle 2 sections (H/3 <= y <= 2H/3) → selection / dwell
        #   bottom 2 sections (y > 2H/3)       → scroll down
        self.scroll_threshold = screen_height // 6   # distance from centre to scroll boundary
        self.max_scroll_speed = 2.8  # display-positions per second at screen edge

        self._last_t = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_words(self, words: list[str]) -> None:
        """Load up to 5 words and reset scroll/dwell state."""
        self.words         = list(words[:5])
        self.scroll_offset = 0.0
        self.dwell_start      = None
        self.dwell_target_idx = None
        self._last_t          = time.time()

    def update(self, gaze_x, gaze_y) -> str | None:
        """Advance scroll and dwell.  Returns selected word string or None."""
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now

        if not self.words or gaze_y is None:
            return None

        screen_cy = self.screen_height // 2
        delta     = gaze_y - screen_cy

        # ---- scrolling --------------------------------------------------
        if abs(delta) > self.scroll_threshold:
            effective = abs(delta) - self.scroll_threshold
            half_h    = max(screen_cy - self.scroll_threshold, 1)
            speed     = self.max_scroll_speed * (effective / half_h)
            self.scroll_offset += speed * dt * (1 if delta > 0 else -1)

            # Clamp so a valid display position always sits near centre
            entries   = self._display_entries()
            if entries:
                min_off = min(p for p, _, _ in entries) - 2.0
                max_off = max(p for p, _, _ in entries) - 2.0
                self.scroll_offset = max(min_off, min(max_off, self.scroll_offset))

        # ---- find word closest to screen centre -------------------------
        entries         = self._display_entries()
        center_word_idx = None
        min_dist        = float("inf")
        for dp, wi, _ in entries:
            yc   = self._word_yc(dp)
            dist = abs(yc - screen_cy)
            if dist < min_dist:
                min_dist        = dist
                center_word_idx = wi

        # ---- dwell (only in the middle 2/6 sections) --------------------
        dwell_zone = self.scroll_threshold   # same boundary as scroll → clean split
        if abs(delta) <= dwell_zone and center_word_idx is not None:
            if self.dwell_target_idx != center_word_idx:
                self.dwell_target_idx = center_word_idx
                self.dwell_start      = now
            elif self.dwell_start is not None:
                if now - self.dwell_start >= self.dwell_time:
                    selected = self.words[self.dwell_target_idx]
                    self.dwell_start      = None
                    self.dwell_target_idx = None
                    return selected
        else:
            self.dwell_target_idx = None
            self.dwell_start      = None

        return None

    def get_dwell_progress(self) -> float:
        if self.dwell_start is None:
            return 0.0
        return min((time.time() - self.dwell_start) / self.dwell_time, 1.0)

    def draw(self, canvas: np.ndarray) -> None:
        if not self.words:
            return

        entries   = self._display_entries()
        screen_cy = self.screen_height // 2
        dp_prog   = self.get_dwell_progress()

        # ---- draw 6-section zone indicators -----------------------------
        line_top = self.screen_height // 3        # boundary between scroll-up and select
        line_bot = (self.screen_height * 2) // 3  # boundary between select and scroll-down

        # Subtle tint for scroll zones (top and bottom)
        zone_overlay = canvas.copy()
        cv2.rectangle(zone_overlay, (0, 0), (self.screen_width, line_top),
                      (20, 20, 60), -1)
        cv2.rectangle(zone_overlay, (0, line_bot), (self.screen_width, self.screen_height),
                      (20, 20, 60), -1)
        cv2.addWeighted(zone_overlay, 0.25, canvas, 0.75, 0, canvas)

        # Faint dashed boundary lines
        dash_len, gap_len = 30, 18
        x = 0
        while x < self.screen_width:
            x2 = min(x + dash_len, self.screen_width)
            cv2.line(canvas, (x, line_top), (x2, line_top), (100, 110, 220), 1)
            cv2.line(canvas, (x, line_bot), (x2, line_bot), (100, 110, 220), 1)
            x += dash_len + gap_len

        # Zone labels
        cv2.putText(canvas, "LOOK HERE  to scroll up",
                    (self.screen_width // 2 - 180, line_top // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 110, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "LOOK HERE  to scroll down",
                    (self.screen_width // 2 - 195, (line_bot + self.screen_height) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 110, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "LOOK HERE  to select",
                    (self.screen_width // 2 - 160, screen_cy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 130), 2, cv2.LINE_AA)

        for dp, wi, word in entries:
            yc = self._word_yc(dp)
            h  = self._word_h(yc)

            y1 = yc - h // 2
            y2 = yc + h // 2
            x1 = self.panel_x
            x2 = self.panel_x + self.panel_width

            if y2 < 0 or y1 > self.screen_height:
                continue

            y1c = max(0, y1)
            y2c = min(self.screen_height - 1, y2)

            is_dwelled = (wi == self.dwell_target_idx)
            is_center  = abs(yc - screen_cy) < 25

            if is_dwelled:
                bg, border, thick = (50, 90, 50),  (0, 255, 60),   4
            elif is_center:
                bg, border, thick = (50, 50, 95),  (80, 180, 255), 3
            else:
                bg, border, thick = (22, 22, 42),  (65, 65, 105),  2

            if y1c < y2c:
                overlay = canvas.copy()
                cv2.rectangle(overlay, (x1, y1c), (x2, y2c), bg, -1)
                cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
                cv2.rectangle(canvas, (x1, y1c), (x2, y2c), border, thick)

            # Text – scale with button height
            fs = max(0.45, min(1.65, h / 68.0))
            (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            tx = x1 + (self.panel_width - tw) // 2
            ty = max(y1c + th + 4, min(y2c - 4, yc + th // 2))
            cv2.putText(canvas, word, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

            # Dwell arc
            if is_dwelled and dp_prog > 0:
                cx     = (x1 + x2) // 2
                cy_cir = max(10, min(self.screen_height - 10, yc))
                radius = max(8, h // 3)
                angle  = int(360 * dp_prog)
                cv2.ellipse(canvas, (cx, cy_cir), (radius, radius),
                            -90, 0, angle, (0, 255, 100), 4)

        # Header
        label = "TRIE PREDICTIONS" if self.panel_type == "trie" else "NEXT WORD (BIGRAM)"
        col   = (255, 200, 50) if self.panel_type == "trie" else (200, 100, 255)
        cv2.putText(canvas, label,
                    (self.panel_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Look up/down to scroll  |  Pause at centre to select",
                    (self.panel_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (155, 155, 155), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _display_entries(self) -> list[tuple[int, int, str]]:
        """Return (display_pos, word_index, word) for every valid entry."""
        n = len(self.words)
        return [
            (dp, wi, self.words[wi])
            for dp, wi in enumerate(WORD_PANEL_ORDER)
            if wi < n
        ]

    def _word_yc(self, display_pos: int) -> int:
        """Y screen coordinate of word centre at *display_pos* given current scroll."""
        screen_cy = self.screen_height // 2
        return screen_cy + int((display_pos - 2 - self.scroll_offset) * self.word_spacing)

    def _word_h(self, yc: int) -> int:
        """Button height as a function of vertical distance from screen centre."""
        screen_cy = self.screen_height // 2
        dist      = abs(yc - screen_cy) / self.word_spacing   # in word-spacing units
        h         = int(self.max_word_height * ((2 / 3) ** dist))
        return max(h, self.min_word_height)


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
        self.current_predictions: list[str] = []
        self.current_ngram_predictions: list[str] = []
        self.selected_prediction_idx = 0

        # Bigram model
        self.bigram = BigramModel()

        # Word selection panels
        self.trie_panel  = WordSelectionPanel(screen_width, screen_height, "trie")
        self.ngram_panel = WordSelectionPanel(screen_width, screen_height, "ngram")

        # Sub-mode: "typing" | "trie_select" | "ngram_select"
        self.submode = "typing"

        # Zone detection for triggering selection mode
        self.left_zone_start:  float | None = None
        self.right_zone_start: float | None = None
        self.zone_dwell_time = SELECTION_ZONE_DWELL_TIME
        self.left_zone_x  = int(screen_width * LEFT_ZONE_RATIO)
        self.right_zone_x = int(screen_width * RIGHT_ZONE_RATIO)

        # Initialize output file
        Path(OUTPUT_FILE).write_text("")

    # ------------------------------------------------------------------
    # Key bounds / hover helpers (unchanged)
    # ------------------------------------------------------------------

    def get_key_bounds(self, key_index):
        """Get the bounding box for a key"""
        y1 = key_index * (self.key_height + self.key_gap)
        y2 = y1 + self.key_height
        x1 = self.keyboard_x
        x2 = self.keyboard_x + self.keyboard_width
        return x1, y1, x2, y2

    def get_hovered_key(self, x, y):
        """Determine which key (if any) is being hovered over"""
        if x < self.keyboard_x or x > self.keyboard_x + self.keyboard_width:
            return None

        for key_index in range(self.num_keys):
            x1, y1, x2, y2 = self.get_key_bounds(key_index)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return key_index
        return None

    # ------------------------------------------------------------------
    # Main update – routes to correct sub-mode
    # ------------------------------------------------------------------

    def update(self, gaze_x, gaze_y):
        """Update state based on current sub-mode."""

        # ---- trie selection mode ----------------------------------------
        if self.submode == "trie_select":
            selected = self.trie_panel.update(gaze_x, gaze_y)
            if selected is not None:
                self._accept_word(selected)
                self.submode = "typing"
            return None

        # ---- ngram selection mode ----------------------------------------
        if self.submode == "ngram_select":
            selected = self.ngram_panel.update(gaze_x, gaze_y)
            if selected is not None:
                # Accept top trie word + ngram word together
                if self.current_predictions:
                    top_word = self.current_predictions[0]
                    self._accept_word(top_word + " " + selected)
                else:
                    self._accept_word(selected)
                self.submode = "typing"
            return None

        # ---- typing mode: zone-trigger detection ------------------------
        if gaze_x is not None:
            if gaze_x < self.left_zone_x:
                self.right_zone_start = None
                if self.left_zone_start is None:
                    self.left_zone_start = time.time()
                elif time.time() - self.left_zone_start >= self.zone_dwell_time:
                    # Enter trie selection only when predictions exist
                    if self.current_predictions:
                        self.trie_panel.set_words(self.current_predictions)
                        self.submode = "trie_select"
                    self.left_zone_start = None
            elif gaze_x > self.right_zone_x:
                self.left_zone_start = None
                if self.right_zone_start is None:
                    self.right_zone_start = time.time()
                elif time.time() - self.right_zone_start >= self.zone_dwell_time:
                    # Enter ngram selection using top trie prediction as input
                    if self.current_predictions:
                        top_word   = self.current_predictions[0]
                        ngram_pred = self.bigram.get_predictions(top_word)
                        if ngram_pred:
                            self.ngram_panel.set_words(ngram_pred)
                            self.submode = "ngram_select"
                    self.right_zone_start = None
            else:
                # Inside keyboard zone – reset zone timers
                self.left_zone_start  = None
                self.right_zone_start = None

        return self._update_keyboard(gaze_x, gaze_y)

    # ------------------------------------------------------------------
    # Core keyboard dwell logic
    # ------------------------------------------------------------------

    def _update_keyboard(self, gaze_x, gaze_y):
        """Dwell-based keyboard key selection (original update logic)."""
        if gaze_x is None or gaze_y is None:
            self.current_key = None
            self.dwell_start_time = None
            return None

        hovered_key = self.get_hovered_key(gaze_x, gaze_y)

        if hovered_key != self.current_key:
            self.current_key = hovered_key
            self.dwell_start_time = time.time() if hovered_key is not None else None
            return None

        if self.current_key is not None and self.dwell_start_time is not None:
            dwell_duration = time.time() - self.dwell_start_time
            if dwell_duration >= self.dwell_time:
                selected_key = self.current_key
                self.process_key_selection(selected_key)
                self.current_key = None
                self.dwell_start_time = None
                return selected_key

        return None

    def process_key_selection(self, key_index):
        """Process the selection of a key"""
        key_label = self.keys[key_index]

        if key_label == "SPACE":
            # Accept top prediction if available, otherwise add space
            if self.current_predictions:
                word = self.current_predictions[0]
                self._accept_word(word)
            else:
                self.typed_text += " "
                with open(OUTPUT_FILE, "a") as f:
                    f.write(" ")

        elif key_label == "DELETE":
            if self.current_key_sequence:
                self.current_key_sequence.pop()
                self._update_predictions()
            elif self.typed_text:
                self.typed_text = self.typed_text[:-1]
                with open(OUTPUT_FILE, "w") as f:
                    f.write(self.typed_text)

        elif key_label == "ACCEPT":
            if self.current_predictions:
                word = self.current_predictions[self.selected_prediction_idx]
                self._accept_word(word)

        else:
            # Letter key
            if key_index < len(KEYBOARD_KEYS) - 1:
                self.current_key_sequence.append(key_index)
                self._update_predictions()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _accept_word(self, word_or_phrase: str) -> None:
        """Append word/phrase + space to typed text and reset prediction state."""
        text = word_or_phrase + " "
        self.typed_text += text
        with open(OUTPUT_FILE, "a") as f:
            f.write(text)
        self.current_key_sequence      = []
        self.current_predictions       = []
        self.current_ngram_predictions = []
        self.selected_prediction_idx   = 0

    def _update_predictions(self):
        """Update word predictions based on current key sequence (max 5)."""
        self.current_predictions = self.trie.search_predictions(
            self.current_key_sequence, max_results=5
        )
        self.selected_prediction_idx = 0
        if self.current_predictions:
            self.current_ngram_predictions = self.bigram.get_predictions(self.current_predictions[0])
        else:
            self.current_ngram_predictions = []

    def handle_backspace(self):
        """Handle backspace gesture (looking above screen)"""
        if self.current_key_sequence:
            self.current_key_sequence.pop()
            self._update_predictions()
        elif self.typed_text:
            self.typed_text = self.typed_text[:-1]
            with open(OUTPUT_FILE, "w") as f:
                f.write(self.typed_text)

    def get_dwell_progress(self):
        """Get current dwell progress (0.0 to 1.0)"""
        if self.current_key is None or self.dwell_start_time is None:
            return 0.0
        dwell_duration = time.time() - self.dwell_start_time
        return min(dwell_duration / self.dwell_time, 1.0)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, canvas, gaze_x=None, gaze_y=None):
        """Draw the keyboard or the active word-selection panel."""

        if self.submode == "trie_select":
            self.trie_panel.draw(canvas)
            self._draw_typed_text(canvas)
            return

        if self.submode == "ngram_select":
            self.ngram_panel.draw(canvas)
            self._draw_typed_text(canvas)
            if self.current_predictions:
                hint = f"Next word after: \"{self.current_predictions[0]}\""
                cv2.putText(canvas, hint,
                            (self.ngram_panel.panel_x, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 150, 255), 2, cv2.LINE_AA)
            return

        # ---- normal keyboard drawing ------------------------------------
        self._draw_keyboard(canvas)
        self._draw_typed_text(canvas)
        self._draw_zone_progress(canvas, gaze_x, gaze_y)

    def _draw_keyboard(self, canvas):
        """Draw the vertical keyboard keys."""
        dwell_progress = self.get_dwell_progress()

        for i, key_label in enumerate(self.keys):
            x1, y1, x2, y2 = self.get_key_bounds(i)

            color     = (255, 255, 255)
            thickness = 2
            if i == self.current_key:
                color     = (0, 255, 0)
                thickness = 4

            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            font_scale = 1.2 if len(key_label) <= 5 else 0.9
            (text_w, text_h), _ = cv2.getTextSize(
                key_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            text_x = x1 + (x2 - x1 - text_w) // 2
            text_y = y1 + (y2 - y1 + text_h) // 2
            cv2.putText(canvas, key_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

            if i == self.current_key and dwell_progress > 0:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius   = int(DWELL_FEEDBACK_CIRCLE_MAX * dwell_progress)
                cv2.circle(canvas, (center_x, center_y), radius, (0, 255, 255), 3)
                angle = int(360 * dwell_progress)
                cv2.ellipse(canvas, (center_x, center_y), (radius + 10, radius + 10),
                            -90, 0, angle, (0, 255, 0), 5)

        # Predictions panel (left side)
        if self.current_predictions:
            panel_width = 400
            panel_x     = 50
            panel_y     = 200

            overlay = canvas.copy()
            cv2.rectangle(
                overlay,
                (panel_x, panel_y),
                (panel_x + panel_width,
                 panel_y + 60 + len(self.current_predictions) * 50),
                (30, 30, 30), -1
            )
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

            cv2.putText(canvas, "Predictions:",
                        (panel_x + 10, panel_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

            for idx, word in enumerate(self.current_predictions):
                y = panel_y + 60 + idx * 50
                if idx == self.selected_prediction_idx:
                    cv2.rectangle(canvas,
                                  (panel_x + 5, y - 30),
                                  (panel_x + panel_width - 5, y + 10),
                                  (0, 255, 0), 2)
                cv2.putText(canvas, f"{idx + 1}. {word}",
                            (panel_x + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Ngram predictions panel (right side)
        if self.current_ngram_predictions:
            panel_width  = 400
            panel_x_right = self.screen_width - 50 - panel_width
            panel_y      = 200

            overlay = canvas.copy()
            cv2.rectangle(
                overlay,
                (panel_x_right, panel_y),
                (panel_x_right + panel_width,
                 panel_y + 60 + len(self.current_ngram_predictions) * 50),
                (30, 10, 40), -1
            )
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

            cv2.putText(canvas, "Next Word:",
                        (panel_x_right + 10, panel_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 100, 255), 2, cv2.LINE_AA)

            for idx, word in enumerate(self.current_ngram_predictions):
                y = panel_y + 60 + idx * 50
                cv2.putText(canvas, f"{idx + 1}. {word}",
                            (panel_x_right + 15, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 255), 2, cv2.LINE_AA)

        # Key sequence display
        if self.current_key_sequence:
            seq_text = "Keys: " + "-".join([self.keys[k] for k in self.current_key_sequence])
            cv2.putText(canvas, seq_text, (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)

        # Zone hint labels (static, always visible)
        if self.current_predictions:
            # Left label – stacked so it fits in the narrow zone strip
            cv2.putText(canvas, "<",
                        (10, self.screen_height // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, "WORD",
                        (10, self.screen_height // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "SELECT",
                        (10, self.screen_height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2, cv2.LINE_AA)
            # Right label
            cv2.putText(canvas, ">",
                        (self.right_zone_x + 10, self.screen_height // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 100, 200), 3, cv2.LINE_AA)
            cv2.putText(canvas, "NEXT",
                        (self.right_zone_x + 10, self.screen_height // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 100, 200), 2, cv2.LINE_AA)
            cv2.putText(canvas, "WORD",
                        (self.right_zone_x + 10, self.screen_height // 2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 100, 200), 2, cv2.LINE_AA)

    def _draw_typed_text(self, canvas):
        """Draw the typed-text area at the bottom of the canvas."""
        text_x   = 50
        text_y   = self.screen_height - 100
        box_h    = 80
        box_y    = text_y - 60

        cv2.rectangle(canvas, (0, box_y), (self.screen_width, box_y + box_h), (0, 0, 0), -1)
        cv2.rectangle(canvas, (0, box_y), (self.screen_width, box_y + box_h), (100, 100, 100), 2)

        display_text = self.typed_text
        if self.current_predictions:
            display_text += "[" + self.current_predictions[0] + "]"

        # Show last portion to avoid text running off screen
        cv2.putText(canvas, display_text[-70:], (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    def _draw_zone_progress(self, canvas, gaze_x, gaze_y):
        """Draw left/right zone dwell progress bars when predictions exist."""
        if not self.current_predictions:
            return

        cy = self.screen_height // 2

        if self.left_zone_start is not None:
            progress = min((time.time() - self.left_zone_start) / self.zone_dwell_time, 1.0)
            bar_w = int(self.left_zone_x * progress)
            bar_h = 8
            cv2.rectangle(canvas,
                          (0, cy - bar_h // 2), (bar_w, cy + bar_h // 2),
                          (0, 200, 255), -1)

        if self.right_zone_start is not None:
            progress = min((time.time() - self.right_zone_start) / self.zone_dwell_time, 1.0)
            bar_w = int((self.screen_width - self.right_zone_x) * progress)
            bar_h = 8
            cv2.rectangle(canvas,
                          (self.screen_width - bar_w, cy - bar_h // 2),
                          (self.screen_width,          cy + bar_h // 2),
                          (255, 100, 200), -1)


# ============================================================
#                   MENU SYSTEM
# ============================================================

# Menu state tracking
menu_options = ["keyboard", "fixed_phrases", "home_iot", "emergency_call"]
menu_dwell_times = [None] * len(menu_options)
menu_dwell_start = [None] * len(menu_options)
MENU_DWELL_TIME = 1.5
MENU_BUTTON_WIDTH = 400  # Width of menu buttons (centered)
MENU_BUTTON_GAP = 20  # Gap between menu buttons

CONFIRM_DWELL_TIME = 1.2
CONFIRM_BUTTON_WIDTH = 460
CONFIRM_BUTTON_HEIGHT = 140
confirm_options = ["yes", "no"]
confirm_dwell_start = [None] * len(confirm_options)

def get_menu_button_bounds(option_index, screen_width, screen_height):
    """Get the bounding box for a menu button"""
    num_buttons = len(menu_options)
    available_height = screen_height - ((num_buttons - 1) * MENU_BUTTON_GAP)
    button_height = available_height // max(num_buttons, 1)

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
    for i in range(len(menu_options)):
        x1, y1, x2, y2 = get_menu_button_bounds(i, screen_width, screen_height)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None

def update_menu(gaze_x, gaze_y, screen_width, screen_height):
    """Update menu dwell state and return selected option if any"""
    global menu_dwell_start

    hovered = get_hovered_menu_option(gaze_x, gaze_y, screen_width, screen_height)

    # Reset non-hovered options
    for i in range(len(menu_options)):
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
    """Draw the main menu with centered buttons with gaps"""
    hovered = get_hovered_menu_option(gaze_x, gaze_y, screen_width, screen_height)

    menu_labels = ["KEYBOARD", "FIXED PHRASES", "HOME IoT", "EMERGENCY CALL"]
    menu_colors = [(0, 200, 0), (100, 100, 100), (100, 100, 100), (0, 0, 255)]

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


def _confirm_button_bounds(option_index: int, screen_width: int, screen_height: int):
    # Two buttons stacked vertically, centered.
    gap = 30
    total_h = (CONFIRM_BUTTON_HEIGHT * 2) + gap
    top_y = (screen_height - total_h) // 2 + 90
    x1 = (screen_width - CONFIRM_BUTTON_WIDTH) // 2
    x2 = x1 + CONFIRM_BUTTON_WIDTH
    y1 = top_y + option_index * (CONFIRM_BUTTON_HEIGHT + gap)
    y2 = y1 + CONFIRM_BUTTON_HEIGHT
    return x1, y1, x2, y2


def _get_hovered_confirm_option(x, y, screen_width, screen_height):
    if x is None or y is None:
        return None
    for i in range(len(confirm_options)):
        x1, y1, x2, y2 = _confirm_button_bounds(i, screen_width, screen_height)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None


def update_emergency_confirm(gaze_x, gaze_y, screen_width, screen_height):
    global confirm_dwell_start
    hovered = _get_hovered_confirm_option(gaze_x, gaze_y, screen_width, screen_height)
    for i in range(len(confirm_options)):
        if i != hovered:
            confirm_dwell_start[i] = None
    if hovered is None:
        return None
    if confirm_dwell_start[hovered] is None:
        confirm_dwell_start[hovered] = time.time()
    dwell_duration = time.time() - confirm_dwell_start[hovered]
    if dwell_duration >= CONFIRM_DWELL_TIME:
        confirm_dwell_start[hovered] = None
        return confirm_options[hovered]
    return None


def draw_emergency_confirm(canvas, screen_width, screen_height, gaze_x=None, gaze_y=None):
    hovered = _get_hovered_confirm_option(gaze_x, gaze_y, screen_width, screen_height)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (screen_width, screen_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    title = "Are you sure you want to call emergency services?"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = (screen_width - tw) // 2
    cv2.putText(canvas, title, (tx, screen_height // 2 - 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    hint = "Look at YES/NO and hold to confirm"
    (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    hx = (screen_width - hw) // 2
    cv2.putText(canvas, hint, (hx, screen_height // 2 - 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    labels = ["YES", "NO"]
    colors = [(0, 200, 0), (0, 0, 255)]
    for i, label in enumerate(labels):
        x1, y1, x2, y2 = _confirm_button_bounds(i, screen_width, screen_height)
        is_hovered = (i == hovered)
        bg = (70, 70, 70) if is_hovered else (35, 35, 35)
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)
        border = (255, 255, 255) if is_hovered else (120, 120, 120)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 4 if is_hovered else 2)

        if is_hovered and confirm_dwell_start[i] is not None:
            progress = min((time.time() - confirm_dwell_start[i]) / CONFIRM_DWELL_TIME, 1.0)
            cx = (x1 + x2) // 2
            cy = y1 + 30
            radius = 26
            cv2.ellipse(canvas, (cx, cy), (radius, radius), -90, 0, int(360 * progress), (0, 255, 0), 5)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
        lx = x1 + (x2 - x1 - lw) // 2
        ly = y1 + (y2 - y1 + lh) // 2 + 10
        cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 1.6, colors[i], 4, cv2.LINE_AA)


def launch_scrcpy(device_serial: str | None = None) -> None:
    scrcpy = shutil.which("scrcpy")
    if not scrcpy and os.name == "nt":
        scrcpy = os.path.join(
            os.environ.get("LOCALAPPDATA", ""),
            "Microsoft",
            "WinGet",
            "Packages",
            "Genymobile.scrcpy_Microsoft.Winget.Source_8wekyb3d8bbwe",
            "scrcpy-win64-v3.3.4",
            "scrcpy.exe",
        )
        if not os.path.isfile(scrcpy):
            scrcpy = None
    if not scrcpy:
        print("[scrcpy] scrcpy not found on PATH (restart terminal after install)")
        return
    cmd = [scrcpy]
    if device_serial:
        cmd += ["-s", device_serial]
    try:
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        print("[scrcpy] launched")
    except Exception as e:
        print(f"[scrcpy] failed to launch: {e}")


# ============================================================
#                   IoT SUBMENU (PHONE UI)
# ============================================================

IOT_DWELL_TIME = 1.2
iot_options = ["show_phone_ui", "play_flappy_bird", "back"]
iot_dwell_start = [None] * len(iot_options)


def _iot_button_bounds(option_index: int, screen_width: int, screen_height: int):
    w = 520
    h = 130
    gap = 30
    total_h = (h * len(iot_options)) + gap * (len(iot_options) - 1)
    top_y = (screen_height - total_h) // 2
    x1 = (screen_width - w) // 2
    x2 = x1 + w
    y1 = top_y + option_index * (h + gap)
    y2 = y1 + h
    return x1, y1, x2, y2


def _get_hovered_iot_option(x, y, screen_width, screen_height):
    if x is None or y is None:
        return None
    for i in range(len(iot_options)):
        x1, y1, x2, y2 = _iot_button_bounds(i, screen_width, screen_height)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None


def update_iot_menu(gaze_x, gaze_y, screen_width, screen_height):
    global iot_dwell_start
    hovered = _get_hovered_iot_option(gaze_x, gaze_y, screen_width, screen_height)
    for i in range(len(iot_options)):
        if i != hovered:
            iot_dwell_start[i] = None
    if hovered is None:
        return None
    if iot_dwell_start[hovered] is None:
        iot_dwell_start[hovered] = time.time()
    if time.time() - iot_dwell_start[hovered] >= IOT_DWELL_TIME:
        iot_dwell_start[hovered] = None
        return iot_options[hovered]
    return None


def draw_iot_menu(canvas, screen_width, screen_height, gaze_x=None, gaze_y=None):
    hovered = _get_hovered_iot_option(gaze_x, gaze_y, screen_width, screen_height)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (screen_width, screen_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    title = "HOME IoT"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(canvas, title, ((screen_width - tw) // 2, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    subtitle = "Phone controls"
    (sw, sh), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(canvas, subtitle, ((screen_width - sw) // 2, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    labels = ["SHOW PHONE UI", "PLAY FLAPPY BIRD", "BACK"]
    for i, label in enumerate(labels):
        x1, y1, x2, y2 = _iot_button_bounds(i, screen_width, screen_height)
        is_hovered = (i == hovered)
        bg = (70, 70, 70) if is_hovered else (35, 35, 35)
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)
        border = (255, 255, 255) if is_hovered else (120, 120, 120)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 4 if is_hovered else 2)

        if is_hovered and iot_dwell_start[i] is not None:
            progress = min((time.time() - iot_dwell_start[i]) / IOT_DWELL_TIME, 1.0)
            cx = (x1 + x2) // 2
            cy = y1 + 28
            radius = 24
            cv2.ellipse(canvas, (cx, cy), (radius, radius), -90, 0, int(360 * progress), (0, 255, 0), 5)

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
        lx = x1 + (x2 - x1 - lw) // 2
        ly = y1 + (y2 - y1 + lh) // 2 + 6
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)


def run_demo():
    """Main demo function with menu and integrated keyboard"""
    args = parse_common_args()

    filter_method = args.filter
    camera_index = args.camera
    calibration_method = args.calibration
    background_path = args.background
    confidence_level = args.confidence
    cursor_enabled = args.cursor
    scan_path_enabled = args.scan_path or args.cursor
    scan_path_max = args.scan_path_max
    scan_path_log = args.scan_path_log

    gaze_estimator = GazeEstimator(
        model_name=args.model,
        landmark_alpha=args.landmark_alpha,
        feature_alpha=args.feature_alpha,
        include_face_position=args.multi_position,
    )
    gaze_estimator._pose_damping = args.pose_damping

    # Load or calibrate model
    if args.model_file and os.path.isfile(args.model_file):
        gaze_estimator.load_model(args.model_file)
        print(f"[demo] Loaded gaze model from {args.model_file}")
    elif args.multi_position:
        run_multi_position_calibration(
            gaze_estimator, camera_index=camera_index,
            calibration_method=calibration_method, multi_pose=args.multi_pose,
            single_column=args.single_column,
        )
    else:
        mp = args.multi_pose
        if calibration_method == "9p":
            run_9_point_calibration(gaze_estimator, camera_index=camera_index,
                                    multi_pose=mp)
        elif calibration_method == "5p":
            run_5_point_calibration(gaze_estimator, camera_index=camera_index,
                                    multi_pose=mp)
        elif calibration_method == "vertical":
            run_vertical_enhanced_calibration(gaze_estimator, camera_index=camera_index,
                                              single_column=args.single_column,
                                              multi_pose=mp)
        else:
            run_lissajous_calibration(gaze_estimator, camera_index=camera_index)

    if args.save_calibration and not (args.model_file and os.path.isfile(args.model_file)):
        gaze_estimator.save_model(args.save_calibration)
        print(f"[demo] Calibration saved to {args.save_calibration}")

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

    # Mode tracking: "menu", "keyboard", "home_iot", "emergency_confirm"
    current_mode = "menu"
    emergency_cfg = EmergencyCallConfig(
        device_serial=None,
        phone_number="+971559877486",
        message="hi this is a test",
    )
    phone_cfg = EmergencyCallConfig(device_serial=None)
    flappy_active = False
    flappy_center: tuple[int, int] | None = None
    last_flappy_tap = 0.0
    FLAPPY_TAP_COOLDOWN_S = 0.35

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

    # Triple-blink detection → triggers LLM sentence generation
    TRIPLE_BLINK_MAX_GAP = 0.6   # max gap between consecutive blinks
    TRIPLE_BLINK_MIN_GAP = 0.08  # min gap (filters noise)
    TRIPLE_BLINK_WINDOW  = 2.0   # total time window for all 3 blinks
    blink_active = False
    blink_times: list = []
    blink_count = 0
    double_blink_flash_until = 0.0
    llm_generating = False
    llm_banner_until = 0.0

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

            # ── Triple-blink detection ─────────────────────────────
            if blink_detected:
                if not blink_active:
                    blink_active = True
            else:
                if blink_active:
                    blink_active = False
                    blink_count += 1
                    now_ts = time.time()
                    # Drop blinks outside the rolling window
                    blink_times = [t for t in blink_times if now_ts - t <= TRIPLE_BLINK_WINDOW]
                    if blink_times:
                        gap = now_ts - blink_times[-1]
                        if TRIPLE_BLINK_MIN_GAP <= gap <= TRIPLE_BLINK_MAX_GAP:
                            blink_times.append(now_ts)
                        else:
                            blink_times = [now_ts]  # gap too large, start fresh
                    else:
                        blink_times = [now_ts]
                    if len(blink_times) >= 3:
                        double_blink_flash_until = time.time() + 2.0
                        print(f"[triple-blink] Detected! (blinks={blink_count})")
                        if current_mode == "keyboard" and not llm_generating:
                            keywords = keyboard.typed_text.strip()
                            if keyboard.current_predictions:
                                keywords += " " + keyboard.current_predictions[0]
                            keywords = keywords.strip()
                            if keywords:
                                print(f"[triple-blink] Sending to LLM: '{keywords}'")
                                generate_sentence_async(keywords)
                                llm_generating = True
                                llm_banner_until = time.time() + 30
                            else:
                                print("[triple-blink] No text to send")
                        blink_times = []


            # ── Poll for LLM result ───────────────────────────────
            if llm_generating:
                llm_result = get_pending_result()
                if llm_result is not None:
                    print(f"[LLM] Result: {llm_result}")
                    keyboard.typed_text = llm_result + " "
                    with open(OUTPUT_FILE, "w") as f:
                        f.write(keyboard.typed_text)
                    keyboard.current_key_sequence = []
                    keyboard.current_predictions = []
                    keyboard.current_ngram_predictions = []
                    keyboard.selected_prediction_idx = 0
                    llm_generating = False
                    llm_banner_until = time.time() + 3

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
                    if selected_option == "emergency_call":
                        current_mode = "emergency_confirm"
                        print("[menu] Emergency confirm opened")
                    else:
                        current_mode = selected_option
                        print(f"[menu] Selected: {selected_option}")
            elif current_mode == "home_iot":
                choice = update_iot_menu(x_pred, y_pred, screen_width, screen_height)
                if choice == "show_phone_ui":
                    try:
                        phone_cfg = ensure_device(phone_cfg)
                    except Exception as e:
                        print(f"[scrcpy] No device: {e}")
                        phone_cfg = EmergencyCallConfig(device_serial=None)
                    launch_scrcpy(device_serial=phone_cfg.device_serial)
                elif choice == "play_flappy_bird":
                    # Open a browser-based flappy bird and map blinks -> center taps.
                    try:
                        phone_cfg = ensure_device(phone_cfg)
                        # Show the phone UI so the user can see the game.
                        launch_scrcpy(device_serial=phone_cfg.device_serial)
                        open_url(phone_cfg, "https://flappybird.io/")
                        # Give the browser a moment to come to foreground and load.
                        time.sleep(2.0)
                        w, h = get_phone_screen_size(phone_cfg)
                        flappy_center = (w // 2, h // 2)
                        flappy_active = True
                        print(f"[flappy] Enabled blink->tap at {flappy_center[0]},{flappy_center[1]}")
                    except Exception as e:
                        flappy_active = False
                        flappy_center = None
                        print(f"[flappy] Failed to start: {e}")
                elif choice == "back":
                    current_mode = "menu"
                    flappy_active = False
                    flappy_center = None
            elif current_mode == "emergency_confirm":
                decision = update_emergency_confirm(x_pred, y_pred, screen_width, screen_height)
                if decision == "yes":
                    print("[menu] Emergency call confirmed")
                    try:
                        trigger_emergency_call(emergency_cfg)
                    except Exception as e:
                        print(f"[emergency] Failed: {e}")
                    current_mode = "menu"
                elif decision == "no":
                    print("[menu] Emergency call cancelled")
                    current_mode = "menu"
            elif current_mode == "keyboard":
                selected_key = keyboard.update(x_pred, y_pred)
                if selected_key is not None:
                    print(f"[keyboard] Key selected: {keyboard.keys[selected_key] if isinstance(selected_key, int) else selected_key}")

            # Blink-to-tap bridge (IoT flappy mode)
            if current_mode == "home_iot" and flappy_active and flappy_center and blink_detected:
                now_ts = time.time()
                if now_ts - last_flappy_tap >= FLAPPY_TAP_COOLDOWN_S:
                    try:
                        tap(phone_cfg, flappy_center[0], flappy_center[1])
                        last_flappy_tap = now_ts
                        print(f"[flappy] tap @{flappy_center[0]},{flappy_center[1]}")
                    except Exception as e:
                        flappy_active = False
                        print(f"[flappy] Tap failed, disabled: {e}")

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
            elif current_mode == "home_iot":
                # Dedicated IoT screen
                draw_iot_menu(canvas, screen_width, screen_height, x_pred, y_pred)
            elif current_mode == "emergency_confirm":
                # Dedicated confirmation screen (no underlying menu)
                draw_emergency_confirm(canvas, screen_width, screen_height, x_pred, y_pred)
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

            # Draw cursor if enabled via --cursor flag
            if cursor_enabled and x_pred is not None and y_pred is not None and cursor_alpha > 0:
                draw_cursor(canvas, x_pred, y_pred, cursor_alpha)

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

            # ── LLM / Double-blink status overlay ─────────────
            right_x = screen_width - 380
            cv2.putText(canvas, "Triple-blink: AI sentence",
                        (right_x, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Blinks: {blink_count}",
                        (right_x, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (180, 180, 180), 1, cv2.LINE_AA)

            if time.time() < double_blink_flash_until:
                cv2.putText(canvas, "ACTIVATED",
                            (right_x, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if llm_generating:
                banner_text = "Generating sentence..."
                ts = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                bx = (screen_width - ts[0]) // 2
                by = screen_height // 2
                cv2.rectangle(canvas, (bx - 20, by - 40),
                              (bx + ts[0] + 20, by + 15), (40, 40, 40), -1)
                cv2.rectangle(canvas, (bx - 20, by - 40),
                              (bx + ts[0] + 20, by + 15), (0, 200, 255), 2)
                cv2.putText(canvas, banner_text, (bx, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)

            elif time.time() < llm_banner_until and not llm_generating:
                result_text = keyboard.typed_text.strip()[:80]
                if result_text:
                    ts = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    bx = (screen_width - ts[0]) // 2
                    by = screen_height // 2
                    cv2.rectangle(canvas, (bx - 20, by - 40),
                                  (bx + ts[0] + 20, by + 15), (40, 40, 40), -1)
                    cv2.rectangle(canvas, (bx - 20, by - 40),
                                  (bx + ts[0] + 20, by + 15), (0, 255, 100), 2)
                    cv2.putText(canvas, result_text, (bx, by),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)

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
