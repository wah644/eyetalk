import os
import shutil
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from eyetrax.app.swipe_dtw import SwipeTemplateDB, SWIPE_ARM_DWELL, SWIPE_END_DWELL
from eyetrax.app.keyword_to_sentence import (
    generate_sentence_async,
    get_pending_result,
    is_generating,
)
from eyetrax.calibration import (
    run_5_point_calibration,
    run_9_point_calibration,
    run_lissajous_calibration,
    run_multi_position_calibration,
    run_vertical_center_calibration,
    run_vertical_enhanced_calibration,
    run_vertical_only_calibration,
)
from eyetrax.cli import parse_common_args
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.gaze import GazeEstimator
from eyetrax.utils.draw import draw_cursor, draw_scan_path, make_thumbnail
from eyetrax.utils.screen import get_screen_size
from eyetrax.utils.video import camera, fullscreen, iter_frames
from eyetrax.utils.tts import OpenVoiceSpeaker
from eyetrax.integrations.adb_emergency import (
    EmergencyCallConfig, ensure_device, get_phone_screen_size,
    open_url, tap, trigger_emergency_call,
)
from eyetrax.integrations.lights import lights_on_async, lights_off_async


# ============================================================
#                   KEYBOARD CONFIGURATION
# ============================================================

KEYBOARD_KEYS_DWELL = [
    "ABCD",
    "EFGH",
    "IJKL",
    "MNOP",
    "QRSTU",
    "VWXYZ",
    "SPACE"
]

# Swipe mode has no SPACE key — gives more vertical room to letter rows
KEYBOARD_KEYS_SWIPE = [
    "ABCD",
    "EFGH",
    "IJKL",
    "MNOP",
    "QRSTU",
    "VWXYZ",
]

# Letter → row index for start/end row filtering (matches KEYBOARD_KEYS_SWIPE order)
_SWIPE_LETTER_ROW: dict[str, int] = {
    ch.lower(): row_idx
    for row_idx, row_str in enumerate(KEYBOARD_KEYS_SWIPE)
    for ch in row_str
}

# Default alias (used by legacy code paths)
KEYBOARD_KEYS = KEYBOARD_KEYS_DWELL

# Special actions (can be added as separate buttons if needed)
SPECIAL_KEYS = ["DELETE", "ACCEPT"]

# Map letters to key indices (uses full dwell layout — row indices identical in both modes)
LETTER_TO_KEY = {}
for idx, key in enumerate(KEYBOARD_KEYS_DWELL[:-1]):  # Exclude SPACE key
    for letter in key.lower():
        LETTER_TO_KEY[letter] = idx

DWELL_TIME = 0.85  # seconds to dwell on a key to select it
DWELL_FEEDBACK_CIRCLE_MAX = 30  # max radius for dwell feedback circle
OUTPUT_FILE = "keyboard_output.txt"

# Swipe-typing constants (dwell times imported from swipe_dtw.py)
SWIPE_CANCEL_TIME = 8.0   # auto-cancel a swipe if it takes longer than this
SWIPE_RERANK_TOLERANCE = 0.15  # if 2nd-best combined score within 15% of best → open corrections panel
SWIPE_FREQ_ALPHA = 0.5    # combined score = dtw / (usage_count + 1)^alpha; 0=pure DTW, 1=full freq pull

AI_BLINK_DURATION = 1.0        # 1.0 s blink → beep + AI sentence generation
SENTENCE_BLINK_DURATION = 2.0  # 2.0 s blink → second beep + sentence complete
SENTENCE_COMPLETE_SOUND = Path(__file__).parent / "sentence_complete_sound.mp3"

# Fixed phrases for the Fixed Phrases panel
FIXED_PHRASES = [
    "Yes",
    "No",
    "Thank you",
    "Please help me",
    "I need water",
    "I need food",
    "I am in pain",
    "Call the nurse",
    "I feel sick",
    "I am cold",
    "I am hot",
    "Turn on the light",
    "Turn off the light",
]

# Word selection panel constants
SELECTION_ZONE_DWELL_TIME = 2.0   # seconds looking up/down to enter panel
UP_ZONE_RATIO   = 0.15            # top 15% of screen → trie panel trigger
DOWN_ZONE_RATIO = 0.85            # bottom 15% of screen → bigram panel trigger
WORD_PANEL_DWELL_TIME = 1.5       # seconds to dwell at centre to select a word
TRIE_PANEL_ORDER  = [4, 3, 2, 1, 0]  # worst→best top-to-bottom; best (idx 0) at dp=4 (bottom)
NGRAM_PANEL_ORDER = [0, 1, 2, 3, 4]  # best→worst top-to-bottom; best (idx 0) at dp=0 (top)
TRIE_BACKSPACE_TIME   = 2.0       # seconds in up-scroll zone → backspace
TRIE_EXIT_TIME        = 2.0       # seconds at bottom scroll limit (looking down) → exit
NGRAM_EXIT_DOWN_TIME  = 5.0       # seconds at bottom scroll limit (looking down) → home
NGRAM_EXIT_UP_TIME    = 2.0       # seconds at top scroll limit (looking up) → exit

_WORDS_CSV = Path(__file__).parent / "english_words.csv"
_DICTIONARY_LIMIT = 35_000


def load_dictionary():
    """Load top _DICTIONARY_LIMIT words from english_words.csv (word,freq format)."""
    if not _WORDS_CSV.exists():
        print(f"[Dictionary] {_WORDS_CSV.name} not found, using fallback")
        return [("hello",1),("hi",2),("hey",3),("help",4),("how",5),
                ("are",6),("you",7),("i",8),("am",9),("the",10)]
    import csv
    out: list[tuple[str, int]] = []
    seen: set[str] = set()
    with open(_WORDS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(out) >= _DICTIONARY_LIMIT:
                break
            word = row.get("word", "").strip().lower()
            if not word or not word.isalpha() or word in seen:
                continue
            seen.add(word)
            out.append((word, len(out) + 1))
    print(f"[Dictionary] Loaded {len(out)} words from {_WORDS_CSV.name}")
    return out


DICTIONARY = load_dictionary()  # list of (word, rank)


# ============================================================
#                   TRIE IMPLEMENTATION
# ============================================================

class TrieNode:
    def __init__(self):
        self.children = {}
        self.words: list[tuple[int, str]] = []   # (rank, word) — multiple words per node


class T9Trie:
    """Trie structure for T9-style word prediction"""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, rank: int = 999999):
        node = self.root
        word = word.lower()
        for char in word:
            ki = LETTER_TO_KEY.get(char)
            if ki is None:
                return
            if ki not in node.children:
                node.children[ki] = TrieNode()
            node = node.children[ki]
        node.words.append((rank, word))

    def search_predictions(self, key_sequence, max_results=5):
        if not key_sequence:
            return []
        node = self.root
        for ki in key_sequence:
            if ki not in node.children:
                return []
            node = node.children[ki]
        results: list[tuple[int, str]] = []
        self._collect_words(node, results)
        results.sort(key=lambda x: x[0])
        return [w for _, w in results[:max_results]]

    def _collect_words(self, node, out):
        out.extend(node.words)
        for child in node.children.values():
            self._collect_words(child, out)


# ============================================================
#                   BIGRAM MODEL
# ============================================================

def load_bigrams(filepath=None):
    """Load bigrams from a text file with format: 'word1 word2    frequency'"""
    if filepath is None:
        filepath = Path(__file__).parent / "Bigram.txt"
    raw: dict[str, dict[str, int]] = {}
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
                raw.setdefault(word1, {})
                raw[word1][word2] = raw[word1].get(word2, 0) + freq
        bigrams: dict[str, list[tuple[str, int]]] = {}
        for word, freq_map in raw.items():
            bigrams[word] = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
        count = sum(len(v) for v in bigrams.values())
        print(f"[Bigram] Loaded {count} bigram entries from {filepath}")
    except FileNotFoundError:
        print(f"[Bigram] Warning: {filepath} not found, bigram predictions unavailable")
        bigrams = {}
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


class UsageTracker:
    """Persistent usage counters for unigram and bigram re-ranking."""

    def __init__(self, path: str | Path = "user_usage.json"):
        self.path = Path(path)
        self._data: dict = {"words": {}, "bigrams": {}}
        self._load()

    def _load(self):
        if self.path.exists():
            import json
            self._data = json.loads(self.path.read_text())

    def _save(self):
        import json
        self.path.write_text(json.dumps(self._data, indent=2))

    def record_word(self, word: str):
        """Increment unigram usage count for each individual word in the string."""
        for w in word.lower().split():
            if w.isalpha():
                self._data["words"][w] = self._data["words"].get(w, 0) + 1
        self._save()

    def record_bigram(self, prev_word: str, next_word: str):
        """Increment bigram usage count for (prev -> next)."""
        p, n = prev_word.lower().strip(), next_word.lower().strip()
        self._data["bigrams"].setdefault(p, {})[n] = (
            self._data["bigrams"].get(p, {}).get(n, 0) + 1
        )
        self._save()

    def rerank_words(self, words: list[str]) -> list[str]:
        """Re-rank a list of words: usage count DESC, preserve original order as tiebreak."""
        counts = self._data["words"]
        return sorted(words, key=lambda w: -counts.get(w.lower().strip(), 0))

    def rerank_bigrams(self, prev_word: str, words: list[str]) -> list[str]:
        """Re-rank bigram suggestions: usage count DESC, preserve original order as tiebreak."""
        p = prev_word.lower().strip()
        bigram_counts = self._data["bigrams"].get(p, {})
        return sorted(words, key=lambda w: -bigram_counts.get(w.lower().strip(), 0))

    def record_sentence_bigrams(self, sentence: str):
        """Record every consecutive word-pair from a completed sentence."""
        words = [w for w in sentence.lower().split() if w.isalpha()]
        for i in range(len(words) - 1):
            self.record_bigram(words[i], words[i + 1])

    def get_merged_bigram_predictions(self, prev_word: str, bigram_model, max_results: int = 5) -> list[str]:
        """Return up to max_results next-word predictions.
        Custom bigram (sorted by personal usage count) fills first, then model bigram fills
        remaining slots so there are always up to max_results suggestions."""
        p = prev_word.lower().strip()
        custom_counts = self._data["bigrams"].get(p, {})
        custom_sorted = [w for w, _ in sorted(custom_counts.items(), key=lambda x: -x[1])]
        result = custom_sorted[:max_results]
        if len(result) < max_results:
            seen = set(result)
            for w in bigram_model.get_predictions(prev_word, max_results):
                if w not in seen:
                    result.append(w)
                    if len(result) >= max_results:
                        break
        return result


# ============================================================
#               WORD SELECTION PANEL
# ============================================================

class WordSelectionPanel:
    """Scrollable vertical word-selection panel (replaces keyboard in selection mode).

    Words are arranged top-to-bottom per panel type: trie shows worst→best
    (best at bottom, highlighted); bigram shows best→worst (best at top).  The user
    scrolls by looking above/below screen-centre; the further from centre, the
    faster the scroll.  Dwelling at screen-centre for WORD_PANEL_DWELL_TIME
    seconds selects the word currently nearest the centre.
    """

    def __init__(self, screen_width: int, screen_height: int, panel_type: str = "trie"):
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.panel_type    = panel_type  # "trie" or "ngram"

        # Word order for this panel type
        if panel_type in ("trie", "swipe_corrections"):
            self._order = TRIE_PANEL_ORDER
        else:
            self._order = NGRAM_PANEL_ORDER

        # Panel geometry (centred horizontally)
        self.panel_width = 360
        self.panel_x     = (screen_width - self.panel_width) // 2

        # Word layout
        self.word_spacing    = 155   # px between word-button centres
        self.max_word_height = 115   # button height when at screen centre
        self.min_word_height = 28    # floor for very distant words

        # Scroll state — initialised properly per panel in set_words()
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

        # Exit / backspace timers
        self._exit_start:      float | None = None
        self._backspace_start: float | None = None
        self._exit_edge: str = "bottom"  # "top" or "bottom" — which edge triggered exit

        self._last_t = time.time()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def set_words(self, words: list[str]) -> None:
        """Load up to 5 words and reset scroll/dwell state."""
        self.words = list(words[:5])
        n = len(self.words)
        # Compute valid display positions for the loaded words
        valid_dps = [dp for dp, wi in enumerate(self._order) if wi < n]
        if valid_dps:
            if self.panel_type in ("trie", "swipe_corrections"):
                # Best word is at highest dp (bottom) — start scrolled there
                self.scroll_offset = float(max(valid_dps) - 2)
            else:
                # Best word is at dp=0 (top) — start scrolled there
                self.scroll_offset = float(min(valid_dps) - 2)
        else:
            self.scroll_offset = 0.0
        self.dwell_start      = None
        self.dwell_target_idx = None
        self._exit_start      = None
        self._backspace_start = None
        self._exit_edge       = "bottom"
        self._last_t          = time.time()

    def update(self, gaze_x, gaze_y) -> str | None:
        """Advance scroll and dwell.  Returns selected word string or None."""
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now

        if gaze_y is None:
            return None

        screen_cy = self.screen_height // 2
        delta     = gaze_y - screen_cy

        # ---- scrolling --------------------------------------------------
        if self.words and abs(delta) > self.scroll_threshold:
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

        # ---- exit / backspace detection ---------------------------------
        # Only fire when gaze is on-screen — prevents the opening gesture
        # (y < -40 or y > height+40) from immediately triggering timers.
        if gaze_y is None or not (0 <= gaze_y <= self.screen_height):
            self._exit_start      = None
            self._backspace_start = None
        else:
            entries = self._display_entries()
            if entries:
                min_off = min(p for p, _, _ in entries) - 2.0
                max_off = max(p for p, _, _ in entries) - 2.0
            else:
                min_off = max_off = 0.0

            at_bottom = abs(self.scroll_offset - max_off) < 0.05
            at_top    = abs(self.scroll_offset - min_off) < 0.05
            in_up_zone   = delta < -self.scroll_threshold
            in_down_zone = delta >  self.scroll_threshold

            if self.panel_type == "trie":
                # At top + looking up → backspace (only fires when scroll is exhausted)
                if in_up_zone and at_top:
                    if self._backspace_start is None:
                        self._backspace_start = now
                    elif now - self._backspace_start >= TRIE_BACKSPACE_TIME:
                        self._backspace_start = None
                        return "__BACKSPACE__"
                else:
                    self._backspace_start = None
                # At bottom + looking down → exit (only fires when scroll is exhausted)
                if in_down_zone and at_bottom:
                    if self._exit_start is None:
                        self._exit_start = now
                        self._exit_edge  = "bottom"
                    elif now - self._exit_start >= TRIE_EXIT_TIME:
                        self._exit_start = None
                        return "__EXIT__"
                else:
                    self._exit_start = None

            elif self.panel_type == "swipe_corrections":
                # At bottom + looking down OR at top + looking up → exit (dismiss panel)
                if (in_down_zone and at_bottom) or (in_up_zone and at_top):
                    if self._exit_start is None:
                        self._exit_start = now
                        self._exit_edge  = "bottom" if in_down_zone else "top"
                    elif now - self._exit_start >= TRIE_EXIT_TIME:
                        self._exit_start = None
                        return "__EXIT__"
                else:
                    self._exit_start = None

            elif self.panel_type == "ngram":
                # At top + looking up → exit (only fires when scroll is exhausted)
                if in_up_zone and at_top:
                    if self._exit_start is None:
                        self._exit_start = now
                        self._exit_edge  = "top"
                    elif now - self._exit_start >= NGRAM_EXIT_UP_TIME:
                        self._exit_start = None
                        return "__EXIT__"
                # At bottom + looking down → home screen (only fires when scroll is exhausted)
                elif in_down_zone and at_bottom:
                    if self._exit_start is None:
                        self._exit_start = now
                        self._exit_edge  = "bottom"
                    elif now - self._exit_start >= NGRAM_EXIT_DOWN_TIME:
                        self._exit_start = None
                        return "__HOME__"
                else:
                    self._exit_start = None

        return None

    def get_dwell_progress(self) -> float:
        if self.dwell_start is None:
            return 0.0
        return min((time.time() - self.dwell_start) / self.dwell_time, 1.0)

    def draw(self, canvas: np.ndarray) -> None:
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

        if not entries:
            msg = "No suggestions"
            (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(canvas, msg,
                        (self.panel_x + (self.panel_width - mw) // 2, self.screen_height // 2 + mh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (130, 130, 130), 2, cv2.LINE_AA)

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

            is_dwelled  = (wi == self.dwell_target_idx)
            is_center   = abs(yc - screen_cy) < 25
            is_best     = (wi == 0)  # word index 0 = best prediction

            if is_dwelled:
                bg, border, thick = (50, 90, 50),  (0, 255, 60),   4
            elif is_best and self.panel_type == "trie":
                bg, border, thick = (40, 60, 20),  (0, 230, 255),  3
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

        # ---- action progress indicators (backspace / exit) -----------------
        _now = time.time()
        if self._backspace_start is not None:
            prog = min((_now - self._backspace_start) / TRIE_BACKSPACE_TIME, 1.0)
            bar_w = max(1, int(self.screen_width * prog))
            cv2.rectangle(canvas, (0, 0), (bar_w, 12), (60, 60, 220), -1)
            lbl = "BACKSPACE"
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            cv2.putText(canvas, lbl, ((self.screen_width - lw) // 2, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (120, 120, 255), 2, cv2.LINE_AA)

        if self._exit_start is not None:
            if self.panel_type == "trie":
                timeout = TRIE_EXIT_TIME
                edge = "bottom"
            elif self._exit_edge == "top":
                timeout = NGRAM_EXIT_UP_TIME
                edge = "top"
            else:
                timeout = NGRAM_EXIT_DOWN_TIME
                edge = "bottom"
            prog = min((_now - self._exit_start) / timeout, 1.0)
            bar_w = max(1, int(self.screen_width * prog))
            if edge == "bottom":
                cv2.rectangle(canvas, (0, self.screen_height - 12),
                              (bar_w, self.screen_height), (0, 140, 255), -1)
                if self.panel_type == "ngram":
                    lbl = "HOME"
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
                    cv2.putText(canvas, lbl,
                                ((self.screen_width - lw) // 2, self.screen_height - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(canvas, (0, 0), (bar_w, 12), (0, 140, 255), -1)

        # Header (trie and ngram only — swipe_corrections draws its own label)
        if self.panel_type == "trie":
            label, col = "TRIE PREDICTIONS", (255, 200, 50)
            cv2.putText(canvas, label,
                        (self.panel_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
        elif self.panel_type == "ngram":
            label, col = "NEXT WORD (BIGRAM)", (200, 100, 255)
            cv2.putText(canvas, label,
                        (self.panel_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _display_entries(self) -> list[tuple[int, int, str]]:
        """Return (display_pos, word_index, word) for every valid entry."""
        n = len(self.words)
        return [
            (dp, wi, self.words[wi])
            for dp, wi in enumerate(self._order)
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
#               FIXED PHRASES PANEL
# ============================================================

class FixedPhrasesPanel:
    """Scrollable vertical panel for fixed phrases (not limited to 5 like WordSelectionPanel).

    Phrases are listed sequentially top-to-bottom.  The user scrolls by
    looking above/below screen-centre and dwells at centre to select.
    """

    def __init__(self, screen_width: int, screen_height: int, phrases: list[str] | None = None):
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.phrases       = list(phrases or FIXED_PHRASES)

        # Panel geometry — wider than WordSelectionPanel for longer text
        self.panel_width = 600
        self.panel_x     = (screen_width - self.panel_width) // 2

        # Layout
        self.phrase_spacing    = 130   # px between phrase-button centres
        self.max_phrase_height = 100
        self.min_phrase_height = 28

        # Scroll state (0.0 → first phrase at screen centre)
        self.scroll_offset: float = 0.0

        # Dwell / selection
        self.dwell_start:      float | None = None
        self.dwell_target_idx: int   | None = None
        self.dwell_time = WORD_PANEL_DWELL_TIME

        self.scroll_threshold = screen_height // 6
        self.max_scroll_speed = 2.8

        self._last_t = time.time()

    def update(self, gaze_x, gaze_y) -> str | None:
        """Advance scroll and dwell.  Returns selected phrase string or None."""
        now = time.time()
        dt  = now - self._last_t
        self._last_t = now

        if not self.phrases or gaze_y is None:
            return None

        screen_cy = self.screen_height // 2
        delta     = gaze_y - screen_cy

        # Scrolling
        if abs(delta) > self.scroll_threshold:
            effective = abs(delta) - self.scroll_threshold
            half_h    = max(screen_cy - self.scroll_threshold, 1)
            speed     = self.max_scroll_speed * (effective / half_h)
            self.scroll_offset += speed * dt * (1 if delta > 0 else -1)
            max_off = max(0, len(self.phrases) - 1)
            self.scroll_offset = max(0.0, min(float(max_off), self.scroll_offset))

        # Find phrase closest to screen centre
        center_idx = None
        min_dist   = float("inf")
        for i in range(len(self.phrases)):
            yc   = self._phrase_yc(i)
            dist = abs(yc - screen_cy)
            if dist < min_dist:
                min_dist   = dist
                center_idx = i

        # Dwell (only in the middle zone)
        dwell_zone = self.scroll_threshold
        if abs(delta) <= dwell_zone and center_idx is not None:
            if self.dwell_target_idx != center_idx:
                self.dwell_target_idx = center_idx
                self.dwell_start      = now
            elif self.dwell_start is not None:
                if now - self.dwell_start >= self.dwell_time:
                    selected = self.phrases[self.dwell_target_idx]
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
        if not self.phrases:
            return

        screen_cy = self.screen_height // 2
        dp_prog   = self.get_dwell_progress()

        # Zone indicators
        line_top = self.screen_height // 3
        line_bot = (self.screen_height * 2) // 3

        zone_overlay = canvas.copy()
        cv2.rectangle(zone_overlay, (0, 0), (self.screen_width, line_top), (20, 20, 60), -1)
        cv2.rectangle(zone_overlay, (0, line_bot), (self.screen_width, self.screen_height), (20, 20, 60), -1)
        cv2.addWeighted(zone_overlay, 0.25, canvas, 0.75, 0, canvas)

        dash_len, gap_len = 30, 18
        x = 0
        while x < self.screen_width:
            x2 = min(x + dash_len, self.screen_width)
            cv2.line(canvas, (x, line_top), (x2, line_top), (100, 110, 220), 1)
            cv2.line(canvas, (x, line_bot), (x2, line_bot), (100, 110, 220), 1)
            x += dash_len + gap_len

        cv2.putText(canvas, "LOOK HERE  to scroll up",
                    (self.screen_width // 2 - 180, line_top // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 110, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "LOOK HERE  to scroll down",
                    (self.screen_width // 2 - 195, (line_bot + self.screen_height) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 110, 220), 2, cv2.LINE_AA)
        cv2.putText(canvas, "LOOK HERE  to select",
                    (self.screen_width // 2 - 160, screen_cy - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 130), 2, cv2.LINE_AA)

        for i, phrase in enumerate(self.phrases):
            yc = self._phrase_yc(i)
            h  = self._phrase_h(yc)
            y1 = yc - h // 2
            y2 = yc + h // 2
            x1 = self.panel_x
            x2 = self.panel_x + self.panel_width

            if y2 < 0 or y1 > self.screen_height:
                continue

            y1c = max(0, y1)
            y2c = min(self.screen_height - 1, y2)

            is_dwelled = (i == self.dwell_target_idx)
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

            fs = max(0.45, min(1.4, h / 75.0))
            (tw, th), _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            tx = x1 + (self.panel_width - tw) // 2
            ty = max(y1c + th + 4, min(y2c - 4, yc + th // 2))
            cv2.putText(canvas, phrase, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

            if is_dwelled and dp_prog > 0:
                cx     = (x1 + x2) // 2
                cy_cir = max(10, min(self.screen_height - 10, yc))
                radius = max(8, h // 3)
                angle  = int(360 * dp_prog)
                cv2.ellipse(canvas, (cx, cy_cir), (radius, radius),
                            -90, 0, angle, (0, 255, 100), 4)

        # Header
        cv2.putText(canvas, "FIXED PHRASES",
                    (self.panel_x, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (50, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Look up/down to scroll  |  Pause at centre to select",
                    (self.panel_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (155, 155, 155), 1, cv2.LINE_AA)

    def _phrase_yc(self, idx: int) -> int:
        screen_cy = self.screen_height // 2
        return screen_cy + int((idx - self.scroll_offset) * self.phrase_spacing)

    def _phrase_h(self, yc: int) -> int:
        screen_cy = self.screen_height // 2
        dist      = abs(yc - screen_cy) / self.phrase_spacing
        h         = int(self.max_phrase_height * ((2 / 3) ** dist))
        return max(h, self.min_phrase_height)


# ============================================================
#                   KEYBOARD CONTROLLER
# ============================================================

def _wrap_text(text: str, font: int, scale: float, thickness: int, max_width: int) -> list[str]:
    """Break *text* into lines that each fit within *max_width* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = (current + " " + word).strip()
        w, _ = cv2.getTextSize(candidate, font, scale, thickness)[0]
        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines if lines else [""]


class KeyboardController:
    """Handles vertical keyboard layout and dwell-based selection with T9 prediction"""

    def __init__(self, screen_width, screen_height, keys=None, dwell_time=DWELL_TIME,
                 swipe_db: SwipeTemplateDB | None = None,
                 input_mode: str = "eyeswipe",
                 tts: OpenVoiceSpeaker | None = None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.input_mode = input_mode
        if keys is not None:
            self.keys = keys
        elif input_mode == "eyeswipe":
            self.keys = list(KEYBOARD_KEYS_SWIPE)
        else:
            self.keys = list(KEYBOARD_KEYS_DWELL)
        self.num_keys = len(self.keys)
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

        # TTS speaker (always-on per-word + sentence-on-long-blink)
        self.tts: OpenVoiceSpeaker | None = tts

        # ---- Swipe-typing state ----
        self.swipe_db: SwipeTemplateDB | None = swipe_db
        # "idle" | "armed" (first-key dwell building up) | "recording" | "ending" (end-key dwell)
        self.swipe_state: str = "idle"
        self.swipe_trajectory: list[float] = []   # normalised y positions
        self.swipe_armed_key: int | None = None   # row being dwelled on to start
        self.swipe_arm_start: float | None = None
        self.swipe_start_time: float | None = None  # when recording began (for timeout)
        self.swipe_end_key: int | None = None     # row being dwelled on to finish
        self.swipe_end_start: float | None = None
        # Last swipe auto-selection (for left-zone correction and right-zone bigram)
        self.last_swipe_auto_word: str | None = None
        self.last_swipe_matches: list[str] = []

        # T9 prediction
        self.trie = T9Trie()
        for word, rank in DICTIONARY:
            self.trie.insert(word, rank=rank)

        self.current_key_sequence = []
        self.current_predictions: list[str] = []
        self.current_ngram_predictions: list[str] = []
        self.selected_prediction_idx = 0

        # Bigram model
        self.bigram = BigramModel()

        # Usage-based re-ranking
        self.usage       = UsageTracker("user_usage.json")       # dwell: trie rerank + bigrams
        self.swipe_usage = UsageTracker("user_usage_swipe.json") # swipe: DTW combined score

        # Word selection panels
        self.trie_panel             = WordSelectionPanel(screen_width, screen_height, "trie")
        self.ngram_panel            = WordSelectionPanel(screen_width, screen_height, "ngram")
        self.swipe_corrections_panel = WordSelectionPanel(screen_width, screen_height, "swipe_corrections")

        # Sub-mode: "typing" | "trie_select" | "ngram_select" | "swipe_corrections"
        self.submode = "typing"

        # Zone detection for triggering selection mode
        self.up_zone_start:   float | None = None
        self.down_zone_start: float | None = None
        self.zone_dwell_time = SELECTION_ZONE_DWELL_TIME
        self.up_zone_y   = int(screen_height * UP_ZONE_RATIO)
        self.down_zone_y = int(screen_height * DOWN_ZONE_RATIO)
        self._last_accepted_word: str = ""
        # True when corrections panel opened without a prior auto-accept
        self._swipe_panel_direct: bool = False

        # Initialize output file
        Path(OUTPUT_FILE).write_text("")

    @property
    def _is_swipe_mode(self) -> bool:
        return self.input_mode == "eyeswipe"

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

        # ---- trie selection mode -----------------------------------------
        if self.submode == "trie_select":
            selected = self.trie_panel.update(gaze_x, gaze_y)
            if selected == "__BACKSPACE__":
                self.handle_backspace()
            elif selected == "__EXIT__":
                self.submode = "typing"
            elif selected is not None:
                self._accept_word(selected)
                self.submode = "typing"
            return None

        # ---- ngram selection mode ----------------------------------------
        if self.submode == "ngram_select":
            selected = self.ngram_panel.update(gaze_x, gaze_y)
            if selected == "__EXIT__":
                self.submode = "typing"
            elif selected == "__HOME__":
                self.submode = "typing"
                return "__MENU__"
            elif selected is not None:
                if self._is_swipe_mode:
                    self._accept_word(selected)
                elif self.current_predictions:
                    top_word = self.current_predictions[0]
                    self._accept_word(top_word + " " + selected)
                else:
                    self._accept_word(selected)
                self.submode = "typing"
            return None

        # ---- swipe correction mode (swipe mode only) ---------------------
        if self.submode == "swipe_corrections":
            selected = self.swipe_corrections_panel.update(gaze_x, gaze_y)
            if selected == "__EXIT__":
                self.submode = "typing"
            elif selected is not None:
                if self._swipe_panel_direct:
                    # Panel opened without prior auto-accept — accept fresh
                    self._accept_word(selected)
                    self._swipe_panel_direct = False
                else:
                    # Panel opened after auto-accept — replace the accepted word
                    self._replace_last_word(selected)
                self.submode = "typing"
            return None

        # ---- typing mode: off-screen look-up/down triggers ---------------
        # Look above screen (y < -40) for 2s → trie panel
        # Look below screen (y > screen_height + 40) for 2s → bigram panel
        if gaze_y is not None:
            if gaze_y < -40:
                self.down_zone_start = None
                if self.up_zone_start is None:
                    self.up_zone_start = time.time()
                elif time.time() - self.up_zone_start >= self.zone_dwell_time:
                    self.up_zone_start = None
                    if self._is_swipe_mode:
                        if self.last_swipe_matches:
                            self.swipe_corrections_panel.set_words(self.last_swipe_matches)
                            self.submode = "swipe_corrections"
                    else:
                        self.trie_panel.set_words(self.current_predictions)
                        self.submode = "trie_select"
            elif gaze_y > self.screen_height + 40:
                self.up_zone_start = None
                if self.down_zone_start is None:
                    self.down_zone_start = time.time()
                elif time.time() - self.down_zone_start >= self.zone_dwell_time:
                    self.down_zone_start = None
                    source_word = self._bigram_source()
                    ngram_pred = self.usage.get_merged_bigram_predictions(
                        source_word, self.bigram
                    ) if source_word else []
                    self.ngram_panel.set_words(ngram_pred)
                    self.submode = "ngram_select"
            else:
                self.up_zone_start   = None
                self.down_zone_start = None

        return self._update_keyboard(gaze_x, gaze_y)

    # ------------------------------------------------------------------
    # Core keyboard dwell logic
    # ------------------------------------------------------------------

    def _update_keyboard(self, gaze_x, gaze_y):
        """Dwell-based keyboard key selection, with swipe-typing when a DB is loaded."""
        if gaze_x is None or gaze_y is None:
            self.current_key = None
            self.dwell_start_time = None
            return None

        # ---- swipe mode ----
        if self._is_swipe_mode:
            return self._update_swipe(gaze_x, gaze_y)

        # ---- original dwell logic ----
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

    # ------------------------------------------------------------------
    # Swipe-typing state machine
    # ------------------------------------------------------------------

    def _update_swipe(self, gaze_x, gaze_y):
        """Run one frame of the swipe-typing state machine."""
        now = time.time()
        hovered = self.get_hovered_key(gaze_x, gaze_y)

        # Always collect y-position when recording
        if self.swipe_state == "recording":
            self.swipe_trajectory.append(gaze_y / self.screen_height)

            # Auto-cancel if swipe runs too long
            if self.swipe_start_time and now - self.swipe_start_time > SWIPE_CANCEL_TIME:
                print("[swipe] Auto-cancelled (timeout)")
                self._swipe_reset()
                return None

        if self.swipe_state == "idle":
            # ---- wait for a short dwell on any key to arm the swipe ----
            if hovered is not None:
                if hovered != self.swipe_armed_key:
                    self.swipe_armed_key = hovered
                    self.swipe_arm_start = now
                elif now - self.swipe_arm_start >= SWIPE_ARM_DWELL:
                    # Armed → start recording
                    self.swipe_state     = "recording"
                    self.swipe_trajectory = []
                    self.swipe_start_time = now
                    self.swipe_end_key   = None
                    self.swipe_end_start = None
                    print(f"[swipe] Recording started on row {hovered} ({self.keys[hovered]})")
            else:
                self.swipe_armed_key = None
                self.swipe_arm_start = None

        elif self.swipe_state == "recording":
            # ---- detect dwell on end key ----
            if hovered is not None:
                if hovered != self.swipe_end_key:
                    self.swipe_end_key   = hovered
                    self.swipe_end_start = now
                elif now - self.swipe_end_start >= SWIPE_END_DWELL:
                    # End dwell complete → run DTW
                    traj      = list(self.swipe_trajectory)
                    start_row = self.swipe_armed_key
                    end_row   = self.swipe_end_key
                    if self.swipe_db is not None:
                        scored = self.swipe_db.match(traj, top_k=5,
                                                      start_row=start_row, end_row=end_row,
                                                      letter_to_row=_SWIPE_LETTER_ROW)
                    else:
                        scored = []
                    # Combined ranking: score = dtw / (usage_count + 1)^alpha
                    # Words with 0 uses → pure DTW order (divisor = 1).
                    # As usage builds, frequency pulls preferred words up.
                    if scored:
                        counts = self.swipe_usage._data["words"]
                        combined = [
                            (w, d / (counts.get(w, 0) + 1) ** SWIPE_FREQ_ALPHA)
                            for w, d in scored
                        ]
                        combined.sort(key=lambda x: x[1])
                        matches = [w for w, _ in combined]
                        # Auto-accept only when best combined score is clearly dominant
                        close = [w for w, s in combined
                                 if s <= combined[0][1] * (1 + SWIPE_RERANK_TOLERANCE)]
                    else:
                        matches = []
                        close   = []
                    print(f"[swipe] Finished. trajectory_len={len(traj)}, "
                          f"rows={start_row}→{end_row}, matches={matches}")
                    self._swipe_reset()
                    if matches:
                        self.last_swipe_matches = matches
                        # Auto-accept only when the best match is clearly dominant
                        # (only one word fell within the 15% DTW tolerance band).
                        # If multiple words are close, open corrections panel directly.
                        if len(close) <= 1:
                            self.last_swipe_auto_word = matches[0]
                            self._swipe_panel_direct  = False
                            self._accept_word(matches[0])
                            print(f"[swipe] Auto-accepted (dominant): '{matches[0]}'")
                        else:
                            self._swipe_panel_direct = True
                            print(f"[swipe] {len(close)} close matches → corrections panel")
                            self.swipe_corrections_panel.set_words(matches)
                            self.submode = "swipe_corrections"
            else:
                # Gaze left the keyboard — reset end-key tracker but keep recording
                self.swipe_end_key   = None
                self.swipe_end_start = None

        return None

    def _swipe_reset(self):
        """Clear all transient swipe state."""
        self.swipe_state     = "idle"
        self.swipe_trajectory = []
        self.swipe_armed_key  = None
        self.swipe_arm_start  = None
        self.swipe_start_time = None
        self.swipe_end_key    = None
        self.swipe_end_start  = None

    def set_mode(self, mode: str) -> None:
        """Switch between 'eyeswipe' and 'dwell' without losing typed text or TTS."""
        if mode == self.input_mode:
            return
        self.input_mode = mode
        self.keys = list(KEYBOARD_KEYS_SWIPE if mode == "eyeswipe" else KEYBOARD_KEYS_DWELL)
        self.num_keys = len(self.keys)
        self.available_height = self.screen_height - (self.num_keys - 1) * self.key_gap
        self.key_height = self.available_height // self.num_keys
        self._swipe_reset()
        self.current_key = None
        self.dwell_start_time = None
        self.submode = "typing"
        # tts reference is preserved — no reset needed

    def _replace_last_word(self, new_word: str) -> None:
        """Remove the most recently accepted word and substitute new_word."""
        self.swipe_usage.record_word(new_word.strip())
        text = self.typed_text.rstrip(" ")
        last_space = text.rfind(" ")
        if last_space >= 0:
            self.typed_text = text[: last_space + 1] + new_word + " "
        else:
            self.typed_text = new_word + " "
        with open(OUTPUT_FILE, "w") as f:
            f.write(self.typed_text)
        self.last_swipe_auto_word = new_word
        self.last_swipe_matches   = []
        self._last_accepted_word  = new_word.strip()
        self.current_ngram_predictions = self.usage.get_merged_bigram_predictions(
            self._last_accepted_word, self.bigram
        )

    def get_swipe_arm_progress(self) -> float:
        """0-1 progress toward arming a swipe (for visual feedback)."""
        if self.swipe_state != "idle" or self.swipe_arm_start is None:
            return 0.0
        return min((time.time() - self.swipe_arm_start) / SWIPE_ARM_DWELL, 1.0)

    def get_swipe_end_progress(self) -> float:
        """0-1 progress toward completing the end-key dwell."""
        if self.swipe_state != "recording" or self.swipe_end_start is None:
            return 0.0
        return min((time.time() - self.swipe_end_start) / SWIPE_END_DWELL, 1.0)

    def process_key_selection(self, key_index):
        """Process the selection of a key"""
        key_label = self.keys[key_index]

        if key_label == "SPACE":
            if self.current_predictions:
                # Mid-word: accept top trie prediction
                self._accept_word(self.current_predictions[0])
            elif self.current_ngram_predictions:
                # Between words: accept top bigram suggestion
                self._accept_word(self.current_ngram_predictions[0])
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

    def _bigram_source(self) -> str:
        """Word to use as bigram context: always the last fully accepted word."""
        if self._is_swipe_mode:
            return self.last_swipe_auto_word or ""
        return self._last_accepted_word

    def _accept_word(self, word_or_phrase: str) -> None:
        """Append word/phrase + space to typed text and reset prediction state."""
        accepted = word_or_phrase.strip()
        self._last_accepted_word = accepted.split()[-1]
        if self._is_swipe_mode:
            self.swipe_usage.record_word(accepted)
        else:
            self.usage.record_word(accepted)
        text = word_or_phrase + " "
        self.typed_text += text
        with open(OUTPUT_FILE, "a") as f:
            f.write(text)
        # Speak each accepted word immediately (cache hit = instant, miss = async+saved)
        if self.tts is not None:
            self.tts.speak_word(accepted)
        self.current_key_sequence      = []
        self.current_predictions       = []
        self.selected_prediction_idx   = 0
        # Update ngram predictions from the last word of the accepted phrase
        self.current_ngram_predictions = self.usage.get_merged_bigram_predictions(
            self._last_accepted_word, self.bigram
        )

    def _update_predictions(self):
        """Update word predictions based on current key sequence (max 5)."""
        self.current_predictions = self.usage.rerank_words(
            self.trie.search_predictions(self.current_key_sequence, max_results=5)
        )
        self.selected_prediction_idx = 0
        source = self._bigram_source()
        if source:
            self.current_ngram_predictions = self.usage.get_merged_bigram_predictions(
                source, self.bigram
            )
        # If no source word, preserve whatever bigrams were last computed

    def handle_backspace(self):
        """Handle backspace gesture (looking above screen).
        Swipe mode: deletes the last whole word.
        Dwell mode: deletes the last key from sequence, or last character.
        """
        if self._is_swipe_mode:
            text = self.typed_text.rstrip(" ")
            last_space = text.rfind(" ")
            if last_space >= 0:
                self.typed_text = text[: last_space + 1]
            else:
                self.typed_text = ""
            self.last_swipe_auto_word = None
            self.last_swipe_matches = []
            with open(OUTPUT_FILE, "w") as f:
                f.write(self.typed_text)
        else:
            if self.current_key_sequence:
                self.current_key_sequence.pop()
                self._update_predictions()
            elif self.typed_text:
                # No active key sequence — delete the whole last word
                text = self.typed_text.rstrip(" ")
                last_space = text.rfind(" ")
                if last_space >= 0:
                    self.typed_text = text[: last_space + 1]
                else:
                    self.typed_text = ""
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
            source = self.last_swipe_auto_word if self._is_swipe_mode else (
                self.current_predictions[0] if self.current_predictions else None)
            if source:
                hint = f"Next word after: \"{source}\""
                cv2.putText(canvas, hint,
                            (self.ngram_panel.panel_x, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 150, 255), 2, cv2.LINE_AA)
            return

        if self.submode == "swipe_corrections":
            self.swipe_corrections_panel.draw(canvas)
            self._draw_typed_text(canvas)
            cv2.putText(canvas, "Word Corrections",
                        (self.swipe_corrections_panel.panel_x, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            return

        # ---- normal keyboard drawing ------------------------------------
        self._draw_keyboard(canvas)
        self._draw_typed_text(canvas)
        self._draw_zone_progress(canvas, gaze_x, gaze_y)

    def _draw_keyboard(self, canvas):
        """Draw the vertical keyboard keys."""
        swipe_active = self._is_swipe_mode
        dwell_progress   = self.get_dwell_progress()
        arm_progress     = self.get_swipe_arm_progress()
        end_progress     = self.get_swipe_end_progress()
        is_recording     = self.swipe_state == "recording"

        for i, key_label in enumerate(self.keys):
            x1, y1, x2, y2 = self.get_key_bounds(i)

            # Background and border colours depend on mode
            if swipe_active and is_recording:
                # Pulsing red-tinted background while recording
                bg     = (50, 30, 30)
                color  = (120, 120, 120)
                thick  = 2
                if i == self.swipe_end_key:
                    color = (0, 200, 255)
                    thick = 4
            else:
                bg    = (40, 40, 40)
                color = (255, 255, 255)
                thick = 2
                if i == self.current_key or (swipe_active and i == self.swipe_armed_key):
                    color = (0, 255, 0)
                    thick = 4

            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)

            font_scale = 1.2 if len(key_label) <= 5 else 0.9
            (text_w, text_h), _ = cv2.getTextSize(
                key_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            text_x = x1 + (x2 - x1 - text_w) // 2
            text_y = y1 + (y2 - y1 + text_h) // 2
            cv2.putText(canvas, key_label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)

            # ---- dwell arc (normal mode) ----
            if not swipe_active and i == self.current_key and dwell_progress > 0:
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius   = int(DWELL_FEEDBACK_CIRCLE_MAX * dwell_progress)
                cv2.circle(canvas, (center_x, center_y), radius, (0, 255, 255), 3)
                angle = int(360 * dwell_progress)
                cv2.ellipse(canvas, (center_x, center_y), (radius + 10, radius + 10),
                            -90, 0, angle, (0, 255, 0), 5)

            # ---- arm arc (swipe idle — building up start dwell) ----
            if swipe_active and not is_recording and i == self.swipe_armed_key and arm_progress > 0:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                angle = int(360 * arm_progress)
                cv2.ellipse(canvas, (cx, cy), (28, 28), -90, 0, angle, (0, 255, 150), 5)

            # ---- end-dwell arc (swipe recording — building up end dwell) ----
            if swipe_active and is_recording and i == self.swipe_end_key and end_progress > 0:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                angle = int(360 * end_progress)
                cv2.ellipse(canvas, (cx, cy), (28, 28), -90, 0, angle, (0, 200, 255), 5)

        # ---- recording indicator (dot only, no text) ----
        if swipe_active and is_recording:
            rec_col = (0, 0, 200) if int(time.time() * 2) % 2 == 0 else (0, 0, 130)
            cv2.circle(canvas, (self.screen_width - 40, 40), 15, rec_col, -1)

        # Compute panel bounds clamped to not overlap keyboard
        kbd_left  = self.keyboard_x
        kbd_right = self.keyboard_x + self.keyboard_width
        PANEL_MARGIN = 10
        panel_y = 160

        screen_mid    = self.screen_height // 2
        left_panel_x  = PANEL_MARGIN
        left_panel_w  = kbd_left - PANEL_MARGIN * 2
        right_panel_x = kbd_right + PANEL_MARGIN
        right_panel_w = self.screen_width - right_panel_x - PANEL_MARGIN
        ROW_H         = 45   # original spacing
        FONT_S        = 0.75 # original font scale

        # ── Trie predictions — left panel, grows UP from screen centre ────
        # Best word (rank 1) sits at screen_mid, worse words above it.
        left_words = self.last_swipe_matches if self._is_swipe_mode else self.current_predictions
        if left_words and left_panel_w > 50:
            n_left  = len(left_words)
            # Background covers the upward extent of all words
            top_y   = screen_mid - n_left * ROW_H
            overlay = canvas.copy()
            cv2.rectangle(overlay, (left_panel_x, max(0, top_y)),
                          (left_panel_x + left_panel_w, screen_mid + ROW_H // 2),
                          (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
            # Draw worst at top, best at bottom (screen_mid)
            for draw_pos, word in enumerate(reversed(left_words)):
                rank    = n_left - draw_pos       # n→1 top→bottom
                y       = screen_mid - (n_left - 1 - draw_pos) * ROW_H
                is_best = (rank == 1)
                if is_best and not self._is_swipe_mode:
                    cv2.rectangle(canvas, (left_panel_x + 2, y - 24),
                                  (left_panel_x + left_panel_w - 2, y + 8),
                                  (0, 230, 255), 2)
                txt_color = (0, 230, 255) if is_best and not self._is_swipe_mode else (200, 200, 200)
                cv2.putText(canvas, f"{rank}. {word}", (left_panel_x + 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_S, txt_color, 2, cv2.LINE_AA)

        # ── Bigram predictions — right panel, grows DOWN from screen centre
        # Best word (rank 1) sits at screen_mid, worse words below it.
        if self.current_ngram_predictions and right_panel_w > 50:
            n_right = len(self.current_ngram_predictions)
            bot_y   = screen_mid + n_right * ROW_H
            overlay = canvas.copy()
            cv2.rectangle(overlay, (right_panel_x, screen_mid - ROW_H // 2),
                          (right_panel_x + right_panel_w, min(self.screen_height, bot_y)),
                          (30, 10, 40), -1)
            cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
            for idx, word in enumerate(self.current_ngram_predictions):
                y = screen_mid + idx * ROW_H
                cv2.putText(canvas, f"{idx + 1}. {word}", (right_panel_x + 8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_S, (255, 200, 255), 2, cv2.LINE_AA)

        # ── Key sequence display ───────────────────────────────────────────
        if self.current_key_sequence:
            seq_text = "Keys: " + "-".join([self.keys[k] for k in self.current_key_sequence])
            cv2.putText(canvas, seq_text, (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)


    def _draw_typed_text(self, canvas):
        """Draw the typed-text panel to the right of the keyboard with line wrapping."""
        MARGIN = 10
        FONT   = cv2.FONT_HERSHEY_SIMPLEX
        SCALE  = 0.75
        THICK  = 2
        LINE_H = 34

        kbd_right   = self.keyboard_x + self.keyboard_width
        panel_x     = kbd_right + MARGIN
        panel_width = self.screen_width - panel_x - MARGIN
        panel_y_top = MARGIN
        panel_y_bot = int(self.screen_height * 0.25)

        if panel_width < 60:
            return

        cv2.rectangle(canvas, (panel_x, panel_y_top), (panel_x + panel_width, panel_y_bot),
                      (0, 0, 0), -1)
        cv2.rectangle(canvas, (panel_x, panel_y_top), (panel_x + panel_width, panel_y_bot),
                      (100, 100, 100), 2)

        display_text = self.typed_text
        if self.current_predictions:
            display_text += "[" + self.current_predictions[0] + "]"

        lines = _wrap_text(display_text.strip(), FONT, SCALE, THICK, panel_width - 20)

        # Render from bottom up so newest text is always visible
        y = panel_y_bot - 12
        for line in reversed(lines):
            if y - LINE_H < panel_y_top + 5:
                break
            cv2.putText(canvas, line, (panel_x + 10, y),
                        FONT, SCALE, (0, 255, 0), THICK, cv2.LINE_AA)
            y -= LINE_H

    def _draw_zone_progress(self, canvas, gaze_x, gaze_y):
        """Draw up/down zone dwell progress bars."""
        bar_h = 8
        cx = self.screen_width // 2

        if self.up_zone_start is not None:
            progress = min((time.time() - self.up_zone_start) / self.zone_dwell_time, 1.0)
            bar_w = int(self.screen_width * progress)
            cv2.rectangle(canvas,
                          (cx - bar_w // 2, 0),
                          (cx + bar_w // 2, bar_h),
                          (0, 200, 255), -1)

        if self.down_zone_start is not None:
            progress = min((time.time() - self.down_zone_start) / self.zone_dwell_time, 1.0)
            bar_w = int(self.screen_width * progress)
            cv2.rectangle(canvas,
                          (cx - bar_w // 2, self.screen_height - bar_h),
                          (cx + bar_w // 2, self.screen_height),
                          (255, 100, 200), -1)


# ============================================================
#                   MENU SYSTEM
# ============================================================

# Menu state tracking
menu_options = ["keyboard", "fixed_phrases", "home_iot", "mobile_phone", "selection_mode"]
NUM_MENU_BUTTONS = 5
menu_dwell_start = [None] * NUM_MENU_BUTTONS
MENU_DWELL_TIME = 1.2
MENU_BUTTON_WIDTH = 400
MENU_BUTTON_GAP = 15

# Active input mode — persists across menu/keyboard transitions; default EyeSwipe
selected_input_mode = "eyeswipe"

# Selection-mode sub-screen dwell state
_sel_mode_dwell_start = [None, None]   # [eyeswipe, dwell]
SEL_MODE_OPTIONS = ["eyeswipe", "dwell"]


def get_menu_button_bounds(option_index, screen_width, screen_height):
    """Get the bounding box for a menu button (4-button layout)."""
    n = NUM_MENU_BUTTONS
    available_height = screen_height - (n - 1) * MENU_BUTTON_GAP
    button_height = available_height // n
    y1 = option_index * (button_height + MENU_BUTTON_GAP)
    y2 = y1 + button_height
    x1 = (screen_width - MENU_BUTTON_WIDTH) // 2
    x2 = x1 + MENU_BUTTON_WIDTH
    return x1, y1, x2, y2


def get_hovered_menu_option(x, y, screen_width, screen_height):
    if x is None or y is None:
        return None
    for i in range(NUM_MENU_BUTTONS):
        x1, y1, x2, y2 = get_menu_button_bounds(i, screen_width, screen_height)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None


def update_menu(gaze_x, gaze_y, screen_width, screen_height):
    """Update menu dwell state and return selected option string or None."""
    global menu_dwell_start
    hovered = get_hovered_menu_option(gaze_x, gaze_y, screen_width, screen_height)
    for i in range(NUM_MENU_BUTTONS):
        if i != hovered:
            menu_dwell_start[i] = None
    if hovered is None:
        return None
    if menu_dwell_start[hovered] is None:
        menu_dwell_start[hovered] = time.time()
    if time.time() - menu_dwell_start[hovered] >= MENU_DWELL_TIME:
        menu_dwell_start[hovered] = None
        return menu_options[hovered]
    return None


def draw_menu(canvas, screen_width, screen_height, gaze_x=None, gaze_y=None):
    """Draw the main menu with four centered buttons."""
    hovered = get_hovered_menu_option(gaze_x, gaze_y, screen_width, screen_height)

    menu_labels = ["KEYBOARD", "FIXED PHRASES", "HOME IoT", "MOBILE PHONE", "INPUT MODE"]
    menu_colors = [(0, 200, 0), (0, 215, 255), (0, 200, 100), (255, 200, 0), (255, 130, 0)]

    for i, label in enumerate(menu_labels):
        x1, y1, x2, y2 = get_menu_button_bounds(i, screen_width, screen_height)
        is_hovered = (i == hovered)
        bg_color = (80, 80, 80) if is_hovered else (40, 40, 40)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

        border_color = (255, 255, 255) if is_hovered else (100, 100, 100)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 4 if is_hovered else 2)

        if is_hovered and menu_dwell_start[i] is not None:
            progress = min((time.time() - menu_dwell_start[i]) / MENU_DWELL_TIME, 1.0)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(canvas, (cx, cy - 55), (45, 45), 0, -90,
                        -90 + int(360 * progress), (0, 255, 0), 7)

        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        text_x = (x1 + x2 - size[0]) // 2
        text_y = (y1 + y2) // 2 + size[1] // 2
        cv2.putText(canvas, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, menu_colors[i], 3, cv2.LINE_AA)

        if i == 4:
            # Show currently active mode beneath the button label
            _MODE_NAMES = {"eyeswipe": "EyeSwipe", "dwell": "Dwell"}
            mode_str = _MODE_NAMES.get(selected_input_mode, selected_input_mode)
            sub = f"Active: {mode_str}"
            s, _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
            cv2.putText(canvas, sub, ((x1 + x2 - s[0]) // 2, text_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 220), 2, cv2.LINE_AA)

    inst = "Look UP or DOWN for HOME"
    s, _ = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(canvas, inst, ((screen_width - s[0]) // 2, screen_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)


# ---- Selection-mode sub-screen ----------------------------------------

def _sel_button_bounds(idx, screen_width, screen_height):
    """Two large buttons stacked vertically for EyeSwipe / Dwell selection."""
    margin  = 60
    gap     = 20
    n       = len(SEL_MODE_OPTIONS)
    btn_w   = screen_width - 2 * margin
    total_h = screen_height - 2 * margin - 80   # leave room for title at top
    btn_h   = (total_h - gap * (n - 1)) // n
    x1 = margin
    x2 = x1 + btn_w
    y1 = 100 + margin // 2 + idx * (btn_h + gap)
    y2 = y1 + btn_h
    return x1, y1, x2, y2


def update_selection_mode(gaze_x, gaze_y, screen_width, screen_height):
    """Returns selected mode string ('eyeswipe'/'dwell') or None."""
    global _sel_mode_dwell_start
    n = len(SEL_MODE_OPTIONS)
    hovered = None
    if gaze_x is not None and gaze_y is not None:
        for i in range(n):
            x1, y1, x2, y2 = _sel_button_bounds(i, screen_width, screen_height)
            if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                hovered = i
                break
    for i in range(n):
        if i != hovered:
            _sel_mode_dwell_start[i] = None
    if hovered is None:
        return None
    if _sel_mode_dwell_start[hovered] is None:
        _sel_mode_dwell_start[hovered] = time.time()
    if time.time() - _sel_mode_dwell_start[hovered] >= MENU_DWELL_TIME:
        _sel_mode_dwell_start[hovered] = None
        return SEL_MODE_OPTIONS[hovered]
    return None


def draw_selection_mode(canvas, screen_width, screen_height, gaze_x=None, gaze_y=None):
    """Draw the input-mode selection sub-screen."""
    labels        = ["EYESWIPE", "DWELL"]
    colors_active = [(0, 230, 255), (100, 255, 100)]
    bg_active     = [(50, 100, 110), (40, 90, 40)]
    n = len(SEL_MODE_OPTIONS)

    hovered = None
    if gaze_x is not None and gaze_y is not None:
        for i in range(n):
            x1, y1, x2, y2 = _sel_button_bounds(i, screen_width, screen_height)
            if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
                hovered = i

    for i, label in enumerate(labels):
        x1, y1, x2, y2 = _sel_button_bounds(i, screen_width, screen_height)
        mode_key    = SEL_MODE_OPTIONS[i]
        is_selected = (selected_input_mode == mode_key)
        is_hovered  = (i == hovered)

        bg = bg_active[i] if is_selected else (60, 60, 60) if is_hovered else (30, 30, 30)
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)

        border_col   = colors_active[i] if is_selected else (200, 200, 200) if is_hovered else (80, 80, 80)
        border_thick = 5 if is_selected else 3 if is_hovered else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_thick)

        if is_selected:
            badge = "SELECTED"
            bs, _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.putText(canvas, badge, ((x1 + x2 - bs[0]) // 2, y1 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors_active[i], 2, cv2.LINE_AA)

        fs = 1.3
        s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
        cv2.putText(canvas, label, ((x1 + x2 - s[0]) // 2, (y1 + y2) // 2 + s[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    colors_active[i] if is_selected else (200, 200, 200), 3, cv2.LINE_AA)

        if is_hovered and _sel_mode_dwell_start[i] is not None:
            prog = min((time.time() - _sel_mode_dwell_start[i]) / MENU_DWELL_TIME, 1.0)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(canvas, (cx, cy + 60), (35, 35), 0, -90,
                        -90 + int(360 * prog), (0, 255, 100), 5)

    title = "SELECT INPUT MODE"
    ts, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(canvas, title, ((screen_width - ts[0]) // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "Dwell on a mode to select it — returns to menu",
                (screen_width // 2 - 320, screen_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)


# ============================================================
#             EMERGENCY CONFIRMATION DIALOG
# ============================================================

CONFIRM_DWELL_TIME = 1.2
confirm_options = ["yes", "no"]
confirm_dwell_start = [None] * len(confirm_options)


def _confirm_button_bounds(option_index: int, screen_width: int, screen_height: int):
    """Full-width buttons matching the selection-mode style."""
    margin = 60
    gap = 20
    n = len(confirm_options)
    btn_w = screen_width - 2 * margin
    total_h = screen_height - 2 * margin - 80
    btn_h = (total_h - gap * (n - 1)) // n
    x1 = margin
    x2 = x1 + btn_w
    y1 = 100 + margin // 2 + option_index * (btn_h + gap)
    y2 = y1 + btn_h
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

    labels = ["YES", "NO"]
    colors_active = [(0, 230, 100), (80, 80, 255)]
    bg_active = [(30, 80, 40), (60, 30, 30)]

    # Title
    title = "EMERGENCY CALL"
    ts, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(canvas, title, ((screen_width - ts[0]) // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    hint = "Are you sure?  Dwell on a button to confirm."
    hs, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(canvas, hint, ((screen_width - hs[0]) // 2, screen_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

    for i, label in enumerate(labels):
        x1, y1, x2, y2 = _confirm_button_bounds(i, screen_width, screen_height)
        is_hovered = (i == hovered)

        bg = bg_active[i] if is_hovered else (30, 30, 30)
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)

        border_col = colors_active[i] if is_hovered else (80, 80, 80)
        border_thick = 5 if is_hovered else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_thick)

        # Dwell arc below centre
        if is_hovered and confirm_dwell_start[i] is not None:
            prog = min((time.time() - confirm_dwell_start[i]) / CONFIRM_DWELL_TIME, 1.0)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(canvas, (cx, cy + 60), (35, 35), 0, -90,
                        -90 + int(360 * prog), (0, 255, 100), 5)

        fs = 1.3
        s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
        cv2.putText(canvas, label, ((x1 + x2 - s[0]) // 2, (y1 + y2) // 2 + s[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, colors_active[i] if is_hovered else (200, 200, 200), 3, cv2.LINE_AA)


# ============================================================
#                   IoT SUBMENU (PHONE UI)
# ============================================================

IOT_DWELL_TIME = 1.2
iot_options = ["show_phone_ui", "emergency_call"]
iot_dwell_start = [None] * len(iot_options)


def launch_scrcpy(device_serial: str | None = None) -> subprocess.Popen | None:
    """Launch scrcpy and return the process handle (or None on failure)."""
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
        return None
    cmd = [scrcpy]
    if device_serial:
        cmd += ["-s", device_serial]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        print("[scrcpy] launched")
        # On macOS, bring the scrcpy window to the front after a short delay
        if sys.platform == "darwin":
            import threading
            def _bring_to_front():
                time.sleep(2.0)
                try:
                    subprocess.run(
                        ["osascript", "-e",
                         'tell application "System Events" to set frontmost of '
                         '(first process whose name contains "scrcpy") to true'],
                        capture_output=True, timeout=5,
                    )
                except Exception:
                    pass
            threading.Thread(target=_bring_to_front, daemon=True).start()
        return proc
    except Exception as e:
        print(f"[scrcpy] failed to launch: {e}")
        return None


def _iot_button_bounds(option_index: int, screen_width: int, screen_height: int):
    """Full-width buttons matching the selection-mode style."""
    margin = 60
    gap = 20
    n = len(iot_options)
    btn_w = screen_width - 2 * margin
    total_h = screen_height - 2 * margin - 80
    btn_h = (total_h - gap * (n - 1)) // n
    x1 = margin
    x2 = x1 + btn_w
    y1 = 100 + margin // 2 + option_index * (btn_h + gap)
    y2 = y1 + btn_h
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

    labels = ["SHOW PHONE UI", "EMERGENCY CALL"]
    colors_active = [(0, 230, 255), (100, 255, 100), (0, 0, 255)]

    # Title
    title = "MOBILE PHONE"
    ts, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(canvas, title, ((screen_width - ts[0]) // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    subtitle = "Dwell on a button to select"
    ss, _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(canvas, subtitle, ((screen_width - ss[0]) // 2, screen_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

    for i, label in enumerate(labels):
        x1, y1, x2, y2 = _iot_button_bounds(i, screen_width, screen_height)
        is_hovered = (i == hovered)

        bg = (60, 60, 60) if is_hovered else (30, 30, 30)
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)

        border_col = (200, 200, 200) if is_hovered else (80, 80, 80)
        border_thick = 3 if is_hovered else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_thick)

        # Dwell arc below centre
        if is_hovered and iot_dwell_start[i] is not None:
            prog = min((time.time() - iot_dwell_start[i]) / IOT_DWELL_TIME, 1.0)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(canvas, (cx, cy + 60), (35, 35), 0, -90,
                        -90 + int(360 * prog), (0, 255, 100), 5)

        fs = 1.3
        s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
        cv2.putText(canvas, label, ((x1 + x2 - s[0]) // 2, (y1 + y2) // 2 + s[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, colors_active[i] if is_hovered else (200, 200, 200), 3, cv2.LINE_AA)


# ============================================================
#             HOME IoT LIGHTS SUBMENU
# ============================================================

lights_options = ["lights_on", "lights_off"]
lights_dwell_start = [None, None]


def _lights_button_bounds(option_index: int, screen_width: int, screen_height: int):
    """Two full-width buttons for LIGHTS ON / LIGHTS OFF."""
    margin = 60
    gap = 20
    n = len(lights_options)
    btn_w = screen_width - 2 * margin
    total_h = screen_height - 2 * margin - 80
    btn_h = (total_h - gap * (n - 1)) // n
    x1 = margin
    x2 = x1 + btn_w
    y1 = 100 + margin // 2 + option_index * (btn_h + gap)
    y2 = y1 + btn_h
    return x1, y1, x2, y2


def _get_hovered_lights_option(x, y, screen_width, screen_height):
    if x is None or y is None:
        return None
    for i in range(len(lights_options)):
        x1, y1, x2, y2 = _lights_button_bounds(i, screen_width, screen_height)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None


def update_lights_menu(gaze_x, gaze_y, screen_width, screen_height):
    global lights_dwell_start
    hovered = _get_hovered_lights_option(gaze_x, gaze_y, screen_width, screen_height)
    for i in range(len(lights_options)):
        if i != hovered:
            lights_dwell_start[i] = None
    if hovered is None:
        return None
    if lights_dwell_start[hovered] is None:
        lights_dwell_start[hovered] = time.time()
    if time.time() - lights_dwell_start[hovered] >= CONFIRM_DWELL_TIME:
        lights_dwell_start[hovered] = None
        return lights_options[hovered]
    return None


def draw_lights_menu(canvas, screen_width, screen_height, gaze_x=None, gaze_y=None):
    hovered = _get_hovered_lights_option(gaze_x, gaze_y, screen_width, screen_height)

    labels = ["LIGHTS ON", "LIGHTS OFF"]
    colors_active = [(0, 230, 100), (80, 80, 255)]
    bg_active = [(30, 80, 40), (60, 30, 30)]

    title = "HOME IoT — LIGHTS"
    ts, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.putText(canvas, title, ((screen_width - ts[0]) // 2, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 100), 3, cv2.LINE_AA)
    hint = "Dwell on a button to toggle lights"
    hs, _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.putText(canvas, hint, ((screen_width - hs[0]) // 2, screen_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 160, 160), 1, cv2.LINE_AA)

    for i, label in enumerate(labels):
        x1, y1, x2, y2 = _lights_button_bounds(i, screen_width, screen_height)
        is_hovered = (i == hovered)

        bg = bg_active[i] if is_hovered else (30, 30, 30)
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.85, canvas, 0.15, 0, canvas)

        border_col = colors_active[i] if is_hovered else (80, 80, 80)
        border_thick = 5 if is_hovered else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, border_thick)

        if is_hovered and lights_dwell_start[i] is not None:
            prog = min((time.time() - lights_dwell_start[i]) / CONFIRM_DWELL_TIME, 1.0)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.ellipse(canvas, (cx, cy + 60), (35, 35), 0, -90,
                        -90 + int(360 * prog), (0, 255, 100), 5)

        fs = 1.3
        s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)
        cv2.putText(canvas, label, ((x1 + x2 - s[0]) // 2, (y1 + y2) // 2 + s[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs,
                    colors_active[i] if is_hovered else (200, 200, 200), 3, cv2.LINE_AA)


def run_demo():
    """Main demo function with menu and integrated keyboard"""
    global selected_input_mode
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
        blink_threshold_ratio=0.75,
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
        elif calibration_method == "vertical-only":
            run_vertical_only_calibration(gaze_estimator, camera_index=camera_index,
                                          multi_pose=mp)
        elif calibration_method == "vertical-center":
            run_vertical_center_calibration(gaze_estimator, camera_index=camera_index,
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

    # Load swipe template DB — explicit flag or auto-detect swipe_templates.json
    swipe_db = None
    swipe_model_path = getattr(args, "swipe_model", None)
    if not swipe_model_path:
        default_path = Path("swipe_templates.json")
        if default_path.exists():
            swipe_model_path = str(default_path)
    if swipe_model_path and os.path.isfile(swipe_model_path):
        swipe_db = SwipeTemplateDB()
        swipe_db.load(swipe_model_path)

    # Auto-init TTS if speaker.wav is present (background warm — non-blocking)
    tts_speaker = None
    _repo_root   = Path(__file__).resolve().parents[3]
    _speaker_wav = _repo_root / "speaker.wav"
    if _speaker_wav.exists():
        tts_speaker = OpenVoiceSpeaker(
            openvoice_repo=_repo_root / "OpenVoice",
            work_dir=_repo_root / "openvoice_runtime",
            speaker_wav=_speaker_wav,
            cache_dir=_repo_root / "speech_cache",
        )
        print(f"[tts] Initialised — warming models in background")
    else:
        print(f"[tts] speaker.wav not found at {_speaker_wav} — TTS disabled")

    # Initialize keyboard controller with default EyeSwipe mode (changed via UI)
    keyboard = KeyboardController(screen_width, screen_height,
                                  swipe_db=swipe_db,
                                  input_mode=selected_input_mode,
                                  tts=tts_speaker)

    # Mode tracking: "menu" | "keyboard" | "fixed_phrases" | "home_iot" | "lights_menu"
    #                | "mobile_phone" | "emergency_confirm" | "selection_mode"
    current_mode = "menu"

    # Fixed phrases panel
    phrases_panel = FixedPhrasesPanel(screen_width, screen_height)

    # Scrcpy process tracking
    scrcpy_process: subprocess.Popen | None = None

    # ADB / IoT / Emergency state
    emergency_cfg = EmergencyCallConfig(
        device_serial=None,
        phone_number="+971554611264",
        message="Hello, This is an emergency, I am a person of determination with locked in syndrome and this is a preset message, I repeat, This is an emergency, I am a person of determination with locked in syndrome and this is a preset message. Please come to my location in Khalifa University urgently",
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

    cam_width, cam_height = 220, 165
    BORDER = 2
    MARGIN = 20
    cursor_alpha = 0.0
    cursor_step = 0.05

    # Initialize scan path tracking
    scan_path_points = deque(maxlen=scan_path_max if scan_path_enabled else 0)
    scan_path_timestamps = deque(maxlen=scan_path_max if scan_path_enabled else 0)


    # Long-blink detection — action fires on eye-open, based on total duration:
    #   >= 1.0 s → complete sentence + TTS
    #   >= 2.0 s → send to LLM + TTS result
    # Sounds play mid-blink at each threshold as tactile feedback only.
    long_blink_start: float | None = None
    back_gesture_start: float | None = None  # look-up-beyond-screen → back to menu
    BACK_GESTURE_TIME = 2.0
    BACK_GESTURE_MODES = {"selection_mode", "lights_menu", "mobile_phone",
                          "fixed_phrases", "emergency_confirm"}
    blink_sound_1s_played = False    # beep fired at 1 s mark (feedback only)
    blink_sound_2s_played = False    # beep fired at 2 s mark (feedback only)
    ai_blink_flash_until = 0.0
    llm_generating = False
    llm_banner_until = 0.0
    llm_result_clear_until: float | None = None
    new_sentence_flash_until = 0.0

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
                if getattr(gaze_estimator, "vertical_only", False):
                    x = gaze_estimator.vertical_center_x
                    y = int(gaze_point)
                else:
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

            # ── Long-blink detection (keyboard only) ─────────────────────
            # Sounds fire mid-blink as feedback so the user knows when to open.
            # The actual action is decided on eye-open from total blink duration:
            #   >= 2.0 s → LLM expand → TTS result
            #   >= 1.0 s → complete sentence → TTS
            if current_mode == "keyboard":
                if blink_detected:
                    if long_blink_start is None:
                        long_blink_start = time.time()
                        blink_sound_1s_played = False
                        blink_sound_2s_played = False
                    else:
                        elapsed = time.time() - long_blink_start
                        # Play beep at 1 s mark (feedback only — no action yet)
                        if not blink_sound_1s_played and elapsed >= AI_BLINK_DURATION:
                            blink_sound_1s_played = True
                            if SENTENCE_COMPLETE_SOUND.exists():
                                subprocess.Popen(["afplay", str(SENTENCE_COMPLETE_SOUND)])
                        # Play second beep at 2 s mark (feedback only — no action yet)
                        if not blink_sound_2s_played and elapsed >= SENTENCE_BLINK_DURATION:
                            blink_sound_2s_played = True
                            if SENTENCE_COMPLETE_SOUND.exists():
                                subprocess.Popen(["afplay", str(SENTENCE_COMPLETE_SOUND)])
                else:
                    # Eyes just opened — decide action from total blink duration
                    if long_blink_start is not None:
                        blink_duration = time.time() - long_blink_start
                        if blink_duration >= SENTENCE_BLINK_DURATION:
                            # ── 2 s blink: send to LLM ──
                            if not llm_generating:
                                keywords = keyboard.typed_text.strip()
                                if keyboard.current_predictions:
                                    keywords += " " + keyboard.current_predictions[0]
                                keywords = keywords.strip()
                                if keywords:
                                    keyboard.usage.record_sentence_bigrams(keyboard.typed_text.strip())
                                    if keyboard._is_swipe_mode:
                                        keyboard.swipe_usage.record_word(keyboard.typed_text.strip())
                                    else:
                                        keyboard.usage.record_word(keyboard.typed_text.strip())
                                    print(f"[blink-2.0s] Sending to LLM: '{keywords}'")
                                    ai_blink_flash_until = time.time() + 2.0
                                    generate_sentence_async(keywords)
                                    llm_generating = True
                                    llm_banner_until = time.time() + 30
                                    keyboard.typed_text = ""
                                    keyboard.current_key_sequence = []
                                    keyboard.current_predictions = []
                                    keyboard.current_ngram_predictions = []
                                    keyboard.selected_prediction_idx = 0
                                    keyboard.last_swipe_auto_word = None
                                    keyboard.last_swipe_matches = []
                                    with open(OUTPUT_FILE, "w") as f:
                                        f.write("")
                                else:
                                    print("[blink-2.0s] No text to send to LLM")
                        elif blink_duration >= AI_BLINK_DURATION:
                            # ── 1 s blink: complete sentence + TTS ──
                            sentence = keyboard.typed_text.strip()
                            if sentence:
                                keyboard.usage.record_sentence_bigrams(sentence)
                                if keyboard._is_swipe_mode:
                                    keyboard.swipe_usage.record_word(sentence)
                                else:
                                    keyboard.usage.record_word(sentence)
                                print(f"[blink-1.0s] Sentence complete: '{sentence}'")
                                if keyboard.tts is not None:
                                    keyboard.tts.speak_sentence(sentence)
                            keyboard.typed_text = ""
                            keyboard.current_key_sequence = []
                            keyboard.current_predictions = []
                            keyboard.current_ngram_predictions = []
                            keyboard.last_swipe_auto_word = None
                            keyboard.last_swipe_matches = []
                            with open(OUTPUT_FILE, "w") as f:
                                f.write("")
                            new_sentence_flash_until = time.time() + 1.5
                            print("[blink-1.0s] New sentence started")
                    long_blink_start = None
                    blink_sound_1s_played = False
                    blink_sound_2s_played = False
            else:
                # Reset blink state when not in keyboard mode
                long_blink_start = None
                blink_sound_1s_played = False
                blink_sound_2s_played = False


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
                    if keyboard.tts is not None:
                        keyboard.tts.speak_sentence(llm_result.strip())
                    llm_result_clear_until = time.time() + 4.0

            # ── Auto-clear LLM result after display period ────────────
            if llm_result_clear_until is not None and time.time() >= llm_result_clear_until:
                keyboard.typed_text = ""
                keyboard.current_key_sequence = []
                keyboard.current_predictions = []
                keyboard.current_ngram_predictions = []
                keyboard.selected_prediction_idx = 0
                keyboard.last_swipe_auto_word = None
                keyboard.last_swipe_matches = []
                with open(OUTPUT_FILE, "w") as f:
                    f.write("")
                llm_result_clear_until = None
                new_sentence_flash_until = time.time() + 1.5

            # ── Auto-detect dead scrcpy process ──────────────────────
            if scrcpy_process is not None and scrcpy_process.poll() is not None:
                print("[scrcpy] Process exited")
                scrcpy_process = None


            # Update based on current mode
            if current_mode == "menu":
                selected_option = update_menu(x_pred, y_pred, screen_width, screen_height)
                if selected_option:
                    if selected_option == "selection_mode":
                        current_mode = "selection_mode"
                    elif selected_option == "keyboard":
                        keyboard.set_mode(selected_input_mode)
                        current_mode = "keyboard"
                    elif selected_option == "home_iot":
                        current_mode = "lights_menu"
                    elif selected_option == "mobile_phone":
                        current_mode = "mobile_phone"
                    elif selected_option == "fixed_phrases":
                        current_mode = "fixed_phrases"
                    else:
                        current_mode = selected_option
                    print(f"[menu] Selected: {selected_option}")
            elif current_mode == "selection_mode":
                chosen = update_selection_mode(x_pred, y_pred, screen_width, screen_height)
                if chosen is not None:
                    selected_input_mode = chosen
                    keyboard.set_mode(chosen)
                    current_mode = "menu"
                    print(f"[selection_mode] Input mode set to: {chosen}")
            elif current_mode == "lights_menu":
                choice = update_lights_menu(x_pred, y_pred, screen_width, screen_height)
                if choice == "lights_on":
                    lights_on_async()
                    print("[lights] Turning ON")
                    current_mode = "menu"
                elif choice == "lights_off":
                    lights_off_async()
                    print("[lights] Turning OFF")
                    current_mode = "menu"
            elif current_mode == "mobile_phone":
                # If scrcpy is running, skip UI dwell detection (only gestures active)
                if scrcpy_process is None or scrcpy_process.poll() is not None:
                    scrcpy_process = None  # clean up dead ref
                    choice = update_iot_menu(x_pred, y_pred, screen_width, screen_height)
                    if choice == "show_phone_ui":
                        try:
                            phone_cfg = ensure_device(phone_cfg)
                        except Exception as e:
                            print(f"[scrcpy] No device: {e}")
                            phone_cfg = EmergencyCallConfig(device_serial=None)
                        scrcpy_process = launch_scrcpy(device_serial=phone_cfg.device_serial)
                    elif choice == "play_flappy_bird":
                        try:
                            phone_cfg = ensure_device(phone_cfg)
                            scrcpy_process = launch_scrcpy(device_serial=phone_cfg.device_serial)
                            open_url(phone_cfg, "https://flappybird.io/")
                            time.sleep(2.0)
                            w, h = get_phone_screen_size(phone_cfg)
                            flappy_center = (w // 2, h // 2)
                            flappy_active = True
                            print(f"[flappy] Enabled blink->tap at {flappy_center[0]},{flappy_center[1]}")
                        except Exception as e:
                            flappy_active = False
                            flappy_center = None
                            print(f"[flappy] Failed to start: {e}")
                    elif choice == "emergency_call":
                        current_mode = "emergency_confirm"
                        print("[menu] Emergency confirm opened")
            elif current_mode == "fixed_phrases":
                selected_phrase = phrases_panel.update(x_pred, y_pred)
                if selected_phrase is not None:
                    print(f"[fixed_phrases] Selected: {selected_phrase}")
                    if tts_speaker is not None:
                        tts_speaker.speak_sentence(selected_phrase)
                    current_mode = "menu"
            elif current_mode == "emergency_confirm":
                decision = update_emergency_confirm(x_pred, y_pred, screen_width, screen_height)
                if decision == "yes":
                    print("[menu] Emergency call confirmed")
                    try:
                        trigger_emergency_call(
                            emergency_cfg,
                            speak_fn=tts_speaker.speak_sentence if tts_speaker else None,
                        )
                    except Exception as e:
                        print(f"[emergency] Failed: {e}")
                    keyboard.set_mode(selected_input_mode)
                    current_mode = "keyboard"
                elif decision == "no":
                    print("[menu] Emergency call cancelled")
                    current_mode = "menu"
            elif current_mode == "keyboard":
                selected_key = keyboard.update(x_pred, y_pred)
                if selected_key == "__MENU__":
                    current_mode = "menu"
                    print("[keyboard] Returning to menu (bigram home gesture)")
                elif selected_key is not None:
                    print(f"[keyboard] Key selected: {keyboard.keys[selected_key] if isinstance(selected_key, int) else selected_key}")

            # Blink-to-tap bridge (mobile phone flappy mode)
            if current_mode == "mobile_phone" and flappy_active and flappy_center and blink_detected:
                now_ts = time.time()
                if now_ts - last_flappy_tap >= FLAPPY_TAP_COOLDOWN_S:
                    try:
                        tap(phone_cfg, flappy_center[0], flappy_center[1])
                        last_flappy_tap = now_ts
                        print(f"[flappy] tap @{flappy_center[0]},{flappy_center[1]}")
                    except Exception as e:
                        flappy_active = False
                        print(f"[flappy] Tap failed, disabled: {e}")

            # ── Back gesture: look above screen 3 s → return to menu ────────
            if current_mode in BACK_GESTURE_MODES:
                if y_pred is not None and y_pred < -40:
                    if back_gesture_start is None:
                        back_gesture_start = time.time()
                    elif time.time() - back_gesture_start >= BACK_GESTURE_TIME:
                        back_gesture_start = None
                        current_mode = "menu"
                        print(f"[back-gesture] Returned to menu")
                else:
                    back_gesture_start = None
            else:
                back_gesture_start = None

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
            elif current_mode == "selection_mode":
                draw_selection_mode(canvas, screen_width, screen_height, x_pred, y_pred)
            elif current_mode == "lights_menu":
                draw_lights_menu(canvas, screen_width, screen_height, x_pred, y_pred)
            elif current_mode == "mobile_phone":
                if scrcpy_process is not None and scrcpy_process.poll() is None:
                    # Scrcpy active — minimal overlay
                    msg1 = "scrcpy active"
                    msg2 = "Look UP to close  |  Look DOWN for HOME"
                    s1, _ = cv2.getTextSize(msg1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    s2, _ = cv2.getTextSize(msg2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.putText(canvas, msg1,
                                ((screen_width - s1[0]) // 2, screen_height // 2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2, cv2.LINE_AA)
                    cv2.putText(canvas, msg2,
                                ((screen_width - s2[0]) // 2, screen_height // 2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
                else:
                    draw_iot_menu(canvas, screen_width, screen_height, x_pred, y_pred)
            elif current_mode == "fixed_phrases":
                phrases_panel.draw(canvas)
            elif current_mode == "emergency_confirm":
                draw_emergency_confirm(canvas, screen_width, screen_height, x_pred, y_pred)
            elif current_mode == "keyboard":
                keyboard.draw(canvas, x_pred, y_pred)

            # Back-gesture progress bar (top edge)
            if back_gesture_start is not None:
                prog = min((time.time() - back_gesture_start) / BACK_GESTURE_TIME, 1.0)
                bar_w = max(1, int(screen_width * prog))
                cv2.rectangle(canvas, (0, 0), (bar_w, 12), (0, 140, 255), -1)

            # New-sentence flash (long blink triggered)
            if time.time() < new_sentence_flash_until:
                msg = "NEW SENTENCE"
                ts = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)[0]
                bx = (screen_width - ts[0]) // 2
                by = screen_height // 2 - 20
                cv2.rectangle(canvas, (bx - 20, by - 50), (bx + ts[0] + 20, by + 20),
                              (20, 20, 60), -1)
                cv2.rectangle(canvas, (bx - 20, by - 50), (bx + ts[0] + 20, by + 20),
                              (0, 200, 255), 3)
                cv2.putText(canvas, msg, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                            1.6, (0, 200, 255), 3, cv2.LINE_AA)


            # Draw cursor if enabled via --cursor flag
            if cursor_enabled and x_pred is not None and y_pred is not None and cursor_alpha > 0:
                draw_cursor(canvas, x_pred, y_pred, cursor_alpha)

            # Draw camera thumbnail (bottom-left)
            thumb = make_thumbnail(frame, size=(cam_width, cam_height), border=BORDER)
            h, w = thumb.shape[:2]
            canvas[-h - MARGIN : -MARGIN, MARGIN : MARGIN + w] = thumb

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

            # ── LLM / blink status overlay (keyboard mode only) ────────
            if current_mode == "keyboard":
                right_x = screen_width - 380
                if time.time() < ai_blink_flash_until:
                    cv2.putText(canvas, "ACTIVATED",
                                (right_x, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if current_mode == "keyboard":
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
