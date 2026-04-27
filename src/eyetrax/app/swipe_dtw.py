"""
DTW-based swipe-typing engine for the vertical eye-tracking keyboard.

Each word in the vocabulary can have one or more *template trajectories*
stored as a list of normalised y-positions (0 = top of screen, 1 = bottom).
At inference the observed swipe trajectory is compared to every template
with Dynamic Time Warping; the words whose templates are closest to the
observation are returned as predictions.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared swipe dwell constants (imported by keyboard_demo and train_swipe)
# Change here to affect both.
# ---------------------------------------------------------------------------
SWIPE_ARM_DWELL = 1.0   # seconds to dwell on start key to begin a swipe
SWIPE_END_DWELL = 1.0   # seconds to dwell on end  key to finish a swipe

import numpy as np

# Keyboard row layout (must match keyboard_demo.py / train_swipe.py)
_SWIPE_ROWS = ["abcd", "efgh", "ijkl", "mnop", "qrstu", "vwxyz"]
_LETTER_TO_ROW: dict[str, int] = {ch: i for i, row in enumerate(_SWIPE_ROWS) for ch in row}

BASE_POINTS  = 12   # resample points per unique row transition
MIN_RESAMPLE = 24   # floor so very short words still have resolution


# ---------------------------------------------------------------------------
# DTW core
# ---------------------------------------------------------------------------

def _dtw_distance(s1: list[float], s2: list[float]) -> float:
    """Return the DTW distance between two 1-D sequences."""
    n, m = len(s1), len(s2)
    # Use a flat array for speed (avoids Python list-of-lists overhead)
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _word_row_transitions(word: str) -> int:
    """Count unique consecutive row regions visited by a word's letters."""
    rows = [_LETTER_TO_ROW[c] for c in word.lower() if c in _LETTER_TO_ROW]
    transitions = [r for i, r in enumerate(rows) if i == 0 or r != rows[i - 1]]
    return max(1, len(transitions))


def _resample(seq: list[float], n: int = 60) -> list[float]:
    """Linearly resample *seq* to exactly *n* points.

    DTW handles variable-length sequences natively, but resampling to a
    common length first speeds up matching considerably and makes all
    templates comparable regardless of how fast the user swept.
    """
    if len(seq) == 0:
        return [0.0] * n
    if len(seq) == 1:
        return [seq[0]] * n
    xs = np.linspace(0, len(seq) - 1, n)
    return list(np.interp(xs, np.arange(len(seq)), seq))


# ---------------------------------------------------------------------------
# Template database
# ---------------------------------------------------------------------------

class SwipeTemplateDB:
    """Stores and matches swipe-typing trajectories using DTW."""

    def __init__(self) -> None:
        # word -> list of resampled trajectories
        self._templates: dict[str, list[list[float]]] = {}
        self._letter_to_row = _LETTER_TO_ROW

    # ------------------------------------------------------------------
    # Building the database (training time)
    # ------------------------------------------------------------------

    def add_template(self, word: str, trajectory: list[float]) -> None:
        """Record one swipe trajectory for *word*.

        *trajectory* is a sequence of normalised y-positions (0–1)
        sampled at roughly the webcam frame-rate during the swipe.
        It is resampled to a length determined by the word's row transitions.
        """
        word = word.lower().strip()
        if not word or len(trajectory) < 3:
            return
        resampled = _resample(trajectory, self._word_resample_len(word))
        self._templates.setdefault(word, []).append(resampled)

    def _word_resample_len(self, word: str) -> int:
        return max(MIN_RESAMPLE, _word_row_transitions(word) * BASE_POINTS)

    def words(self) -> list[str]:
        return list(self._templates.keys())

    def sample_count(self, word: str) -> int:
        return len(self._templates.get(word, []))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._templates, f)
        print(f"[SwipeDTW] Saved {len(self._templates)} word templates to {path}")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        with open(path) as f:
            self._templates = json.load(f)
        total = sum(len(v) for v in self._templates.values())
        print(f"[SwipeDTW] Loaded {len(self._templates)} words / {total} templates from {path}")
        self._validate_templates()

    def _validate_templates(self) -> bool:
        for word, samples in self._templates.items():
            if not samples:
                continue
            expected = self._word_resample_len(word)
            actual = len(samples[0])
            if actual != expected:
                print(
                    f"[SwipeDTW] WARNING: Template for '{word}' has {actual} pts, "
                    f"expected {expected}. Old fixed-length templates detected — "
                    f"please retrain with: eyetrax-train-swipe"
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def match(
        self,
        trajectory: list[float],
        top_k: int = 5,
        start_row: int | None = None,
        end_row: int | None = None,
        letter_to_row: dict[str, int] | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to *top_k* (word, dtw_distance) pairs ordered by DTW similarity.

        *trajectory* is a raw (un-resampled) sequence of normalised
        y-positions collected during the swipe.  Returns an empty list
        if the database is empty or the trajectory is too short.
        """
        if not self._templates or len(trajectory) < 3:
            return []

        scores: list[tuple[str, float]] = []
        for word, samples in self._templates.items():
            if letter_to_row is not None:
                if start_row is not None and letter_to_row.get(word[0].lower() if word else "") != start_row:
                    continue
                if end_row is not None and letter_to_row.get(word[-1].lower() if word else "") != end_row:
                    continue
            obs = _resample(trajectory, self._word_resample_len(word))
            dist = min(_dtw_distance(obs, s) for s in samples)
            scores.append((word, dist))

        scores.sort(key=lambda x: x[1])
        return scores[:top_k]
