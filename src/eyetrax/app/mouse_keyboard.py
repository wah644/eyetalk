"""
Mouse-click keyboard for quick testing without eye tracking.

Click a key row to build a T9 sequence → predictions appear on the left.
Click a prediction to accept the word.
Click a bigram suggestion (right panel) to accept the next word.
Right-click anywhere → backspace.

Launch:
    python src/eyetrax/app/mouse_keyboard.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WIN_W, WIN_H   = 1280, 800
KBD_W          = 280
KBD_GAP        = 8
PANEL_MARGIN   = 10
FONT           = cv2.FONT_HERSHEY_SIMPLEX
OUTPUT_FILE    = "mouse_keyboard_output.txt"

WORDFREQ_XLSX  = Path(__file__).parent / "wordFrequency.xlsx"
WORDFREQ_SHEET = "4 forms (219k)"
BIGRAM_FILE    = Path(__file__).parent / "Bigram.txt"

KEYBOARD_KEYS = [
    "ABCD",
    "EFGH",
    "IJKL",
    "MNOP",
    "QRSTU",
    "VWXYZ",
    "SPACE",
]

# Letter → key row index (rows 0-5, excluding SPACE row)
LETTER_TO_KEY: dict[str, int] = {}
for _idx, _row in enumerate(KEYBOARD_KEYS[:-1]):
    for _ch in _row.lower():
        LETTER_TO_KEY[_ch] = _idx


# ---------------------------------------------------------------------------
# T9 Trie
# ---------------------------------------------------------------------------

class _TrieNode:
    __slots__ = ("children", "words")
    def __init__(self):
        self.children: dict[int, _TrieNode] = {}
        self.words: list[tuple[int, str]] = []   # (rank, word) — multiple words per node


class T9Trie:
    def __init__(self):
        self.root = _TrieNode()

    def insert(self, word: str, rank: int = 999999) -> None:
        node = self.root
        for ch in word.lower():
            ki = LETTER_TO_KEY.get(ch)
            if ki is None:
                return
            if ki not in node.children:
                node.children[ki] = _TrieNode()
            node = node.children[ki]
        node.words.append((rank, word))

    def search(self, key_seq: list[int], max_results: int = 5) -> list[str]:
        if not key_seq:
            return []
        node = self.root
        for ki in key_seq:
            if ki not in node.children:
                return []
            node = node.children[ki]
        results: list[tuple[int, str]] = []
        self._collect(node, results)
        results.sort(key=lambda x: x[0])
        return [w for _, w in results[:max_results]]

    def _collect(self, node: _TrieNode, out: list[tuple[int, str]]) -> None:
        out.extend(node.words)
        for child in node.children.values():
            self._collect(child, out)


# ---------------------------------------------------------------------------
# Bigram model
# ---------------------------------------------------------------------------

class BigramModel:
    def __init__(self):
        self._data: dict[str, list[str]] = {}
        if not BIGRAM_FILE.exists():
            print(f"[MouseKbd] Bigram file not found: {BIGRAM_FILE}")
            return
        raw: dict[str, dict[str, int]] = {}
        with open(BIGRAM_FILE, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                w1, w2 = parts[0].lower(), parts[1].lower()
                try:
                    freq = int(parts[2])
                except ValueError:
                    continue
                raw.setdefault(w1, {})
                raw[w1][w2] = raw[w1].get(w2, 0) + freq
        for w1, freq_map in raw.items():
            self._data[w1] = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
        print(f"[MouseKbd] Loaded bigram data ({len(self._data)} words)")

    def predict(self, word: str, top: int = 5) -> list[str]:
        if not word:
            return []
        return [w for w, _ in self._data.get(word.lower().strip(), [])[:top]]


# ---------------------------------------------------------------------------
# Dictionary loader
# ---------------------------------------------------------------------------

_CONTRACTION_ARTIFACTS: set[str] = {
    # n't contractions with apostrophe stripped
    "dont", "doesnt", "didnt", "cant", "wont", "isnt", "wasnt", "arent",
    "werent", "wouldnt", "shouldnt", "couldnt", "havent", "hadnt", "neednt",
    "mustnt",
    # 're contractions
    "youre", "theyre", "theres", "whats", "whos", "hows",
    # 've contractions
    "ive", "youve", "weve", "theyve", "wouldve", "couldve", "shouldve",
    # 'll contractions
    "youll", "theyll", "itll",
    # 'd contractions
    "youd", "theyd",
    # 's contractions
    "hes", "shes",
    # misc tokenisation artifacts
    "th", "yearold", "km",
}


def _load_trie() -> T9Trie:
    trie = T9Trie()
    if not WORDFREQ_XLSX.exists():
        print(f"[MouseKbd] {WORDFREQ_XLSX.name} not found")
        return trie
    try:
        import openpyxl
    except ImportError:
        print("[MouseKbd] openpyxl not installed — run: pip install openpyxl")
        return trie

    wb    = openpyxl.load_workbook(str(WORDFREQ_XLSX), read_only=True, data_only=True)
    ws    = wb[WORDFREQ_SHEET]
    seen: set[str] = set()
    count = 0
    first = True
    for row in ws.iter_rows(values_only=True):
        if first:
            first = False
            continue
        rank_val = row[0]
        word     = row[1]
        if not isinstance(word, str):
            continue
        word = word.strip().lower()
        if not word.isalpha() or word in seen:
            continue
        seen.add(word)
        rank = int(rank_val) if isinstance(rank_val, (int, float)) else count + 1
        trie.insert(word, rank=rank)
        count += 1
    wb.close()
    print(f"[MouseKbd] Loaded {count} words from {WORDFREQ_XLSX.name}")
    return trie


def _load_trie_from_csv(csv_path: Path, top: int = 15000) -> T9Trie:
    import csv
    trie = T9Trie()
    seen: set[str] = set()
    count = 0
    rank = 0
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "word" not in (reader.fieldnames or []):
            print(f"[MouseKbd] CSV missing 'word' column: {csv_path}")
            return trie
        for row in reader:
            word = row["word"].strip().lower()
            if not word.isalpha() or word in seen or word in _CONTRACTION_ARTIFACTS:
                continue
            seen.add(word)
            rank += 1
            trie.insert(word, rank=rank)
            count += 1
            if count >= top:
                break
    print(f"[MouseKbd] Loaded {count} words from {csv_path.name} (top {top}, filtered)")
    return trie


# ---------------------------------------------------------------------------
# Keyboard state
# ---------------------------------------------------------------------------

class KbdState:
    def __init__(self, trie: T9Trie, bigram: BigramModel):
        self.trie         = trie
        self.bigram       = bigram
        self.key_seq:     list[int] = []
        self.predictions: list[str] = []
        self.ngram_preds: list[str] = []
        self.typed_text:  str = ""

    def press_key(self, idx: int) -> None:
        if idx == len(KEYBOARD_KEYS) - 1:   # SPACE row
            if self.predictions:
                self.accept(self.predictions[0])
        else:
            self.key_seq.append(idx)
            self.predictions = self.trie.search(self.key_seq, max_results=5)
            self._refresh_ngram()

    def accept(self, word: str) -> None:
        self.typed_text  += word + " "
        self.key_seq      = []
        self.predictions  = []
        self.ngram_preds  = self.bigram.predict(word)
        Path(OUTPUT_FILE).write_text(self.typed_text)

    def _refresh_ngram(self) -> None:
        if self.predictions:
            self.ngram_preds = self.bigram.predict(self.predictions[0])
        else:
            self.ngram_preds = []

    def backspace(self) -> None:
        if self.key_seq:
            self.key_seq.pop()
            self.predictions = self.trie.search(self.key_seq, max_results=5)
            self._refresh_ngram()
        else:
            stripped = self.typed_text.rstrip()
            sp = stripped.rfind(" ")
            self.typed_text  = stripped[: sp + 1] if sp >= 0 else ""
            self.ngram_preds = []


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _key_bounds(i: int, kbd_x: int, key_h: int) -> tuple[int, int, int, int]:
    y1 = i * (key_h + KBD_GAP)
    return kbd_x, y1, kbd_x + KBD_W, y1 + key_h


def _wrap(text: str, max_w: int, scale: float = 0.65, thick: int = 1) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip()
        if cv2.getTextSize(cand, FONT, scale, thick)[0][0] <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw(canvas: np.ndarray, state: KbdState,
          hover_key: int | None, kbd_x: int, key_h: int) -> None:
    canvas[:] = (45, 45, 45)
    kbd_right = kbd_x + KBD_W

    # ── Keyboard ────────────────────────────────────────────────────────────
    for i, label in enumerate(KEYBOARD_KEYS):
        x1, y1, x2, y2 = _key_bounds(i, kbd_x, key_h)
        hover  = (i == hover_key)
        bg     = (70, 70, 70) if hover else (40, 40, 40)
        color  = (0, 255, 0)  if hover else (200, 200, 200)
        thick  = 3            if hover else 1
        ov = canvas.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), bg, -1)
        cv2.addWeighted(ov, 0.7, canvas, 0.3, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thick)
        fs = 1.0 if len(label) <= 5 else 0.8
        tw, th = cv2.getTextSize(label, FONT, fs, 2)[0]
        cv2.putText(canvas, label,
                    (x1 + (KBD_W - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    FONT, fs, color, 2, cv2.LINE_AA)

    lp_x = PANEL_MARGIN
    lp_w = kbd_x - PANEL_MARGIN * 2
    rp_x = kbd_right + PANEL_MARGIN
    rp_w = WIN_W - rp_x - PANEL_MARGIN

    # ── Left: current key sequence + predictions ─────────────────────────
    if lp_w > 60:
        if state.key_seq:
            seq_label = "Keys: " + " · ".join(KEYBOARD_KEYS[k] for k in state.key_seq)
            cv2.putText(canvas, seq_label, (lp_x, 28),
                        FONT, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Predictions:", (lp_x, 58),
                    FONT, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
        for idx, word in enumerate(state.predictions):
            wy     = 95 + idx * 48
            by1, by2 = wy - 26, wy + 10
            cv2.rectangle(canvas, (lp_x, by1), (lp_x + lp_w, by2), (60, 60, 60), -1)
            cv2.rectangle(canvas, (lp_x, by1), (lp_x + lp_w, by2), (110, 110, 110), 1)
            cv2.putText(canvas, f"{idx + 1}.  {word}", (lp_x + 10, wy),
                        FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Right: bigram next-word ──────────────────────────────────────────
    if rp_w > 60 and state.ngram_preds:
        cv2.putText(canvas, "Next Word:", (rp_x, 58),
                    FONT, 0.75, (200, 100, 255), 2, cv2.LINE_AA)
        for idx, word in enumerate(state.ngram_preds):
            wy     = 95 + idx * 48
            by1, by2 = wy - 26, wy + 10
            cv2.rectangle(canvas, (rp_x, by1), (rp_x + rp_w, by2), (50, 30, 60), -1)
            cv2.rectangle(canvas, (rp_x, by1), (rp_x + rp_w, by2), (100, 60, 120), 1)
            cv2.putText(canvas, f"{idx + 1}.  {word}", (rp_x + 10, wy),
                        FONT, 0.8, (255, 200, 255), 2, cv2.LINE_AA)

    # ── Sentence bar (right of keyboard, lower half) ─────────────────────
    sb_top = int(WIN_H * 0.55)
    sb_bot = WIN_H - PANEL_MARGIN
    if rp_w > 60:
        cv2.rectangle(canvas, (rp_x, sb_top), (rp_x + rp_w, sb_bot), (0, 0, 0), -1)
        cv2.rectangle(canvas, (rp_x, sb_top), (rp_x + rp_w, sb_bot), (100, 100, 100), 2)
        display = state.typed_text
        if state.predictions:
            display += "[" + state.predictions[0] + "]"
        lines = _wrap(display.strip(), rp_w - 20)
        y = sb_bot - 12
        for line in reversed(lines):
            if y - 28 < sb_top + 4:
                break
            cv2.putText(canvas, line, (rp_x + 10, y),
                        FONT, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
            y -= 28

    # ── Hint bar ─────────────────────────────────────────────────────────
    cv2.putText(canvas,
                "Left-click: select key / prediction    Right-click: backspace    Q: quit",
                (kbd_x, WIN_H - 6), FONT, 0.45, (130, 130, 130), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Hit testing
# ---------------------------------------------------------------------------

def _hit_key(mx, my, kbd_x, key_h) -> int | None:
    for i in range(len(KEYBOARD_KEYS)):
        x1, y1, x2, y2 = _key_bounds(i, kbd_x, key_h)
        if x1 <= mx <= x2 and y1 <= my <= y2:
            return i
    return None


def _hit_list(mx, my, px, pw, items, base_y=95, row_h=48) -> int | None:
    for i in range(len(items)):
        wy = base_y + i * row_h
        if px <= mx <= px + pw and wy - 26 <= my <= wy + 10:
            return i
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Mouse keyboard — dictionary test tool")
    ap.add_argument("--dict", metavar="PATH", default=None,
                    help="CSV dictionary file (word,freq). Defaults to wordFrequency.xlsx.")
    ap.add_argument("--top", type=int, default=15000,
                    help="Max words to load from CSV (default 15000).")
    args = ap.parse_args()

    if args.dict:
        trie = _load_trie_from_csv(Path(args.dict), top=args.top)
    else:
        trie = _load_trie()
    bigram = BigramModel()
    state  = KbdState(trie, bigram)

    nk     = len(KEYBOARD_KEYS)
    key_h  = (WIN_H - (nk - 1) * KBD_GAP) // nk
    kbd_x  = (WIN_W - KBD_W) // 2
    lp_x   = PANEL_MARGIN
    lp_w   = kbd_x - PANEL_MARGIN * 2
    rp_x   = kbd_x + KBD_W + PANEL_MARGIN
    rp_w   = WIN_W - rp_x - PANEL_MARGIN

    canvas: np.ndarray = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    hover_key: int | None = None

    def on_mouse(event, mx, my, _flags, _param):
        nonlocal hover_key
        hover_key = _hit_key(mx, my, kbd_x, key_h)

        if event == cv2.EVENT_LBUTTONDOWN:
            pi = _hit_list(mx, my, lp_x, lp_w, state.predictions)
            if pi is not None:
                state.accept(state.predictions[pi])
                return
            ni = _hit_list(mx, my, rp_x, rp_w, state.ngram_preds)
            if ni is not None:
                state.accept(state.ngram_preds[ni])
                return
            ki = _hit_key(mx, my, kbd_x, key_h)
            if ki is not None:
                state.press_key(ki)

        elif event == cv2.EVENT_RBUTTONDOWN:
            state.backspace()

    cv2.namedWindow("Mouse Keyboard", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mouse Keyboard", WIN_W, WIN_H)
    cv2.setMouseCallback("Mouse Keyboard", on_mouse)

    print("[MouseKbd] Running — Q or ESC to quit")
    while True:
        _draw(canvas, state, hover_key, kbd_x, key_h)
        cv2.imshow("Mouse Keyboard", canvas)
        k = cv2.waitKey(16) & 0xFF
        if k in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    print(f"[MouseKbd] Final: {state.typed_text}")
    print(f"[MouseKbd] Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
