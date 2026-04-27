"""
Patch english_words.csv so that 250 LIS-relevant words all appear
in the top-5 T9 trie predictions for their key sequences.

If a word is missing from the top 5, it is inserted just BEFORE
the 2nd-ranked result in the CSV (giving it rank between #1 and #2
for that key sequence).  If it is already in the CSV but ranked
too low, it is moved to that position.

Run from the project root:
    python src/eyetrax/app/patch_lis_words.py
"""

import csv
import sys
from pathlib import Path

# ── Key layout (must match keyboard_demo.py) ──────────────────────────────
KEYBOARD_ROWS = ["ABCD", "EFGH", "IJKL", "MNOP", "QRSTU", "VWXYZ"]
LETTER_TO_KEY: dict[str, int] = {}
for _idx, _row in enumerate(KEYBOARD_ROWS):
    for _ch in _row.lower():
        LETTER_TO_KEY[_ch] = _idx

DICTIONARY_LIMIT = 35_000

# ── 250 LIS patient words ─────────────────────────────────────────────────
LIS_WORDS = [
    # essentials / yes-no
    "yes", "no", "okay", "please", "thank", "thanks", "sorry", "help",

    # pain & physical
    "pain", "hurt", "hurts", "ache", "sore", "numb", "burning", "pressure",
    "chest", "back", "neck", "head", "leg", "arm", "hand", "foot", "knee",
    "stomach", "throat", "mouth", "eye", "nose", "ear", "shoulder", "hip",
    "headache", "dizzy", "nausea", "fever", "cough", "itchy", "cramp",

    # vital needs
    "water", "drink", "food", "eat", "hungry", "thirsty", "medicine",
    "toilet", "bathroom", "bedpan", "wash", "wipe", "clean",

    # comfort / position
    "cold", "hot", "warm", "blanket", "pillow", "bed", "position", "turn",
    "sit", "lie", "lift", "move", "adjust", "uncomfortable", "comfortable",

    # breathing / medical
    "breathe", "breathing", "oxygen", "suction", "nurse", "doctor",
    "hospital", "emergency", "blood", "pressure", "heart", "pulse",
    "tired", "weak", "sick", "better", "worse", "fever",

    # emotional
    "happy", "sad", "scared", "afraid", "anxious", "worried", "calm",
    "frustrated", "confused", "angry", "lonely", "bored", "love",
    "miss", "afraid", "fine", "good", "bad",

    # communication
    "listen", "understand", "repeat", "louder", "slower", "stop", "wait",
    "more", "again", "enough", "done", "finished", "ready", "yes", "no",

    # verbs / state
    "want", "need", "feel", "think", "know", "see", "hear", "can",
    "cannot", "will", "would", "could", "should", "am", "is", "are",
    "was", "were", "have", "has", "had", "do", "did", "not", "come",
    "go", "stay", "call", "open", "close", "change", "try",

    # people
    "mom", "dad", "wife", "husband", "son", "daughter", "sister",
    "brother", "family", "friend", "children", "baby", "nurse",

    # time
    "now", "soon", "later", "today", "tomorrow", "yesterday", "morning",
    "afternoon", "evening", "night", "always", "never", "sometimes",
    "when", "before", "after",

    # place / direction
    "here", "there", "home", "room", "outside", "inside", "window",
    "door", "light", "fan",

    # common function words
    "my", "your", "his", "her", "their", "our", "this", "that", "with",
    "for", "from", "about", "where", "why", "what", "how", "because",
    "but", "and", "or", "if", "then", "also", "just", "very", "really",
    "much", "little", "any", "some", "all", "too", "so", "not",
    "the", "a", "an", "in", "on", "at", "of", "to", "it", "its",
    "me", "him", "us", "them", "who", "which", "every",

    # device / environment
    "phone", "television", "remote", "music", "read", "watch", "visit",
    "talk", "text", "message",

    # sentences starters
    "please", "can", "could", "would", "i", "we",

    # extra medical
    "allergy", "injection", "scan", "test", "result", "report",
    "catheter", "tube", "drip", "monitor", "alarm",

    # quality of life
    "sleep", "rest", "quiet", "noise", "bright", "dark", "fresh",
    "smell", "taste", "voice", "speak", "write",
]

# Deduplicate while preserving order; ensure "hurt"/"hurts" are present
seen_set: set[str] = set()
WORDS: list[str] = []
priority = ["hurt", "hurts"]
for w in priority + LIS_WORDS:
    w = w.lower().strip()
    if w and w.isalpha() and w not in seen_set:
        seen_set.add(w)
        WORDS.append(w)

print(f"Target words: {len(WORDS)}")
assert "hurt"  in WORDS, "hurt missing"
assert "hurts" in WORDS, "hurts missing"


# ── Helpers ───────────────────────────────────────────────────────────────

def word_to_key_seq(word: str) -> list[int] | None:
    seq = []
    for ch in word.lower():
        k = LETTER_TO_KEY.get(ch)
        if k is None:
            return None
        seq.append(k)
    return seq


class TrieNode:
    __slots__ = ("children", "words")
    def __init__(self):
        self.children: dict[int, "TrieNode"] = {}
        self.words: list[tuple[int, str]] = []  # (rank, word)


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, rank: int):
        node = self.root
        for ch in word.lower():
            k = LETTER_TO_KEY.get(ch)
            if k is None:
                return
            node = node.children.setdefault(k, TrieNode())
        node.words.append((rank, word))

    def search(self, key_seq: list[int], top_k: int = 5) -> list[tuple[int, str]]:
        node = self.root
        for k in key_seq:
            if k not in node.children:
                return []
            node = node.children[k]
        results: list[tuple[int, str]] = []
        self._collect(node, results)
        results.sort(key=lambda x: x[0])
        return results[:top_k]

    def _collect(self, node: TrieNode, out: list):
        out.extend(node.words)
        for child in node.children.values():
            self._collect(child, out)


# ── Load CSV ──────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "english_words.csv"

print(f"Loading {CSV_PATH} …")
rows: list[list[str]] = []   # each inner list: [word, freq]
header: list[str] = []

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for r in reader:
        rows.append(r)

print(f"  CSV rows: {len(rows)}")

# Build word → row-index map (0-based after header)
word_to_idx: dict[str, int] = {}
for i, r in enumerate(rows):
    if r:
        word_to_idx[r[0].strip().lower()] = i

# ── Build trie from first DICTIONARY_LIMIT alphabetic words ───────────────
trie = Trie()
rank = 0
for i, r in enumerate(rows):
    if rank >= DICTIONARY_LIMIT:
        break
    if not r:
        continue
    w = r[0].strip().lower()
    if not w or not w.isalpha():
        continue
    rank += 1
    trie.insert(w, rank)

print(f"  Trie built with {rank} words")

# ── Check each target word and collect insertions needed ──────────────────
# We collect (insert_before_row_idx, new_word, new_freq) then apply all at once.

insertions: list[tuple[int, str, str]] = []   # (before_idx, word, freq_str)
already_ok: list[str] = []
missing_key: list[str] = []

for target in WORDS:
    key_seq = word_to_key_seq(target)
    if key_seq is None:
        missing_key.append(target)
        continue

    top5 = trie.search(key_seq, top_k=5)
    top5_words = [w for _, w in top5]

    if target in top5_words:
        already_ok.append(target)
        continue

    # Word not in top 5 — insert just before the 2nd result in the CSV.
    # If there are fewer than 2 results, insert before the 1st result,
    # or at the very top of the loaded slice if none at all.
    if len(top5) >= 2:
        second_word = top5[1][1]
        insert_before = word_to_idx.get(second_word)
    elif len(top5) == 1:
        insert_before = word_to_idx.get(top5[0][1])
    else:
        insert_before = None   # no conflict — just prepend

    # Choose a frequency slightly above the 2nd result so it sorts correctly
    # in any future frequency-ordered tool; for the trie only CSV row-order matters.
    if insert_before is not None and insert_before < len(rows):
        ref_freq = rows[insert_before][1] if len(rows[insert_before]) > 1 else "1"
        try:
            freq_val = int(ref_freq) + 1
        except ValueError:
            freq_val = 1
    else:
        freq_val = 1

    action = "insert" if target not in word_to_idx else "move"
    print(f"  [{action}] '{target}'  seq={key_seq}  top5={top5_words}  before='{rows[insert_before][0] if insert_before is not None else 'TOP'}'")
    insertions.append((insert_before, target, str(freq_val)))

print(f"\nSummary: {len(already_ok)} already in top-5, {len(insertions)} need patching, {len(missing_key)} unmappable")

if not insertions:
    print("Nothing to do — CSV is already correct.")
    sys.exit(0)

# ── Apply insertions ──────────────────────────────────────────────────────
# Sort by insert position descending so row indices stay valid as we insert.
# Words that need to move: remove from old position first.

# Step 1: remove words that already exist in the CSV (they'll be re-inserted)
words_to_move = {w for _, w, _ in insertions if w in word_to_idx}
new_rows: list[list[str]] = []
removed_positions: dict[str, int] = {}   # word → original idx (for reference)
shift = 0
for i, r in enumerate(rows):
    w = r[0].strip().lower() if r else ""
    if w in words_to_move:
        removed_positions[w] = i
        shift += 1
        # recalculate word_to_idx for remaining rows (we'll rebuild below)
    else:
        new_rows.append(r)

# Rebuild word_to_idx after removals
word_to_idx2: dict[str, int] = {r[0].strip().lower(): i for i, r in enumerate(new_rows) if r}

# Step 2: insert each word at its target position (sort ascending so indices stay valid)
insertions_sorted = sorted(insertions, key=lambda x: x[0] if x[0] is not None else 0)

extra_shift = 0
for (before_idx, word, freq_str) in insertions_sorted:
    if before_idx is None:
        target_idx = 0
    else:
        # Adjust for earlier insertions and the removals we did
        # Find the current position of the anchor word
        anchor_word = None
        if before_idx < len(rows):
            anchor_word = rows[before_idx][0].strip().lower()
        if anchor_word and anchor_word in word_to_idx2:
            target_idx = word_to_idx2[anchor_word]
        else:
            target_idx = min(before_idx, len(new_rows))

    new_rows.insert(target_idx, [word, freq_str])
    # Rebuild word_to_idx2 after each insertion
    word_to_idx2 = {r[0].strip().lower(): i for i, r in enumerate(new_rows) if r}

# ── Write patched CSV ─────────────────────────────────────────────────────
print(f"\nWriting patched CSV ({len(new_rows)} data rows) …")
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(new_rows)

print("Done. Run the keyboard to verify predictions.")
