# EyeTrax — UI Reference & Timing Specification

---

## Part 1 — All Timings, Distances, and Thresholds

### Calibration

| Phase | Duration | Dot colour | Notes |
|---|---|---|---|
| Face detection / countdown | 2 s | — | Waits for stable face, then shows countdown arc |
| Pulse (attention phase) | 1 s | Green | No data collected — user aligns gaze |
| Capture (still-head) | 1 s | Red | Gaze features recorded |
| Multi-pose rotation (`--multi-pose`) | 4 s | Orange | User slowly rotates head while keeping eyes on dot |

**Per-dot total:** 2 s (still only) · 6 s (with multi-pose)

**Point counts by calibration mode:**

| Mode | Points | Description |
|---|---|---|
| `vertical-only` | 11 | 9 centre column + 2 off-screen anchors (above + below screen) |
| `vertical_single` | 14 | 9 centre column + 5 corners |
| `vertical` (enhanced) | 23 | 5 corners + 9-point left line + 9-point right line |
| `5p` | 5 | 4 corners + centre |
| `9p` | 9 | 3×3 grid |

---

### Blink Detection

| Parameter | Value | Notes |
|---|---|---|
| EAR threshold ratio | **0.75** | Blink fires when EAR drops below `baseline × 0.75`. Settled on after testing — high enough to catch real blinks without false positives. |

---

### Off-screen Panel Triggers (main keyboard screen)

| Gesture | Pixel threshold | Dwell required | Action |
|---|---|---|---|
| Look above screen | `y < −40 px` | 2 s | Dwell mode: opens trie prediction panel · EyeSwipe mode: opens word corrections panel |
| Look below screen | `y > screen_height + 40 px` | 2 s | Opens bigram next-word panel |

Both panels always open regardless of whether predictions are available. If no suggestions exist, the panel shows a "No suggestions" message.

---

### Trie / Bigram / Corrections Panel — Scroll Zones

The panel is divided into **3 equal vertical sections**:

| Section | Screen region | Behaviour |
|---|---|---|
| Top third | `y < screen_height / 3` | Scroll up (trie/corrections → worse words · bigram → better words) |
| Middle third | `screen_height / 3 ≤ y ≤ 2 × screen_height / 3` | Dwell zone — select centred word |
| Bottom third | `y > 2 × screen_height / 3` | Scroll down (trie/corrections → better words · bigram → worse words) |

Scroll boundary from screen centre = `screen_height / 6` (~167 px on a 1000 px screen).

**Scroll speed formula:**
```
effective  = |gaze_y − screen_centre| − scroll_threshold
speed      = 2.8 × (effective / half_remaining_height)   [word-positions / second]
```
- At the scroll boundary: speed = 0
- At the screen edge: speed = **2.8 word-positions / second** (maximum)
- Speed ramps linearly — no snapping

---

### Trie / Bigram / Corrections Panel — Word Selection & Exit

| Action | Condition | Time | Visual |
|---|---|---|---|
| Select centred word | Gaze in middle third | **1.5 s** dwell | Green arc around word |
| Trie — backspace | At scroll top + looking up | **2 s** | Blue bar grows at top edge + "BACKSPACE" label |
| Trie — back to keyboard | At scroll bottom + looking down | **2 s** | Cyan bar grows at bottom edge |
| Bigram — back to keyboard | At scroll top + looking up | **2 s** | Cyan bar grows at top edge |
| Bigram — home menu | At scroll bottom + looking down | **5 s** | Cyan bar grows at bottom edge + "HOME" label |
| Corrections — dismiss | At scroll top + looking up **or** at scroll bottom + looking down | **2 s** | Cyan bar grows at active edge |

All exit/backspace timers reset immediately if gaze leaves the triggering zone.
Timers do not start while gaze is off-screen (prevents opening gesture pre-arming them).

---

### Dwell Keyboard — Key Selection

After testing, the following values were settled on:

| Action | Time |
|---|---|
| Key selection (dwell mode) | **0.85 s** |
| Swipe — dwell on start key to arm | **1.0 s** |
| Swipe — dwell on end key to commit | **1.0 s** |
| Swipe auto-cancel (no end key found) | 8 s |

**Backspace behaviour (dwell mode):**
- Active key sequence → removes last key from sequence (narrows trie predictions)
- No active key sequence → deletes the entire last word from typed text

---

### Long Blink (keyboard mode only)

| Blink duration | Mid-blink feedback | Action on eye-open |
|---|---|---|
| ≥ 1.0 s | Beep at 1 s mark | Complete sentence → TTS → clear text |
| ≥ 2.0 s | Beep at 1 s + 2 s marks | Send keywords to AI → TTS result → clear text |

Audio beeps fire mid-blink as tactile feedback only — no action until eyes open.

On sentence completion (either path), all consecutive word pairs are saved to personal bigram history, and each word's frequency count is incremented in the appropriate usage file.

---

### Menu Dwell Times

| Screen | Dwell to select |
|---|---|
| Main menu | **1.2 s** |
| Input mode selection | **1.2 s** |
| IoT / lights menu | **1.2 s** |
| Mobile phone menu | **1.2 s** |
| Emergency confirm | **1.2 s** |
| Back gesture — look above screen (any non-keyboard screen) | **2.0 s** |

---

---

## Part 2 — UI Architecture & Screen-by-Screen Explanation

### Overview

EyeTrax is a gaze-controlled communication and device-control interface designed for users with Locked-in Syndrome. The entire system is operated by eye movement alone — no physical input is required. The interface is rendered full-screen using OpenCV and is structured as a finite-state machine with the following top-level modes:

```
menu → keyboard
     → fixed_phrases
     → selection_mode
     → lights_menu
     → mobile_phone → emergency_confirm
```

Navigation between screens is done through **dwell selection** (fixating on a button for a set time) and **off-screen gestures** (looking beyond the screen edges).

---

### 2.1 Main Menu

The main menu is the home screen. It presents a grid of large buttons, each representing a mode:

- **Keyboard** — opens the eye-typing keyboard
- **Fixed Phrases** — opens a scrollable list of pre-written phrases spoken via TTS
- **Home IoT** — controls smart lights via ESP32
- **Mobile Phone** — interface to control an Android phone via ADB/scrcpy
- **Selection Mode** — switch between dwell typing and EyeSwipe (swipe typing)

**Interaction:** The user looks at a button. After **1.2 s** of continuous dwell, the button activates. A dwell progress arc or highlight grows to give visual feedback. Looking away resets the timer.

**Return:** From any non-keyboard screen, looking above the screen (`y < −40 px`) for **2 s** returns to the main menu. A cyan progress bar grows along the top edge of the screen as the timer counts.

---

### 2.2 Keyboard Screen

The keyboard is the primary communication tool. It supports two input modes selectable from the main menu.

No persistent labels or instructions are shown on the main keyboard screen during normal use. A small red dot in the top-right corner pulses when an EyeSwipe is being recorded.

#### 2.2.1 Dwell Mode

A vertically stacked set of key rows. Each row contains letters mapped to a T9-style number key. The user dwells on a row for **0.85 s** to register that key in the current word sequence.

As keys are pressed, the T9 trie searches for matching words and displays up to 5 predictions on the keyboard screen:
- **Trie predictions** — left panel, best match closest to centre, worse matches above
- **Bigram predictions** — right panel, best next-word closest to centre, worse below

The spacebar accepts the top trie prediction immediately. If no word is being typed (key sequence empty), spacebar accepts the top bigram suggestion instead. If neither is available, it inserts a literal space.

DELETE removes the last key from the current sequence. If no sequence is active, it deletes the **entire last word** from the typed text.

**Off-screen gestures from keyboard (dwell mode):**
- Look above screen → trie prediction panel
- Look below screen → bigram next-word panel

Both panels open even when no suggestions are available.

#### 2.2.2 EyeSwipe Mode

Instead of tapping individual keys, the user swipes gaze vertically across key rows to trace a word shape.

1. **Arm** — dwell on the row containing the first letter of the word for **1.0 s**. A red recording dot appears in the top-right corner.
2. **Swipe** — move gaze vertically across the rows corresponding to the word's letters. Gaze Y positions are recorded as a trajectory.
3. **Commit** — dwell on the row containing the last letter for **1.0 s**. The trajectory is matched against stored word templates using DTW (Dynamic Time Warping).
4. **Auto-accept or Corrections panel:**
   - If the top DTW match is **clearly dominant** (no other word scores within 15% of the best combined score) → the word is auto-accepted immediately.
   - If **multiple words score closely** → the corrections panel opens automatically without auto-accepting. The user selects the intended word.
5. **Manual corrections** — from the keyboard, looking above screen for **2 s** opens the corrections panel showing all DTW candidates for the last swipe.

**Off-screen gestures from keyboard (EyeSwipe mode):**
- Look above screen → word corrections panel
- Look below screen → bigram next-word panel

**Swipe scoring formula:**
```
combined_score = dtw_distance / (usage_count + 1) ^ 0.5
```
- Words with 0 uses → pure DTW order (divisor = 1)
- As usage builds, frequently selected words require a proportionally worse DTW match before being displaced
- The 0.5 exponent gives diminishing returns — each additional use matters less than the first

---

### 2.3 Trie Prediction Panel

Opened by looking above the screen (`y < −40 px`) for **2 s** from the keyboard (dwell mode only).

The panel fills the screen with up to 5 word predictions displayed vertically. Words are ordered so that the **best prediction is at the bottom**, with worse predictions stacked above. On open, the scroll position is initialised so the best word sits at screen centre. If no predictions exist, a "No suggestions" message is displayed — the panel still opens and can be exited normally.

**Layout:**
- Top third → scroll zone (look here to scroll up through worse predictions)
- Middle third → selection zone (dwell 1.5 s on the centred word to accept it)
- Bottom third → scroll zone (look here to scroll down toward better predictions)

**Exiting:**
- Scroll to the bottom (best word centred) + keep looking down → **2 s** → returns to keyboard without selecting
- Scroll to the top (worst word centred) + keep looking up → **2 s** → **backspace** (deletes last key from sequence or last word if no sequence) then returns to keyboard. A blue progress bar and "BACKSPACE" label appear at the top edge.

**Word ranking:** Predictions from the T9 trie (top 5 by dictionary frequency from a 35,000-word list optimised for LIS communication), then reranked by personal dwell usage history.

**Bigram context:** After selecting a word from the trie, the bigram panel immediately reflects predictions for the newly accepted word.

---

### 2.4 Word Corrections Panel (EyeSwipe only)

Opened by:
- Looking above the screen for **2 s** from the keyboard (EyeSwipe mode), or
- Automatically, when multiple DTW candidates score within 15% of each other

Shows up to 5 DTW candidate words labelled **"Word Corrections"**. Words are ordered best-at-bottom (same layout as trie panel) using the combined DTW + frequency score.

**Layout:** Identical to trie panel — best word at bottom, scroll up for alternatives.

**Selecting:** Dwell 1.5 s on the centred word.
- If panel opened automatically (no prior auto-accept) → selected word is accepted fresh
- If panel opened after auto-accept (manual correction) → selected word replaces the previously auto-accepted word

**Exiting without selecting:**
- Look down past the bottom **or** look up past the top → **2 s** → dismisses panel, returns to keyboard (keeping the auto-accepted word if one was already committed, otherwise nothing is accepted)

---

### 2.5 Bigram Next-Word Panel

Opened by looking below the screen (`y > screen_height + 40 px`) for **2 s** from the keyboard (both modes).

Shows up to 5 next-word predictions. The **best prediction is at the top**, scroll is initialised so the best word is at centre. Opens even when no suggestions are available.

**Bigram source:** Always based on the last fully accepted complete word — not the partial word currently being typed. This keeps predictions stable and contextually correct regardless of how many keys have been pressed.

**Layout:**
- Top third → scroll zone (look up to scroll toward better predictions)
- Middle third → selection zone (dwell 1.5 s to accept)
- Bottom third → scroll zone (look down to scroll through worse predictions)

**Exiting:**
- At top of scroll (best word) + keep looking up → **2 s** → returns to keyboard without selecting
- At bottom of scroll (worst word) + keep looking down → **5 s** → returns to **main menu**. A cyan progress bar and "HOME" label appear at the bottom edge.

**Word ranking:** Personal bigram history (words previously typed after the current word, sorted by count) fills first; remaining slots come from the static bigram model. Panel always accessible — falls back to predictions based on the last accepted word even between sentences.

---

### 2.6 Long Blink — Sentence Completion

While on the keyboard screen, a long blink (eyes closed beyond 1 s) triggers sentence actions:

- **1 s blink** — the current typed sentence is spoken aloud via TTS, the text box clears, and a new sentence begins.
- **2 s blink** — the typed keywords are sent to a local LLM (Ollama), which expands them into a full sentence. The result is spoken and replaces the text. An "ACTIVATED" flash confirms submission.

Audio beeps fire mid-blink at the 1 s and 2 s marks as tactile feedback. No action fires until eyes open.

On sentence completion (either path):
- All consecutive word pairs in the sentence are saved to personal bigram history (`user_usage.json`)
- Each word's frequency count is incremented in the appropriate mode's usage file

---

### 2.7 Typed Text Display

The typed text appears in a panel at the **top-right** of the screen (10 px margin from top, occupying the top 25% of screen height on the right side). Text wraps automatically. The most recent word is always visible.

---

### 2.8 Fixed Phrases Panel

A scrollable list of pre-written phrases. The user scrolls by looking at the top or bottom thirds of the screen and dwells at centre to select a phrase. On selection, the phrase is spoken immediately via TTS and the screen returns to the main menu.

Scroll speed follows the same linear ramp model as the word panels (max 2.8 positions/second).

---

### 2.9 Home IoT (Lights Menu)

Two large buttons: **Lights ON** and **Lights OFF**. Each requires **1.2 s** dwell to activate. On activation, an HTTP GET request is sent to the ESP32 controller in a background thread so the UI is not blocked. Returns to main menu after selection.

---

### 2.10 Mobile Phone Menu

Two options presented as large buttons (1.2 s dwell each):

- **Show Phone UI** — launches scrcpy to mirror the Android phone screen
- **Emergency Call** — opens the emergency confirmation screen

While scrcpy is active, the phone is controlled directly through the mirrored display.

---

### 2.11 Emergency Confirm Screen

A two-button confirmation (Yes / No) requiring **1.2 s** dwell. If confirmed:
1. An outgoing call is placed via ADB to the configured emergency number
2. The system waits for the call to be answered (polling telecom state, with a 6-second minimum wait to skip the ringing phase)
3. Once answered, the speakerphone is automatically enabled by tapping the speaker button in the Android in-call UI
4. A pre-written emergency message is spoken via TTS on the phone

Returns to keyboard on either outcome.

---

### 2.12 Input Mode Selection

A two-option screen (EyeSwipe / Dwell) reached from the main menu. Selecting an option (1.2 s dwell) switches the keyboard mode immediately and returns to the main menu.

---

### 2.13 Gaze Cursor & Scan Path

An optional real-time gaze dot can be overlaid on screen (`--cursor` flag). A scan path trail shows recent gaze history as a fading line, useful for calibration verification and session review. The trail length is configurable (`--scan-path-max`, default 500 points) and can be saved to CSV on exit.

---

### 2.14 Personal Learning System

EyeTrax maintains two persistent usage files that improve predictions over time:

**`user_usage.json`** — used by the dwell keyboard:
- **Word frequency** — every word accepted via dwell increments its count. Trie predictions are reranked so frequently chosen words surface first.
- **Bigram history** — on sentence completion (either blink path), all consecutive word pairs are extracted and stored. The bigram panel shows personal bigrams first, falling back to the static model for remaining slots. Bigrams are shared across both input modes.

**`user_usage_swipe.json`** — used by EyeSwipe only:
- **Swipe word frequency** — every word accepted or corrected via swipe increments its count in this file. These counts feed the combined DTW + frequency scoring formula. Kept separate from dwell so that typing "back" ten times via dwell does not artificially inflate its swipe ranking.

**Dictionary:** The trie is built from a 35,000-word English frequency list, pre-patched so that approximately 270 words commonly needed by LIS patients (pain, hurt, hurts, nurse, water, bed, etc.) are guaranteed to appear in the top-5 T9 predictions for their key sequences.

Both usage files can be cleared at any time to reset personalisation.
