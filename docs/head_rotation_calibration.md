# Head-Rotation Calibration (`--multi-position`)

## Overview

The `--multi-position` flag previously ran three full calibration rounds, each at a different vertical head height (above / center / below the camera). This required the user to physically reposition themselves between rounds, making the process slow and cumbersome.

It has been replaced with a single-round calibration that collects head-pose-diverse data at every individual calibration point. Instead of repositioning between rounds, the user keeps their eyes fixed on each dot while slowly rotating their head in a full circle. This achieves the same goal — exposing the model to a wide range of head poses — in a single pass.

---

## What Changed

### `src/eyetrax/calibration/multi_position.py` — full rewrite

**Before:** Orchestrated 3 separate calibration rounds. For each round:
1. Showed a splash screen ("Round 1/3 — Face ABOVE")
2. Displayed a live camera feed with a colored zone band and waited for the user's face to lock into the target vertical zone for 2 seconds
3. Ran the chosen calibration method with `train=False` to collect raw `(X, y)` arrays
4. After all 3 rounds, pooled the arrays and called `gaze_estimator.train()` once on the combined dataset

**After:** A thin wrapper that calls the chosen calibration method once with `multi_pose=True` and `multi_pose_d=HEAD_MOVE_DURATION` (4 seconds). No zone detection, no splash screens, no data pooling — training happens inside the calibration method as normal.

Key constant:
```python
HEAD_MOVE_DURATION = 4.0  # seconds of head movement capture per calibration point
```

Supported calibration methods and how they are invoked:

| `--calibration` | Called with |
|---|---|
| `9p` | `run_9_point_calibration(..., multi_pose=True, multi_pose_d=4.0)` |
| `5p` | `run_5_point_calibration(..., multi_pose=True, multi_pose_d=4.0)` |
| `vertical` | `run_vertical_enhanced_calibration(..., multi_pose=True, multi_pose_d=4.0)` |
| `lissajous` | `run_lissajous_calibration(...)` (no head-movement support, runs normally) |

---

### `src/eyetrax/calibration/common.py` — updated `_pulse_and_capture`

The `multi_pose` capture phase (the extra phase appended after the still-head capture at each dot) was updated with:

**Instruction text**
- Before: `"Move your head gently"`
- After: `"Keep eyes on dot  |  Slowly rotate head in a full circle"`

**Visual guide** — three new elements drawn on the canvas during this phase:
1. **Guide ring** — a faint circle of radius 70px centred on the calibration dot, showing the circular path for the head to trace
2. **Orbiting indicator** — a cyan dot that travels around the guide ring, completing exactly one full revolution over the `multi_pose_d` duration. Gives the user a visual target to follow with their head movement
3. **Progress arc** — unchanged from before; an arc shrinking around the dot showing time remaining

```
orbit_angle = 2π × (elapsed / multi_pose_d) − π/2
```

---

### `src/eyetrax/calibration/nine_point.py` — new `multi_pose_d` parameter

```python
# Before
def run_9_point_calibration(gaze_estimator, camera_index=0,
                            multi_pose=False, train=True):

# After
def run_9_point_calibration(gaze_estimator, camera_index=0,
                            multi_pose=False, multi_pose_d=1.0, train=True):
```

`multi_pose_d` is forwarded to `_pulse_and_capture`. Default stays `1.0` so the standalone `--multi-pose` flag behaviour is unchanged.

---

### `src/eyetrax/calibration/five_point.py` — new `multi_pose_d` parameter

Same change as `nine_point.py`.

```python
# Before
def run_5_point_calibration(gaze_estimator, camera_index=0,
                            multi_pose=False, train=True):

# After
def run_5_point_calibration(gaze_estimator, camera_index=0,
                            multi_pose=False, multi_pose_d=1.0, train=True):
```

---

### `src/eyetrax/calibration/vertical_enhanced_calibration.py` — new `multi_pose` and `multi_pose_d` parameters

This file previously had no `multi_pose` support at all. It now accepts both parameters and passes them to `_pulse_and_capture`:

```python
# Before
def run_vertical_enhanced_calibration(gaze_estimator, camera_index=0, train=True):
    ...
    res = _pulse_and_capture(gaze_estimator, cap, all_points, sw, sh)

# After
def run_vertical_enhanced_calibration(gaze_estimator, camera_index=0,
                                      multi_pose=False, multi_pose_d=1.0, train=True):
    ...
    res = _pulse_and_capture(gaze_estimator, cap, all_points, sw, sh,
                             multi_pose=multi_pose, multi_pose_d=multi_pose_d)
```

---

## How the Pipeline Works

### Feature extraction per frame (`gaze.py — extract_features`)

At every captured frame, the following features are stacked into a single vector `X`:

| Feature group | Size | Description |
|---|---|---|
| Eye region landmarks | ~100 × 3 | MediaPipe points around both eyes, nose-relative, rotation-normalized, inter-eye-distance-scaled |
| Head pose | 3 | Yaw, pitch, roll (radians) derived from the rotation matrix |
| Face position *(optional)* | 26 | 13 raw image-space landmarks × (x, y), only if `--multi-position` (i.e. `include_face_position=True`) |

The label `y` is the screen pixel coordinate `(x_dot, y_dot)` of the calibration point currently displayed.

### What the 360 rotation contributes

The eye landmark features are explicitly head-pose-normalized — two samples at the same gaze direction but different head angles will have similar landmark features. However, **yaw, pitch, and roll vary continuously** as the head rotates. During the 4-second head movement phase at a single dot, the model collects dozens of samples like:

```
(landmarks ≈ stable,  yaw=−15°, pitch=+5°,  roll=0°)  → (dot_x, dot_y)
(landmarks ≈ stable,  yaw=0°,   pitch=+10°, roll=−4°) → (dot_x, dot_y)
(landmarks ≈ stable,  yaw=+12°, pitch=−3°,  roll=+3°) → (dot_x, dot_y)
...
```

All labelled with the same target. The model therefore learns that this gaze direction maps to `(dot_x, dot_y)` across a full sweep of head orientations. At inference time, if the user's head drifts or tilts, the model has seen similar pose configurations during training and can still predict the correct gaze point.

### Training

After all calibration points are collected (still-head samples + 360-movement samples concatenated), a single call to `gaze_estimator.train(X, y)` fits the regression model on the full dataset. There is no special weighting between still and moving-head samples.

---

## Usage

```bash
# Vertical calibration with head-rotation (recommended)
python src/eyetrax/app/keyboard_demo.py \
    --calibration vertical \
    --filter kalman \
    --camera 0 \
    --multi-position

# 9-point with head-rotation
python src/eyetrax/app/keyboard_demo.py \
    --calibration 9p \
    --filter kalman \
    --camera 0 \
    --multi-position
```

### Calibration procedure for the user

1. Face the camera and hold still — the countdown will begin automatically
2. For each calibration dot that appears:
   - **Green dot / white arc** — look at the dot and keep your head still
   - **Orange dot / cyan orbiting indicator** — keep your eyes on the dot and slowly rotate your head in a full circle, following the orbiting cyan dot as a guide
3. Repeat for all points — calibration completes and the main application starts

---

## Design Rationale

### Why replace 3 rounds with per-point 360 rotation?

The original 3-position approach collected data from three distinct head heights, which improved robustness to vertical head shifts. However:
- It required the user to physically reposition (above / below camera) which is awkward, especially for assistive technology users
- It only sampled 3 discrete vertical positions, leaving gaps in the pose distribution
- The zone-locking UI (waiting 2 seconds in the right band) added friction

The per-point 360 approach:
- Requires no repositioning — the user stays seated normally
- Samples a continuous range of yaw, pitch, and roll simultaneously
- Integrates naturally into the existing calibration loop with no extra UI screens
- Collects more pose-diverse data per point (4s of movement vs a single static position)

### Why keep `multi_pose_d` separate from the standalone `--multi-pose` flag default?

The `--multi-pose` flag is a lighter augmentation (1 second of gentle head movement). Changing its default would affect existing workflows. `--multi-position` explicitly opts into the full 4-second 360 rotation, so the duration is set there rather than at the shared default level.
