"""
Multi-position calibration: runs the chosen calibration method once, but
with an extended head-movement phase at every calibration point.  The user
keeps their eyes on each dot while slowly rotating their head in a full
circle, giving the model a diverse range of head-pose samples per gaze target.
"""
from __future__ import annotations

from eyetrax.calibration.five_point import run_5_point_calibration
from eyetrax.calibration.lissajous import run_lissajous_calibration
from eyetrax.calibration.nine_point import run_9_point_calibration
from eyetrax.calibration.vertical_enhanced_calibration import (
    run_vertical_enhanced_calibration,
    run_vertical_only_calibration,
    run_vertical_single_calibration,
)

# Seconds the user spends rotating their head at each calibration point
HEAD_MOVE_DURATION = 4.0


def run_multi_position_calibration(
    gaze_estimator,
    camera_index: int = 0,
    calibration_method: str = "9p",
    multi_pose: bool = False,  # kept for API compatibility; head movement always on
    single_column: bool = False,
) -> None:
    """Run a single calibration pass where the user rotates their head in a
    full circle at each point, training a model robust to head-position changes."""

    if calibration_method == "9p":
        run_9_point_calibration(
            gaze_estimator, camera_index=camera_index,
            multi_pose=True, multi_pose_d=HEAD_MOVE_DURATION,
        )
    elif calibration_method == "5p":
        run_5_point_calibration(
            gaze_estimator, camera_index=camera_index,
            multi_pose=True, multi_pose_d=HEAD_MOVE_DURATION,
        )
    elif calibration_method == "vertical":
        run_vertical_enhanced_calibration(
            gaze_estimator, camera_index=camera_index,
            multi_pose=True, multi_pose_d=HEAD_MOVE_DURATION,
            single_column=single_column,
        )
    elif calibration_method == "vertical_single":
        run_vertical_single_calibration(
            gaze_estimator, camera_index=camera_index,
            multi_pose=True, multi_pose_d=HEAD_MOVE_DURATION,
        )
    elif calibration_method == "vertical-only":
        run_vertical_only_calibration(
            gaze_estimator, camera_index=camera_index,
            multi_pose=True, multi_pose_d=HEAD_MOVE_DURATION,
        )
    elif calibration_method == "lissajous":
        run_lissajous_calibration(
            gaze_estimator, camera_index=camera_index,
        )
    else:
        print(f"[multi-position] Unknown calibration method '{calibration_method}', defaulting to 9p.")
        run_9_point_calibration(
            gaze_estimator, camera_index=camera_index,
            multi_pose=True, multi_pose_d=HEAD_MOVE_DURATION,
        )
