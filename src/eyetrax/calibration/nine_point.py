import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size


def run_9_point_calibration(gaze_estimator, camera_index: int = 0,
                            multi_pose: bool = False, multi_pose_d: float = 1.0,
                            train: bool = True):
    """
    Standard nine-point calibration.

    When *train* is False the raw (features, targets) arrays are returned
    instead of training the model, so callers can pool data from multiple
    calibration rounds.
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return None if not train else None

    order = [
        (1, 1),
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
    ]
    pts = compute_grid_points(order, sw, sh)

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh,
                             multi_pose=multi_pose, multi_pose_d=multi_pose_d)
    cap.release()
    cv2.destroyAllWindows()
    if res is None:
        return None
    feats, targs = res
    if not feats:
        return None

    X, y = np.array(feats), np.array(targs)
    if train:
        gaze_estimator.train(X, y)
        return None
    return X, y
