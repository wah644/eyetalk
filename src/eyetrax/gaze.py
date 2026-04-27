from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from eyetrax.constants import (
    FACE_POSITION_INDICES,
    LEFT_EYE_INDICES,
    MUTUAL_INDICES,
    RIGHT_EYE_INDICES,
)
from eyetrax.models import BaseModel, create_model


class GazeEstimator:
    def __init__(
        self,
        model_name: str = "ridge",
        model_kwargs: dict | None = None,
        ear_history_len: int = 50,
        blink_threshold_ratio: float = 0.8,
        min_history: int = 15,
        landmark_alpha: float = 0.7,
        feature_alpha: float | None = None,
        include_face_position: bool = False,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.model: BaseModel = create_model(model_name, **(model_kwargs or {}))

        self._ear_history = deque(maxlen=ear_history_len)
        self._blink_ratio = blink_threshold_ratio
        self._min_history = min_history

        self._last_pose: np.ndarray | None = None
        self._last_ipd_pixels: float | None = None

        self._landmark_alpha = landmark_alpha
        self._prev_landmarks: np.ndarray | None = None
        self._feature_alpha = feature_alpha
        self._prev_features: np.ndarray | None = None

        self._ref_pose: np.ndarray | None = None
        self._pose_damping: float = 0.0

        self._include_face_position = include_face_position
        self._last_face_anchor_y: float | None = None

        self.smoothing_enabled = False

    def extract_features(self, image):
        """
        Processes a single camera frame and returns a feature vector describing
        the user's gaze, along with a boolean indicating whether a blink was detected.

        The feature vector is built in four stages:
          1. Translation  – nose tip becomes the origin
          2. Rotation     – all landmarks rotated into a face-aligned frame
          3. Scale        – coordinates divided by inter-eye distance
          4. Augmentation – head-pose angles (and optionally face position) appended

        Returns (features, blink_detected), or (None, False) if no face is found.
        """

        # ── Run MediaPipe face mesh on the current frame ──────────────────────
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, False

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        # Collect all 478 MediaPipe landmarks as a (478, 3) float32 array.
        # Coordinates are normalised to [0, 1] by MediaPipe (x, y relative to
        # image size; z is depth relative to the nose tip).
        all_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        )

        # Optional EMA smoothing on raw landmark positions to reduce jitter.
        # alpha=1.0 means no smoothing; lower values blend more with the previous frame.
        if (self.smoothing_enabled
                and self._prev_landmarks is not None
                and self._landmark_alpha < 1.0):
            a = self._landmark_alpha
            all_points = a * all_points + (1 - a) * self._prev_landmarks
        self._prev_landmarks = all_points.copy()

        # ── Stage 1: Translation — make the nose tip the origin ───────────────
        # Landmark 4 is the nose tip. Subtracting it from every point means all
        # coordinates are now expressed relative to the nose, removing the effect
        # of the face sitting at different positions within the camera frame.
        nose_anchor = all_points[4]
        left_corner = all_points[33]    # left eye outer corner
        right_corner = all_points[263]  # right eye outer corner
        top_of_head = all_points[10]    # top-of-head landmark

        shifted_points = all_points - nose_anchor

        # ── Stage 2: Rotation — build a face-aligned coordinate frame ─────────
        # x-axis: points from the left eye outer corner to the right eye outer corner.
        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9

        # y-axis: approximate upward direction (nose → top of head), then remove
        # its component along x so the two axes are perfectly orthogonal
        # (Gram–Schmidt orthogonalisation).
        y_approx = top_of_head - nose_anchor
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9

        # z-axis: cross product of x and y gives the axis pointing out of the face.
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9

        # R is the rotation matrix whose columns are the three face-aligned axes.
        # Multiplying by R.T rotates every landmark from camera space into
        # face space, so coordinates become relative to the face's own orientation
        # rather than the camera's orientation.
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T

        # ── Stage 3: Scale — divide by inter-eye distance ─────────────────────
        # Computing the eye-corner separation in the rotated frame and dividing
        # by it removes the effect of the user sitting closer or further from
        # the camera (the landmarks shrink or grow with distance, but their
        # ratio to the inter-eye distance stays constant).
        left_corner_rot = R.T @ (left_corner - nose_anchor)
        right_corner_rot = R.T @ (right_corner - nose_anchor)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist

        # ── Select the landmark subset used as gaze features ──────────────────
        # Only the eye-region landmarks (left eye, right eye, and shared mutual
        # landmarks) are kept — 161 landmarks × 3 coords = 483 values.
        subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
        eye_landmarks = rotated_points[subset_indices]
        features = eye_landmarks.flatten()

        # ── Stage 4a: Append head-pose angles ─────────────────────────────────
        # The rotation matrix R encodes how much the head is turned.  Three
        # Euler angles are extracted from it:
        #   yaw   – left/right head rotation
        #   pitch – up/down head tilt
        #   roll  – head tilt toward the shoulder
        # These three values are appended to the 483 landmark features, giving
        # a final feature vector of 486 values.
        yaw   = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll  = np.arctan2(R[2, 1], R[2, 2])
        features = np.concatenate([features, [yaw, pitch, roll]])

        # Store the current pose so other methods (pose_within_tolerance,
        # predict with pose_damping) can reference it.
        self._last_pose = np.array([yaw, pitch, roll])
        self._last_face_anchor_y = float(all_points[4, 1])

        # ── Stage 4b: Optionally append face-position features ────────────────
        # When --multi-position is enabled, a small set of landmarks describing
        # the face's vertical position in the frame are appended.  This lets the
        # model account for the user sitting at different heights relative to the
        # camera (e.g. head raised / lowered).
        if self._include_face_position:
            face_pos = all_points[FACE_POSITION_INDICES, :2].flatten()
            features = np.concatenate([features, face_pos])

        # ── Inter-pupil distance in pixels (used elsewhere, not in features) ──
        # Iris landmarks 468/473 are in pixel space; the result is stored for
        # any downstream code that needs a pixel-space size reference.
        h, w = image.shape[:2]
        left_iris  = np.array([landmarks[468].x * w, landmarks[468].y * h])
        right_iris = np.array([landmarks[473].x * w, landmarks[473].y * h])
        self._last_ipd_pixels = float(np.linalg.norm(right_iris - left_iris))

        # ── Blink detection via Eye Aspect Ratio (EAR) ────────────────────────
        # EAR = eye_height / eye_width.  When the eye closes the height drops
        # toward zero, so EAR falls sharply.  A rolling mean of recent EAR
        # values forms a personal baseline; a frame is called a blink when EAR
        # drops below (baseline × blink_threshold_ratio).
        left_eye_inner  = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer  = np.array([landmarks[33].x,  landmarks[33].y])
        left_eye_top    = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner  = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer  = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top    = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width  = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top   - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width  = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top   - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        # Average EAR across both eyes for robustness.
        EAR = (left_EAR + right_EAR) / 2

        self._ear_history.append(EAR)
        if len(self._ear_history) >= self._min_history:
            # Adaptive threshold: percentage of the user's own rolling mean EAR.
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            # Not enough history yet — use a fixed fallback threshold.
            thr = 0.2
        blink_detected = EAR < thr

        # ── Optional EMA smoothing on the final feature vector ────────────────
        # Applied after blink detection so the raw EAR used above is not blurred.
        if (self.smoothing_enabled
                and self._feature_alpha is not None
                and self._feature_alpha < 1.0):
            if self._prev_features is not None:
                a = self._feature_alpha
                features = a * features + (1 - a) * self._prev_features
            self._prev_features = features.copy()

        return features, blink_detected

    def pose_within_tolerance(self, yaw_tol: float = 0.26,
                              pitch_tol: float = 0.17) -> bool:
        """Check if current head pose is within tolerance of calibration pose.

        Tolerances are in radians (defaults: ~15 deg yaw, ~10 deg pitch).
        Returns True if no reference is set (pre-calibration).
        """
        if self._ref_pose is None or self._last_pose is None:
            return True
        delta = np.abs(self._last_pose - self._ref_pose)
        return bool(delta[0] <= yaw_tol and delta[1] <= pitch_tol)

    def save_model(self, path: str | Path):
        import pickle
        payload = {
            "model": self.model,
            "vertical_only": getattr(self, "vertical_only", False),
            "vertical_center_x": getattr(self, "vertical_center_x", None),
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    def load_model(self, path: str | Path):
        import pickle
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, dict) and "model" in data:
            self.model = data["model"]
            self.vertical_only = data.get("vertical_only", False)
            self.vertical_center_x = data.get("vertical_center_x", None)
        else:
            # Legacy: plain BaseModel pickle (no vertical_only info)
            self.model = data
        self.enable_smoothing()

    def enable_smoothing(self):
        """Clear EMA buffers and enable runtime smoothing."""
        self._prev_landmarks = None
        self._prev_features = None
        self.smoothing_enabled = True

    def train(self, X, y, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self._ref_pose = X[:, self._pose_slice].mean(axis=0)
        self.model.train(X, y, variable_scaling)
        self.enable_smoothing()

    @property
    def _pose_slice(self) -> slice:
        """Column slice for the yaw/pitch/roll features within the feature vector."""
        if self._include_face_position:
            n_face = len(FACE_POSITION_INDICES) * 2
            return slice(-(n_face + 3), -n_face)
        return slice(-3, None)

    def predict(self, X, pose_damping: float | None = None):
        """
        Predicts gaze location.

        When *pose_damping* > 0 the head-pose features are blended toward the
        reference pose recorded during calibration, suppressing drift caused by
        head movement.  0.0 = no damping, 1.0 = fully clamp to calibration pose.
        """
        d = pose_damping if pose_damping is not None else self._pose_damping
        if d > 0 and self._ref_pose is not None:
            X = X.copy()
            s = self._pose_slice
            X[:, s] = (1 - d) * X[:, s] + d * self._ref_pose
        return self.model.predict(X)
