"""M2: extracción y asociación de landmarks de MediaPipe."""

from __future__ import annotations

import math
from numbers import Real

import cv2
import mediapipe as mp
import numpy as np

from config.settings import (
    HAND_MODEL_COMPLEXITY,
    HAND_WRIST_MATCH_MAX_RATIO,
    HANDS_PROCESS_INTERVAL,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    POSE_MODEL_COMPLEXITY,
    VISIBILITY_THRESHOLD,
)

# MediaPipe Pose: brazo derecho de la persona, sin invertir la inferencia.
SHOULDER = 12
ELBOW = 14
WRIST_POSE = 16

# MediaPipe Hands.
WRIST_HAND = 0
THUMB_TIP = 4
INDEX_TIP = 8


class PoseDetector:
    """Detecta el brazo derecho y asocia la mano que pertenece a su muñeca."""

    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=HAND_MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._cached_hands: list[dict] = []
        self._hands_initialized = False
        self._hands_frames_since_process = HANDS_PROCESS_INTERVAL

    def process(
        self,
        rgb_frame: np.ndarray,
        process_hands: bool = True,
    ) -> dict | None:
        """Procesa un frame RGB sin espejo y devuelve brazo y mano asociados.

        Las coordenadas se convierten usando el tamaño real del frame recibido,
        no la resolución solicitada a la cámara.
        """
        if rgb_frame.ndim < 2:
            raise ValueError("El frame RGB no tiene dimensiones de imagen válidas.")
        frame_height, frame_width = rgb_frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("El frame RGB está vacío.")

        was_writeable = rgb_frame.flags.writeable
        rgb_frame.flags.writeable = False
        try:
            pose_results = self.pose.process(rgb_frame)
            if not pose_results.pose_landmarks:
                self._reset_hand_cache()
                return None

            arm_landmarks = self._extract_arm(
                pose_results.pose_landmarks,
                frame_width,
                frame_height,
                getattr(pose_results, "pose_world_landmarks", None),
            )
            hand_candidates = self._process_hands_if_needed(
                rgb_frame,
                enabled=process_hands,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        finally:
            rgb_frame.flags.writeable = was_writeable

        hand_landmarks = self._select_nearest_hand(
            arm_landmarks,
            hand_candidates,
            frame_width,
            frame_height,
        )
        if hand_landmarks is not None:
            hand_landmarks = self._sync_wrists(arm_landmarks, hand_landmarks)

        return {"arm": arm_landmarks, "hand": hand_landmarks}

    def draw_skeleton(self, bgr_frame: np.ndarray, result: dict | None) -> np.ndarray:
        """Dibuja solo landmarks finitos y suficientemente visibles."""
        if not isinstance(result, dict):
            return bgr_frame

        frame = bgr_frame.copy()
        arm = result.get("arm")
        if isinstance(arm, dict):
            for start_idx, end_idx in (
                (SHOULDER, ELBOW),
                (ELBOW, WRIST_POSE),
            ):
                lm_start = arm.get(start_idx)
                lm_end = arm.get(end_idx)
                if not (
                    self._landmark_drawable(lm_start, require_visibility=True)
                    and self._landmark_drawable(lm_end, require_visibility=True)
                ):
                    continue
                cv2.line(
                    frame,
                    (int(lm_start["x"]), int(lm_start["y"])),
                    (int(lm_end["x"]), int(lm_end["y"])),
                    (34, 197, 94),
                    2,
                )

            for idx in (SHOULDER, ELBOW, WRIST_POSE):
                lm = arm.get(idx)
                if not self._landmark_drawable(lm, require_visibility=True):
                    continue
                cx, cy = int(lm["x"]), int(lm["y"])
                cv2.circle(frame, (cx, cy), 5, (34, 197, 94), -1)
                cv2.putText(
                    frame,
                    str(idx),
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (34, 197, 94),
                    1,
                    cv2.LINE_AA,
                )

        hand = result.get("hand")
        if isinstance(hand, dict) and all(
            self._landmark_drawable(hand.get(idx))
            for idx in (WRIST_HAND, THUMB_TIP, INDEX_TIP)
        ):
            wrist = hand[WRIST_HAND]
            thumb = hand[THUMB_TIP]
            index = hand[INDEX_TIP]
            cv2.line(
                frame,
                (int(thumb["x"]), int(thumb["y"])),
                (int(index["x"]), int(index["y"])),
                (0, 165, 255),
                2,
            )
            for tip in (thumb, index):
                cv2.line(
                    frame,
                    (int(wrist["x"]), int(wrist["y"])),
                    (int(tip["x"]), int(tip["y"])),
                    (0, 165, 255),
                    1,
                )
            labels = {WRIST_HAND: "W", THUMB_TIP: "4", INDEX_TIP: "8"}
            for idx in (WRIST_HAND, THUMB_TIP, INDEX_TIP):
                lm = hand[idx]
                cx, cy = int(lm["x"]), int(lm["y"])
                cv2.circle(frame, (cx, cy), 4, (0, 165, 255), -1)
                cv2.putText(
                    frame,
                    labels[idx],
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def is_arm_visible(self, result: dict | None) -> bool:
        """Indica si los tres landmarks del brazo son finitos y visibles."""
        if not isinstance(result, dict) or not isinstance(result.get("arm"), dict):
            return False
        arm = result["arm"]
        return all(
            self._landmark_drawable(arm.get(idx), require_visibility=True)
            for idx in (SHOULDER, ELBOW, WRIST_POSE)
        )

    def release(self) -> None:
        self.pose.close()
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    def _process_hands_if_needed(
        self,
        rgb_frame: np.ndarray,
        enabled: bool,
        frame_width: int,
        frame_height: int,
    ) -> list[dict]:
        if not enabled:
            self._reset_hand_cache()
            return []

        self._hands_frames_since_process += 1
        should_process = (
            not self._hands_initialized
            or self._hands_frames_since_process >= HANDS_PROCESS_INTERVAL
        )
        if should_process:
            hands_results = self.hands.process(rgb_frame)
            self._cached_hands = self._extract_hands(
                hands_results,
                frame_width,
                frame_height,
            )
            self._hands_initialized = True
            self._hands_frames_since_process = 0

        return self._cached_hands

    def _reset_hand_cache(self) -> None:
        self._cached_hands = []
        self._hands_initialized = False
        self._hands_frames_since_process = HANDS_PROCESS_INTERVAL

    @staticmethod
    def _extract_arm(
        pose_landmarks,
        frame_width: int,
        frame_height: int,
        pose_world_landmarks=None,
    ) -> dict:
        extracted = {}
        world_points = getattr(pose_world_landmarks, "landmark", None)
        for idx in (SHOULDER, ELBOW, WRIST_POSE):
            landmark = pose_landmarks.landmark[idx]
            point = {
                "x": landmark.x * frame_width,
                "y": landmark.y * frame_height,
                "visibility": landmark.visibility,
            }
            if world_points is not None and len(world_points) > idx:
                world = world_points[idx]
                point.update(
                    {
                        "world_x": world.x,
                        "world_y": world.y,
                        "world_z": world.z,
                    }
                )
            extracted[idx] = point
        return extracted

    @staticmethod
    def _extract_hands(
        hands_results,
        frame_width: int,
        frame_height: int,
    ) -> list[dict]:
        all_landmarks = getattr(hands_results, "multi_hand_landmarks", None)
        if not all_landmarks:
            return []

        extracted_hands = []
        for hand_landmarks in all_landmarks[:2]:
            extracted = {}
            for idx in (WRIST_HAND, THUMB_TIP, INDEX_TIP):
                landmark = hand_landmarks.landmark[idx]
                extracted[idx] = {
                    "x": landmark.x * frame_width,
                    "y": landmark.y * frame_height,
                    "visibility": 1.0,
                }
            extracted_hands.append(extracted)
        return extracted_hands

    @classmethod
    def _select_nearest_hand(
        cls,
        arm: dict,
        candidates: list[dict],
        frame_width: int,
        frame_height: int,
    ) -> dict | None:
        pose_wrist = arm.get(WRIST_POSE) if isinstance(arm, dict) else None
        if not cls._landmark_drawable(pose_wrist, require_visibility=True):
            return None

        nearest = None
        nearest_distance = math.inf
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            hand_wrist = candidate.get(WRIST_HAND)
            if not cls._landmark_drawable(hand_wrist):
                continue
            distance = math.hypot(
                pose_wrist["x"] - hand_wrist["x"],
                pose_wrist["y"] - hand_wrist["y"],
            )
            if distance < nearest_distance:
                nearest = candidate
                nearest_distance = distance

        maximum_distance = HAND_WRIST_MATCH_MAX_RATIO * math.hypot(
            frame_width,
            frame_height,
        )
        if nearest is None or nearest_distance > maximum_distance:
            return None
        return nearest

    @staticmethod
    def _sync_wrists(arm: dict, hand: dict) -> dict:
        pose_wrist = arm[WRIST_POSE]
        hand_wrist = hand[WRIST_HAND]
        offset_x = pose_wrist["x"] - hand_wrist["x"]
        offset_y = pose_wrist["y"] - hand_wrist["y"]
        return {
            idx: {
                "x": landmark["x"] + offset_x,
                "y": landmark["y"] + offset_y,
                "visibility": landmark["visibility"],
            }
            for idx, landmark in hand.items()
        }

    @staticmethod
    def _landmark_drawable(
        landmark: object,
        require_visibility: bool = False,
    ) -> bool:
        if not isinstance(landmark, dict):
            return False
        values = (landmark.get("x"), landmark.get("y"))
        if not all(
            isinstance(value, Real)
            and not isinstance(value, (bool, np.bool_))
            and math.isfinite(float(value))
            for value in values
        ):
            return False
        if not require_visibility:
            return True
        visibility = landmark.get("visibility")
        return (
            isinstance(visibility, Real)
            and not isinstance(visibility, (bool, np.bool_))
            and math.isfinite(float(visibility))
            and float(visibility) >= VISIBILITY_THRESHOLD
        )
