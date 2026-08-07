"""Pruebas unitarias de M2 sin inicializar modelos de MediaPipe."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from modules.m2_mediapipe import (
    ELBOW,
    INDEX_TIP,
    SHOULDER,
    THUMB_TIP,
    WRIST_HAND,
    WRIST_POSE,
    PoseDetector,
)


def _landmark(x: float, y: float, visibility: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def _pose_result() -> SimpleNamespace:
    landmarks = [_landmark(0.0, 0.0) for _ in range(17)]
    landmarks[SHOULDER] = _landmark(0.25, 0.20)
    landmarks[ELBOW] = _landmark(0.50, 0.30)
    landmarks[WRIST_POSE] = _landmark(0.80, 0.40)
    return SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=landmarks),
    )


def _hand(wrist_x: float, wrist_y: float, tip_offset: float) -> SimpleNamespace:
    landmarks = [_landmark(wrist_x, wrist_y) for _ in range(9)]
    landmarks[THUMB_TIP] = _landmark(wrist_x + tip_offset, wrist_y)
    landmarks[INDEX_TIP] = _landmark(wrist_x, wrist_y + tip_offset)
    return SimpleNamespace(landmark=landmarks)


class _Processor:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def process(self, _frame: np.ndarray) -> object:
        self.calls += 1
        return self.result


def _detector(pose_result: object, hands_result: object) -> PoseDetector:
    detector = PoseDetector.__new__(PoseDetector)
    detector.pose = _Processor(pose_result)
    detector.hands = _Processor(hands_result)
    detector._cached_hands = []
    detector._hands_initialized = False
    detector._hands_frames_since_process = 0
    return detector


class PoseDetectorTests(unittest.TestCase):
    def test_uses_right_arm_landmark_indices(self) -> None:
        self.assertEqual((SHOULDER, ELBOW, WRIST_POSE), (12, 14, 16))

    def test_constructor_configures_two_hands(self) -> None:
        pose_constructor = Mock(return_value=Mock())
        hands_constructor = Mock(return_value=Mock())
        fake_mediapipe = SimpleNamespace(
            solutions=SimpleNamespace(
                pose=SimpleNamespace(Pose=pose_constructor),
                hands=SimpleNamespace(Hands=hands_constructor),
                drawing_utils=object(),
            ),
        )

        with patch("modules.m2_mediapipe.mp", fake_mediapipe):
            PoseDetector()

        self.assertEqual(hands_constructor.call_args.kwargs["max_num_hands"], 2)

    def test_process_uses_actual_frame_size_and_selects_nearest_hand(self) -> None:
        far_hand = _hand(0.10, 0.10, 0.01)
        near_hand = _hand(0.78, 0.42, 0.04)
        detector = _detector(
            _pose_result(),
            SimpleNamespace(multi_hand_landmarks=[far_hand, near_hand]),
        )
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        result = detector.process(frame)

        self.assertEqual(result["arm"][SHOULDER]["x"], 50.0)
        self.assertEqual(result["arm"][SHOULDER]["y"], 20.0)
        self.assertEqual(result["arm"][WRIST_POSE]["x"], 160.0)
        self.assertEqual(result["hand"][WRIST_HAND]["x"], 160.0)
        # El offset relativo de 0.04 sobre un frame de 200 px identifica la
        # segunda mano; la primera tenía un offset de solo 2 px.
        self.assertEqual(
            result["hand"][THUMB_TIP]["x"] - result["hand"][WRIST_HAND]["x"],
            8.0,
        )
        self.assertTrue(frame.flags.writeable)

    def test_rejects_hand_outside_wrist_match_threshold(self) -> None:
        detector = _detector(
            _pose_result(),
            SimpleNamespace(multi_hand_landmarks=[_hand(0.05, 0.05, 0.02)]),
        )

        result = detector.process(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertIsNone(result["hand"])

    def test_does_not_associate_hand_to_invisible_pose_wrist(self) -> None:
        pose_result = _pose_result()
        pose_result.pose_landmarks.landmark[WRIST_POSE].visibility = 0.0
        detector = _detector(
            pose_result,
            SimpleNamespace(multi_hand_landmarks=[_hand(0.80, 0.40, 0.02)]),
        )

        result = detector.process(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertIsNone(result["hand"])

    def test_extract_hands_limits_external_results_to_two(self) -> None:
        hands_result = SimpleNamespace(
            multi_hand_landmarks=[
                _hand(0.1, 0.1, 0.01),
                _hand(0.2, 0.2, 0.01),
                _hand(0.3, 0.3, 0.01),
            ],
        )

        extracted = PoseDetector._extract_hands(hands_result, 200, 100)

        self.assertEqual(len(extracted), 2)

    def test_non_finite_arm_is_not_visible(self) -> None:
        result = {
            "arm": {
                SHOULDER: {"x": 1.0, "y": 1.0, "visibility": 1.0},
                ELBOW: {"x": float("nan"), "y": 2.0, "visibility": 1.0},
                WRIST_POSE: {"x": 3.0, "y": 3.0, "visibility": 1.0},
            }
        }

        detector = PoseDetector.__new__(PoseDetector)

        self.assertFalse(detector.is_arm_visible(result))


if __name__ == "__main__":
    unittest.main()
