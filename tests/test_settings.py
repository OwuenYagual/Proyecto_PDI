"""Pruebas unitarias para la configuracion central de Robot Mimic."""

import math
import unittest
from unittest.mock import patch

from config import settings


class SettingsValidationTests(unittest.TestCase):
    def assert_invalid(self, name: str, value: object) -> None:
        """Parchea una constante, comprueba el error y verifica su restauracion."""
        original = getattr(settings, name)
        with patch.object(settings, name, value):
            with self.assertRaisesRegex(ValueError, name):
                settings.validate_settings()
        self.assertEqual(getattr(settings, name), original)

    def test_default_settings_are_valid(self) -> None:
        self.assertIsNone(settings.validate_settings())

    def test_security_defaults_match_operational_contract(self) -> None:
        self.assertEqual(settings.CAMERA_MAX_CONSECUTIVE_FAILURES, 3)
        self.assertEqual(settings.HAND_WRIST_MATCH_MAX_RATIO, 0.15)
        self.assertEqual(settings.MIN_ARM_SEGMENT_LENGTH_PX, 5.0)
        self.assertEqual(settings.COPPELIASIM_CONNECTION_TIMEOUT_MS, 10000)
        self.assertEqual(settings.COPPELIASIM_REQUEST_TIMEOUT_MS, 500)
        self.assertEqual(settings.COPPELIASIM_COMMAND_MAX_AGE_MS, 250)
        self.assertEqual(settings.COPPELIASIM_POSE_LOSS_ESTOP_MS, 750)
        self.assertEqual(settings.COPPELIASIM_COLLISION_CHECK_HZ, 20.0)
        self.assertTrue(settings.COPPELIASIM_VIEW_ENABLED)
        self.assertEqual(
            settings.COPPELIASIM_VIEW_SENSOR_PATH,
            "/RobotMimicVisionSensor",
        )
        self.assertEqual(settings.COPPELIASIM_VIEW_HZ, 10.0)
        self.assertEqual(
            settings.COPPELIASIM_VIEW_CAMERA_POSITION,
            (-2.75, -1.55, 1.05),
        )
        self.assertEqual(
            settings.COPPELIASIM_VIEW_TARGET_POSITION,
            (-0.3, 0.0, 0.52),
        )
        self.assertEqual(settings.COPPELIASIM_VIEW_ANGLE_DEG, 50.0)
        self.assertEqual(settings.COPPELIASIM_BASE_YAW_DEG, 90.0)
        self.assertEqual(settings.COPPELIASIM_SHOULDER_DIRECTION, -1.0)
        self.assertEqual(settings.COPPELIASIM_SHOULDER_HUMAN_NEUTRAL_DEG, 90.0)
        self.assertEqual(
            settings.COPPELIASIM_SHOULDER_NEUTRAL_DEADBAND_DEG,
            15.0,
        )
        self.assertEqual(
            (
                settings.COPPELIASIM_SHOULDER_MIN_DEG,
                settings.COPPELIASIM_SHOULDER_MAX_DEG,
            ),
            (-90.0, 90.0),
        )
        self.assertEqual(settings.COPPELIASIM_ELBOW_DIRECTION, 1.0)
        self.assertEqual(settings.COPPELIASIM_ELBOW_HUMAN_STRAIGHT_DEG, 180.0)
        self.assertEqual(
            (
                settings.COPPELIASIM_ELBOW_MIN_DEG,
                settings.COPPELIASIM_ELBOW_MAX_DEG,
            ),
            (0.0, 135.0),
        )
        self.assertEqual(
            (settings.COPPELIASIM_VIEW_WIDTH, settings.COPPELIASIM_VIEW_HEIGHT),
            (640, 480),
        )
        self.assertEqual(settings.COPPELIASIM_HOME_TOLERANCE_DEG, 1.0)
        self.assertEqual(settings.COPPELIASIM_HOME_TIMEOUT_S, 5.0)

    def test_unused_landmark_list_was_removed(self) -> None:
        self.assertFalse(hasattr(settings, "LANDMARKS_OF_INTEREST"))

    def test_rejects_invalid_camera_configuration(self) -> None:
        invalid_values = (
            ("CAMERA_INDEX", -1),
            ("CAMERA_INDEX", True),
            ("FRAME_WIDTH", 0),
            ("FRAME_HEIGHT", 0),
            ("CAMERA_FPS", 0.0),
            ("CAMERA_FPS", math.inf),
            ("CAMERA_MAX_CONSECUTIVE_FAILURES", 0),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_invalid_booleans_and_endpoint(self) -> None:
        invalid_values = (
            ("APPLY_GAUSSIAN", 1),
            ("COPPELIASIM_ENABLED", "yes"),
            ("COPPELIASIM_VIEW_ENABLED", 1),
            ("COPPELIASIM_HOST", "   "),
            ("COPPELIASIM_VIEW_SENSOR_PATH", "RobotMimicVisionSensor"),
            ("COPPELIASIM_VIEW_SENSOR_PATH", "   "),
            ("COPPELIASIM_PORT", 0),
            ("COPPELIASIM_PORT", 65536),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_invalid_confidences_and_visibility(self) -> None:
        for name in (
            "MIN_DETECTION_CONFIDENCE",
            "MIN_TRACKING_CONFIDENCE",
            "VISIBILITY_THRESHOLD",
        ):
            for value in (-0.01, 1.01, math.nan, True):
                with self.subTest(name=name, value=value):
                    self.assert_invalid(name, value)

    def test_accepts_closed_confidence_interval_boundaries(self) -> None:
        for name in (
            "MIN_DETECTION_CONFIDENCE",
            "MIN_TRACKING_CONFIDENCE",
            "VISIBILITY_THRESHOLD",
        ):
            for value in (0.0, 1.0):
                with self.subTest(name=name, value=value):
                    with patch.object(settings, name, value):
                        self.assertIsNone(settings.validate_settings())

    def test_rejects_invalid_model_complexities(self) -> None:
        invalid_values = (
            ("POSE_MODEL_COMPLEXITY", -1),
            ("POSE_MODEL_COMPLEXITY", 3),
            ("HAND_MODEL_COMPLEXITY", -1),
            ("HAND_MODEL_COMPLEXITY", 2),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_invalid_processing_intervals_and_smoothing(self) -> None:
        for name in (
            "HANDS_PROCESS_INTERVAL",
            "SMOOTHING_WINDOW",
            "FPS_SMOOTHING_WINDOW",
        ):
            for value in (0, 1.5, True):
                with self.subTest(name=name, value=value):
                    self.assert_invalid(name, value)

    def test_rejects_invalid_gaussian_kernel(self) -> None:
        for value in ((0, 5), (4, 5), (5,), (5, 5, 5), [5, 5], (True, 5)):
            with self.subTest(value=value):
                self.assert_invalid("GAUSSIAN_KERNEL", value)

    def test_rejects_invalid_vision_geometry_thresholds(self) -> None:
        invalid_values = (
            ("HAND_WRIST_MATCH_MAX_RATIO", 0.0),
            ("HAND_WRIST_MATCH_MAX_RATIO", 1.01),
            ("MIN_ARM_SEGMENT_LENGTH_PX", 0.0),
            ("MIN_ARM_SEGMENT_LENGTH_PX", math.nan),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_invalid_gripper_distance_range(self) -> None:
        invalid_values = (
            ("GRIPPER_D_MIN", -1.0),
            ("GRIPPER_D_MIN", settings.GRIPPER_D_MAX),
            ("GRIPPER_D_MAX", settings.GRIPPER_D_MIN),
            ("GRIPPER_D_MAX", math.inf),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_non_positive_integer_timeouts(self) -> None:
        for name in (
            "COPPELIASIM_CONNECTION_TIMEOUT_MS",
            "COPPELIASIM_REQUEST_TIMEOUT_MS",
            "COPPELIASIM_COMMAND_MAX_AGE_MS",
            "COPPELIASIM_POSE_LOSS_ESTOP_MS",
        ):
            for value in (0, 1.5, True):
                with self.subTest(name=name, value=value):
                    self.assert_invalid(name, value)

    def test_pose_loss_estop_must_allow_hold_before_stopping(self) -> None:
        with patch.object(settings, "COPPELIASIM_COMMAND_MAX_AGE_MS", 750), patch.object(
            settings,
            "COPPELIASIM_POSE_LOSS_ESTOP_MS",
            750,
        ):
            with self.assertRaisesRegex(ValueError, "permitir hold antes del ESTOP"):
                settings.validate_settings()

    def test_rejects_non_positive_or_non_finite_motion_settings(self) -> None:
        for name in (
            "COPPELIASIM_UPDATE_HZ",
            "COPPELIASIM_COLLISION_CHECK_HZ",
            "COPPELIASIM_VIEW_HZ",
            "COPPELIASIM_HOME_TOLERANCE_DEG",
            "COPPELIASIM_HOME_TIMEOUT_S",
            "COPPELIASIM_MAX_VELOCITY_DEG",
            "COPPELIASIM_MAX_ACCELERATION_DEG",
            "COPPELIASIM_MAX_JERK_DEG",
        ):
            for value in (0.0, math.inf, True):
                with self.subTest(name=name, value=value):
                    self.assert_invalid(name, value)

    def test_rejects_joint_indices_out_of_range_or_equal(self) -> None:
        invalid_values = (
            ("COPPELIASIM_JOINT_COUNT", 0),
            ("COPPELIASIM_JOINT_COUNT", 5),
            ("COPPELIASIM_JOINT_COUNT", 7),
            ("COPPELIASIM_SHOULDER_INDEX", -1),
            ("COPPELIASIM_SHOULDER_INDEX", settings.COPPELIASIM_JOINT_COUNT),
            ("COPPELIASIM_ELBOW_INDEX", settings.COPPELIASIM_JOINT_COUNT),
            ("COPPELIASIM_ELBOW_INDEX", settings.COPPELIASIM_SHOULDER_INDEX),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_invalid_simulator_view_resolution(self) -> None:
        for name in ("COPPELIASIM_VIEW_WIDTH", "COPPELIASIM_VIEW_HEIGHT"):
            for value in (0, 1.5, True):
                with self.subTest(name=name, value=value):
                    self.assert_invalid(name, value)

    def test_rejects_invalid_simulator_view_framing(self) -> None:
        invalid_vectors = (
            ("COPPELIASIM_VIEW_CAMERA_POSITION", [0.0, -1.55, 0.45]),
            ("COPPELIASIM_VIEW_CAMERA_POSITION", (0.0, math.nan, 0.45)),
            ("COPPELIASIM_VIEW_TARGET_POSITION", (0.0, 0.0)),
            ("COPPELIASIM_VIEW_TARGET_POSITION", (0.0, True, 0.45)),
        )
        for name, value in invalid_vectors:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

        for value in (0.0, 180.0, math.inf, True):
            with self.subTest(angle=value):
                self.assert_invalid("COPPELIASIM_VIEW_ANGLE_DEG", value)

    def test_rejects_vertical_simulator_view_direction(self) -> None:
        with patch.object(
            settings,
            "COPPELIASIM_VIEW_CAMERA_POSITION",
            (0.0, 0.0, 1.0),
        ), patch.object(
            settings,
            "COPPELIASIM_VIEW_TARGET_POSITION",
            (0.0, 0.0, 0.0),
        ):
            with self.assertRaisesRegex(ValueError, "vista frontal"):
                settings.validate_settings()

    def test_rejects_non_finite_base_yaw(self) -> None:
        for value in (math.nan, math.inf, True, "90"):
            with self.subTest(value=value):
                self.assert_invalid("COPPELIASIM_BASE_YAW_DEG", value)

    def test_rejects_invalid_arm_calibration(self) -> None:
        invalid_values = (
            ("COPPELIASIM_SHOULDER_HUMAN_NEUTRAL_DEG", 181.0),
            ("COPPELIASIM_SHOULDER_ROBOT_NEUTRAL_DEG", math.nan),
            ("COPPELIASIM_SHOULDER_DIRECTION", 0.0),
            ("COPPELIASIM_SHOULDER_DIRECTION", True),
            ("COPPELIASIM_SHOULDER_NEUTRAL_DEADBAND_DEG", -1.0),
            ("COPPELIASIM_SHOULDER_NEUTRAL_DEADBAND_DEG", 90.0),
            ("COPPELIASIM_SHOULDER_NEUTRAL_DEADBAND_DEG", math.inf),
            ("COPPELIASIM_ELBOW_HUMAN_STRAIGHT_DEG", -1.0),
            ("COPPELIASIM_ELBOW_ROBOT_STRAIGHT_DEG", math.inf),
            ("COPPELIASIM_ELBOW_DIRECTION", 0.5),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_unordered_or_non_finite_joint_limits(self) -> None:
        invalid_values = (
            ("COPPELIASIM_SHOULDER_MIN_DEG", math.nan),
            (
                "COPPELIASIM_SHOULDER_MAX_DEG",
                settings.COPPELIASIM_SHOULDER_MIN_DEG,
            ),
            (
                "COPPELIASIM_ELBOW_MIN_DEG",
                settings.COPPELIASIM_ELBOW_MAX_DEG,
            ),
            ("COPPELIASIM_ELBOW_MAX_DEG", math.inf),
        )
        for name, value in invalid_values:
            with self.subTest(name=name, value=value):
                self.assert_invalid(name, value)

    def test_rejects_gripper_threshold_outside_unit_interval(self) -> None:
        for value in (-0.01, 1.01, math.nan, True):
            with self.subTest(value=value):
                self.assert_invalid("COPPELIASIM_GRIPPER_THRESHOLD", value)


if __name__ == "__main__":
    unittest.main()
