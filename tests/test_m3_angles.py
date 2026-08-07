"""Pruebas de validación y aislamiento de canales en M3."""

from dataclasses import FrozenInstanceError
import math
import unittest

from modules.commands import ArmCommand, GripperCommand, MimicCommand
from modules.m2_mediapipe import ELBOW, INDEX_TIP, SHOULDER, THUMB_TIP, WRIST_POSE
from modules.m3_angles import AngleCalculator


def _point(x: float, y: float, visibility: float = 1.0) -> dict:
    return {"x": x, "y": y, "visibility": visibility}


def _arm(
    shoulder: tuple[float, float] = (0.0, 10.0),
    elbow: tuple[float, float] = (0.0, 30.0),
    wrist: tuple[float, float] = (20.0, 30.0),
) -> dict:
    return {
        SHOULDER: _point(*shoulder),
        ELBOW: _point(*elbow),
        WRIST_POSE: _point(*wrist),
    }


def _hand(distance: float = 70.0) -> dict:
    return {
        THUMB_TIP: _point(0.0, 0.0),
        INDEX_TIP: _point(distance, 0.0),
    }


def _result(arm: dict | None = None, hand: dict | None = None) -> dict:
    return {
        "arm": _arm() if arm is None else arm,
        "hand": _hand() if hand is None else hand,
    }


class AngleCalculatorTests(unittest.TestCase):
    def test_compute_returns_typed_frozen_command_with_injected_clock(self) -> None:
        calculator = AngleCalculator(clock=lambda: 123.5)

        command = calculator.compute(_result())

        self.assertIsInstance(command, MimicCommand)
        self.assertEqual(command.sequence, 1)
        self.assertEqual(command.created_at, 123.5)
        self.assertEqual(command.arm, ArmCommand(shoulder_deg=180.0, elbow_deg=90.0))
        self.assertEqual(command.gripper, GripperCommand(aperture=0.5))
        self.assertIsNone(command.arm_error)
        self.assertIsNone(command.gripper_error)
        with self.assertRaises(FrozenInstanceError):
            command.sequence = 2

    def test_compute_always_returns_command_for_missing_result(self) -> None:
        command = AngleCalculator(clock=lambda: 1.0).compute(None)

        self.assertIsInstance(command, MimicCommand)
        self.assertIsNone(command.arm)
        self.assertIsNone(command.gripper)
        self.assertEqual(command.arm_error, "arm_missing")
        self.assertEqual(command.gripper_error, "gripper_hand_missing")

    def test_invalid_arm_clears_only_arm_history(self) -> None:
        calculator = AngleCalculator(clock=lambda: 1.0)
        calculator.compute(_result(hand=_hand(20.0)))
        invisible_arm = _arm()
        invisible_arm[ELBOW]["visibility"] = 0.0

        rejected = calculator.compute({"arm": invisible_arm, "hand": _hand(120.0)})
        recovered = calculator.compute(
            {
                "arm": _arm(elbow=(20.0, 10.0), wrist=(20.0, 30.0)),
                "hand": _hand(120.0),
            }
        )

        self.assertIsNone(rejected.arm)
        self.assertEqual(rejected.arm_error, "arm_not_visible")
        self.assertIsNotNone(rejected.gripper)
        # El historial de brazo se reinició: no se promedia 90° con los 180°
        # anteriores. El historial independiente del gripper sí se conserva.
        self.assertEqual(recovered.arm.shoulder_deg, 90.0)
        self.assertAlmostEqual(recovered.gripper.aperture, 0.667, places=3)

    def test_invalid_gripper_clears_only_gripper_history(self) -> None:
        calculator = AngleCalculator(clock=lambda: 1.0)
        calculator.compute(_result(hand=_hand(20.0)))
        bad_hand = _hand()
        bad_hand[INDEX_TIP]["x"] = math.inf

        rejected = calculator.compute({"arm": _arm(), "hand": bad_hand})
        recovered = calculator.compute(
            {
                "arm": _arm(elbow=(20.0, 10.0), wrist=(20.0, 30.0)),
                "hand": _hand(120.0),
            }
        )

        self.assertIsNotNone(rejected.arm)
        self.assertIsNone(rejected.gripper)
        self.assertEqual(rejected.gripper_error, "gripper_landmark_non_finite")
        # El brazo conserva sus dos muestras de 180° antes de promediar 90°.
        self.assertEqual(recovered.arm.shoulder_deg, 150.0)
        self.assertEqual(recovered.gripper.aperture, 1.0)

    def test_coincident_arm_landmarks_are_rejected_without_buffering(self) -> None:
        calculator = AngleCalculator(clock=lambda: 1.0)
        degenerate = _arm(elbow=(0.0, 10.0), wrist=(20.0, 10.0))

        rejected = calculator.compute({"arm": degenerate, "hand": _hand()})
        recovered = calculator.compute(
            {
                "arm": _arm(elbow=(20.0, 10.0), wrist=(20.0, 30.0)),
                "hand": _hand(),
            }
        )

        self.assertIsNone(rejected.arm)
        self.assertEqual(rejected.arm_error, "arm_segment_too_short")
        self.assertEqual(recovered.arm.shoulder_deg, 90.0)

    def test_non_finite_arm_coordinate_is_rejected(self) -> None:
        arm = _arm()
        arm[WRIST_POSE]["y"] = float("nan")

        command = AngleCalculator(clock=lambda: 1.0).compute(
            {"arm": arm, "hand": _hand()}
        )

        self.assertIsNone(command.arm)
        self.assertEqual(command.arm_error, "arm_landmark_non_finite")
        self.assertIsNotNone(command.gripper)

    def test_missing_hand_emits_no_new_gripper_target(self) -> None:
        calculator = AngleCalculator(clock=lambda: 1.0)
        first = calculator.compute({"arm": _arm(), "hand": _hand(120.0)})
        missing = calculator.compute({"arm": _arm(), "hand": None})

        self.assertEqual(first.gripper.aperture, 1.0)
        self.assertIsNone(missing.gripper)
        self.assertEqual(missing.gripper_error, "gripper_hand_missing")
        self.assertIsNotNone(missing.arm)

    def test_sequence_increments_across_invalid_samples(self) -> None:
        times = iter((10.0, 11.0, 12.0))
        calculator = AngleCalculator(clock=lambda: next(times))

        commands = [
            calculator.compute(_result()),
            calculator.compute(None),
            calculator.compute(_result()),
        ]

        self.assertEqual([command.sequence for command in commands], [1, 2, 3])
        self.assertEqual([command.created_at for command in commands], [10.0, 11.0, 12.0])


if __name__ == "__main__":
    unittest.main()
