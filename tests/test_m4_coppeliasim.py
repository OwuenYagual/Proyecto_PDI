"""Pruebas del mapeo seguro entre MediaPipe y el UR5."""

import math
import unittest

from modules.m4_coppeliasim import CoppeliaRobot


class FakeSim:
    handle_scene = 0

    def __init__(self) -> None:
        self.joint_commands: list[tuple] = []
        self.signal_commands: list[tuple] = []

    def setJointTargetPosition(self, *args) -> None:
        self.joint_commands.append(args)

    def setIntProperty(self, *args) -> None:
        self.signal_commands.append(args)


class CoppeliaRobotMappingTests(unittest.TestCase):
    def test_maps_validated_positions(self) -> None:
        shoulder, elbow = CoppeliaRobot.map_arm_angles(30.0, 120.0)

        self.assertEqual(shoulder, -10.0)
        self.assertEqual(elbow, -20.0)

    def test_scales_pose_within_validated_range(self) -> None:
        shoulder, elbow = CoppeliaRobot.map_arm_angles(150.0, 30.0)

        self.assertEqual(shoulder, -50.0)
        self.assertEqual(elbow, -50.0)

    def test_clamps_human_angles_outside_expected_range(self) -> None:
        shoulder, elbow = CoppeliaRobot.map_arm_angles(220.0, -20.0)

        self.assertEqual(shoulder, -60.0)
        self.assertEqual(elbow, -60.0)

    def test_maps_extended_arm_to_zero_elbow(self) -> None:
        _, elbow = CoppeliaRobot.map_arm_angles(0.0, 180.0)

        self.assertEqual(elbow, 0.0)

    def test_scales_observed_pose_without_immediate_saturation(self) -> None:
        shoulder, elbow = CoppeliaRobot.map_arm_angles(143.7, 26.9)

        self.assertAlmostEqual(shoulder, -47.9)
        self.assertAlmostEqual(elbow, -51.0333333333)

    def test_update_sends_arm_and_binary_gripper_targets(self) -> None:
        fake_sim = FakeSim()
        robot = CoppeliaRobot()
        robot._sim = fake_sim
        robot._joint_handles = list(range(6))
        robot._connected = True
        robot._update_interval = 0.0

        sent = robot.update({
            "shoulder": 30.0,
            "elbow": 120.0,
            "gripper": 0.75,
            "valid": True,
            "hand_detected": True,
        })

        self.assertTrue(sent)
        self.assertEqual(fake_sim.joint_commands[0][0], 1)
        self.assertAlmostEqual(
            fake_sim.joint_commands[0][1],
            math.radians(-10.0),
        )
        self.assertEqual(fake_sim.joint_commands[1][0], 2)
        self.assertAlmostEqual(
            fake_sim.joint_commands[1][1],
            math.radians(-20.0),
        )
        self.assertEqual(
            fake_sim.signal_commands,
            [(0, "signal.RG2_open", 1)],
        )


if __name__ == "__main__":
    unittest.main()
