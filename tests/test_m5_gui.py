"""Pruebas headless del runtime y la presentacion de M5."""

from __future__ import annotations

import sys
import threading
import time
from types import ModuleType
import unittest

import numpy as np


try:
    import cv2 as _cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = ModuleType("cv2")

try:
    import mediapipe as _mediapipe  # noqa: F401
except ModuleNotFoundError:
    sys.modules["mediapipe"] = ModuleType("mediapipe")

from modules.commands import ArmCommand, GripperCommand, MimicCommand
from modules.m4_coppeliasim import SafetySnapshot, SafetyState, SimulatorFrameSnapshot
from modules.m5_gui import (
    CameraFrameSnapshot,
    MimicRuntime,
    RobotMimicApp,
    RuntimeSnapshot,
    metric_values,
    presentation_for,
)


def safety(state: SafetyState, message: str = "estado") -> SafetySnapshot:
    return SafetySnapshot(
        state=state,
        connected=state not in {SafetyState.DISCONNECTED, SafetyState.FAULT},
        message=message,
        collision_pair=None,
    )


EMPTY_SIMULATOR = SimulatorFrameSnapshot(None, 0, 0, 0, None, "sin vista")
EMPTY_CAMERA = CameraFrameSnapshot(None, 0, 0, 0)


def runtime_snapshot(
    state: SafetyState,
    *,
    camera_available: bool = True,
    camera_fatal: bool = False,
    command: MimicCommand | None = None,
) -> RuntimeSnapshot:
    return RuntimeSnapshot(
        safety=safety(state),
        simulator=EMPTY_SIMULATOR,
        camera=EMPTY_CAMERA,
        camera_available=camera_available,
        camera_fatal=camera_fatal,
        fps=29.5,
        command=command,
        report=None,
        message="estado",
    )


class PresentationTests(unittest.TestCase):
    def test_running_state_offers_pause_and_disables_reconnect(self) -> None:
        presentation = presentation_for(runtime_snapshot(SafetyState.RUNNING))

        self.assertEqual(presentation.toggle_text, "Pausar")
        self.assertTrue(presentation.toggle_enabled)
        self.assertFalse(presentation.reconnect_enabled)

    def test_ready_requires_a_healthy_camera_to_start(self) -> None:
        missing = presentation_for(
            runtime_snapshot(SafetyState.READY, camera_available=False)
        )
        fatal = presentation_for(
            runtime_snapshot(SafetyState.READY, camera_fatal=True)
        )

        self.assertFalse(missing.toggle_enabled)
        self.assertFalse(fatal.toggle_enabled)
        self.assertFalse(fatal.reconnect_enabled)

    def test_fault_allows_reconnect_but_not_start(self) -> None:
        presentation = presentation_for(runtime_snapshot(SafetyState.FAULT))

        self.assertFalse(presentation.toggle_enabled)
        self.assertTrue(presentation.reconnect_enabled)

    def test_metrics_are_derived_from_the_latest_command(self) -> None:
        command = MimicCommand(
            1,
            1.0,
            arm=ArmCommand(45.25, 110.75),
            gripper=GripperCommand(0.61),
        )

        self.assertEqual(
            metric_values(runtime_snapshot(SafetyState.RUNNING, command=command)),
            ("45.2°", "110.8°", "61%"),
        )


class FakeRobot:
    def __init__(self) -> None:
        self.snapshot = safety(SafetyState.DISCONNECTED)
        self.simulator_frame = EMPTY_SIMULATOR
        self.connect_event = threading.Event()
        self.estop_reasons: list[str] = []
        self.pause_calls = 0
        self.start_calls = 0
        self.reconnect_calls = 0
        self.close_calls: list[bool] = []

    def connect(self) -> bool:
        self.snapshot = safety(SafetyState.READY)
        self.connect_event.set()
        return True

    def start_imitation(self, require_arm: bool = True) -> bool:
        self.start_calls += 1
        if not require_arm or self.snapshot.state not in {
            SafetyState.READY,
            SafetyState.PAUSED,
        }:
            return False
        self.snapshot = safety(SafetyState.RUNNING)
        return True

    def pause(self) -> bool:
        self.pause_calls += 1
        if self.snapshot.state is not SafetyState.RUNNING:
            return False
        self.snapshot = safety(SafetyState.PAUSED)
        return True

    def emergency_stop(self, reason: str) -> None:
        self.estop_reasons.append(reason)
        self.snapshot = safety(SafetyState.ESTOP, reason)

    def reconnect(self) -> bool:
        self.reconnect_calls += 1
        self.snapshot = safety(SafetyState.READY)
        return True

    def submit(self, command):
        del command
        return None

    def close(self, normal_exit: bool = True) -> None:
        self.close_calls.append(normal_exit)


class FailingCapture:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return False, None, None


class OneFrameCapture:
    def __init__(self) -> None:
        self.frame = np.zeros((2, 3, 3), dtype=np.uint8)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        time.sleep(0.002)
        return True, self.frame.copy(), self.frame.copy()


class FakeDetector:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def process(self, _frame, process_hands: bool = True):
        del process_hands
        return None

    def draw_skeleton(self, frame, _result):
        return frame


class FakeCalculator:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset_buffers(self) -> None:
        self.reset_calls += 1

    def compute(self, _result):
        return MimicCommand(1, time.monotonic())


class RuntimeTests(unittest.TestCase):
    def make_runtime(self, robot: FakeRobot, capture_factory) -> MimicRuntime:
        return MimicRuntime(
            robot_factory=lambda: robot,
            capture_factory=capture_factory,
            detector_factory=FakeDetector,
            calculator_factory=FakeCalculator,
        )

    def test_runtime_publishes_camera_frames_while_connection_is_backgrounded(self) -> None:
        robot = FakeRobot()
        runtime = self.make_runtime(robot, OneFrameCapture)
        runtime.start()
        self.addCleanup(runtime.request_close)

        deadline = time.monotonic() + 1.0
        while runtime.snapshot.camera.rgb is None and time.monotonic() < deadline:
            time.sleep(0.002)

        self.assertTrue(robot.connect_event.is_set())
        self.assertTrue(runtime.snapshot.camera_available)
        self.assertEqual(
            (runtime.snapshot.camera.width, runtime.snapshot.camera.height),
            (3, 2),
        )
        runtime.request_close()
        self.assertTrue(runtime.wait(1.0))
        self.assertEqual(robot.close_calls, [True])

    def test_three_camera_failures_latch_estop_and_skip_home(self) -> None:
        robot = FakeRobot()
        runtime = self.make_runtime(robot, FailingCapture)
        runtime.start()

        deadline = time.monotonic() + 1.0
        while not runtime.snapshot.camera_fatal and time.monotonic() < deadline:
            time.sleep(0.002)

        self.assertTrue(runtime.snapshot.camera_fatal)
        self.assertEqual(robot.snapshot.state, SafetyState.ESTOP)
        self.assertTrue(robot.estop_reasons)
        self.assertFalse(runtime.reconnect())
        runtime.request_close()
        self.assertTrue(runtime.wait(1.0))
        self.assertEqual(robot.close_calls, [False])

    def test_emergency_action_is_published_without_waiting_for_worker(self) -> None:
        robot = FakeRobot()
        robot.snapshot = safety(SafetyState.RUNNING)
        runtime = self.make_runtime(robot, OneFrameCapture)

        runtime.emergency_stop("boton rojo")

        self.assertEqual(robot.estop_reasons, ["boton rojo"])
        self.assertEqual(robot.snapshot.state, SafetyState.ESTOP)
        self.assertEqual(runtime.snapshot.safety.state, SafetyState.ESTOP)


class ActionEquivalenceTests(unittest.TestCase):
    def test_visible_actions_delegate_to_the_runtime(self) -> None:
        calls: list[str] = []
        runtime = type(
            "RuntimeDouble",
            (),
            {
                "toggle_imitation": lambda _self: calls.append("toggle"),
                "emergency_stop": lambda _self: calls.append("estop"),
                "reconnect": lambda _self: calls.append("reconnect"),
                "request_close": lambda _self: calls.append("close"),
            },
        )()
        app = RobotMimicApp.__new__(RobotMimicApp)
        app.runtime = runtime
        app._closing = False

        app._on_toggle()
        app._on_estop()
        app._on_reconnect()
        app._on_close()

        self.assertEqual(calls, ["toggle", "estop", "reconnect", "close"])


if __name__ == "__main__":
    unittest.main()
