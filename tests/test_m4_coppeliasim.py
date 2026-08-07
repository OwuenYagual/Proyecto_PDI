"""Pruebas deterministas del controlador seguro de CoppeliaSim."""

from __future__ import annotations

import math
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from config import settings
from modules.commands import ArmCommand, GripperCommand, MimicCommand
from modules.m4_coppeliasim import CoppeliaRobot, SafetyState


class FakeClock:
    def __init__(self, initial: float = 100.0) -> None:
        self._value = initial
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._value += seconds


class FakeSocket:
    def __init__(self, log: list[tuple]) -> None:
        self.log = log

    def setsockopt(self, option: int, value: int) -> None:
        self.log.append(("sockopt", option, value))


class FakeSim:
    handle_scene = 0
    handle_all = -1
    handle_tree = -2
    handle_single = -3
    simulation_stopped = 0
    simulation_paused = 1
    simulation_advancing_running = 2
    sceneobject_joint = 10
    sceneobject_shape = 11
    joint_revolute = 20
    jointmode_kinematic = 30
    jointmode_dynamic = 31
    jointdynctrl_position = 40
    jointintparam_dynctrlmode = 2039
    jointintparam_dynposctrltype = 2041

    UR5 = 100
    JOINTS = [101, 102, 103, 104, 105, 106]
    RG2 = 200
    BASE_SHAPE = 300
    ARM_SHAPES = [301, 302]
    RG2_SHAPE = 303
    ENVIRONMENT_SHAPE = 400
    VISION_SENSOR = 500

    def __init__(self) -> None:
        self.simulation_state = self.simulation_advancing_running
        self.positions = {handle: 0.0 for handle in self.JOINTS}
        self.intervals = {
            handle: (False, [-math.pi, 2.0 * math.pi])
            for handle in self.JOINTS
        }
        self.joint_modes = {
            handle: self.jointmode_kinematic for handle in self.JOINTS
        }
        self.dynamic_control_modes = {
            handle: self.jointdynctrl_position for handle in self.JOINTS
        }
        self.dynamic_profile_modes = {handle: 1 for handle in self.JOINTS}
        self.force_legacy_dynamic_properties = False
        self.aliases = {
            self.UR5: "/UR5",
            **{
                handle: f"/UR5/joint{index}"
                for index, handle in enumerate(self.JOINTS)
            },
            self.RG2: "/UR5/RG2",
            self.BASE_SHAPE: "/UR5/base",
            self.ARM_SHAPES[0]: "/UR5/link1",
            self.ARM_SHAPES[1]: "/UR5/link2",
            self.RG2_SHAPE: "/UR5/RG2/gripperShape",
            self.ENVIRONMENT_SHAPE: "/Obstacle",
        }
        self.collection_counter = 1000
        self.collection_items: list[tuple[int, int, int, int]] = []
        self.destroyed_collections: list[int] = []
        self.joint_commands: list[tuple[int, float, tuple[float, ...]]] = []
        self.signal_commands: list[tuple[int, str, int]] = []
        self.stop_calls = 0
        self.collision_response: tuple[int, list[int]] = (0, [])
        self.fail_next_collision = False
        self.collision_after_position_reads: int | None = None
        self.fail_alias_after_preflight = False
        self.alias_failed_after_preflight = False
        self.apply_joint_targets = True
        self.fail_stop = False
        self.block_stop = False
        self.vision_available = False
        self.vision_resolution = [2, 2]
        self.vision_image = bytes(range(12))
        self.vision_calls = 0
        self.fail_vision_lookup = False

        self.hold_event = threading.Event()
        self.signal_event = threading.Event()
        self.stop_event = threading.Event()
        self.stop_attempt_event = threading.Event()
        self.stop_block_entered = threading.Event()
        self.stop_block_release = threading.Event()
        self.second_shoulder_event = threading.Event()
        self.block_entered = threading.Event()
        self.block_release = threading.Event()
        self._block_next_target = False
        self._command_lock = threading.Lock()

    def block_next_target(self) -> None:
        self._block_next_target = True
        self.block_entered.clear()
        self.block_release.clear()

    def getSimulationState(self) -> int:
        return self.simulation_state

    def getObject(self, path: str, options: dict | None = None) -> int:
        if path == "/UR5":
            return self.UR5
        if path == "./joint" and options is not None:
            return self.JOINTS[options["index"]]
        if path in {"/UR5/RG2", "./RG2"}:
            return self.RG2
        if path == settings.COPPELIASIM_VIEW_SENSOR_PATH:
            if self.fail_vision_lookup:
                raise TimeoutError("fallo RPC al resolver Vision Sensor")
            if self.vision_available:
                return self.VISION_SENSOR
        if options is not None and options.get("noError"):
            return -1
        raise RuntimeError(f"objeto inexistente: {path}")

    def getVisionSensorImg(self, handle: int) -> tuple[bytes, list[int]]:
        if handle != self.VISION_SENSOR:
            raise RuntimeError("Vision Sensor invalido")
        self.vision_calls += 1
        return self.vision_image, self.vision_resolution

    def getObjectsInTree(self, base: int, object_type: int, options: int = 0) -> list[int]:
        del options
        if object_type == self.handle_all and base == self.UR5:
            return [self.UR5, *self.JOINTS, self.RG2]
        if object_type != self.sceneobject_shape:
            return []
        if base == self.handle_scene:
            return [
                self.BASE_SHAPE,
                *self.ARM_SHAPES,
                self.RG2_SHAPE,
                self.ENVIRONMENT_SHAPE,
            ]
        if base == self.UR5:
            return [self.BASE_SHAPE, *self.ARM_SHAPES, self.RG2_SHAPE]
        if base == self.JOINTS[0]:
            return [*self.ARM_SHAPES, self.RG2_SHAPE]
        if base == self.RG2:
            return [self.RG2_SHAPE]
        return []

    def getObjectAlias(self, handle: int, options: int = 0) -> str:
        del options
        if self.fail_alias_after_preflight:
            self.alias_failed_after_preflight = True
            raise TimeoutError("fallo RPC al consultar alias")
        return self.aliases.get(handle, str(handle))

    def getObjectType(self, handle: int) -> int:
        return self.sceneobject_joint if handle in self.JOINTS else 0

    def getJointType(self, handle: int) -> int:
        del handle
        return self.joint_revolute

    def getJointMode(self, handle: int) -> int:
        return self.joint_modes[handle]

    def getIntProperty(self, handle: int, name: str) -> int:
        if self.force_legacy_dynamic_properties:
            raise RuntimeError("property is unknown")
        if name == "dynCtrlMode":
            return self.dynamic_control_modes[handle]
        if name == "dynPosMode":
            return self.dynamic_profile_modes[handle]
        raise RuntimeError(name)

    def getObjectInt32Param(self, handle: int, parameter: int) -> int:
        if parameter == self.jointintparam_dynctrlmode:
            return self.dynamic_control_modes[handle]
        if parameter == self.jointintparam_dynposctrltype:
            return self.dynamic_profile_modes[handle]
        raise RuntimeError(parameter)

    def getJointInterval(self, handle: int) -> tuple[bool, list[float]]:
        return self.intervals[handle]

    def getJointPosition(self, handle: int) -> float:
        if self.collision_after_position_reads is not None:
            self.collision_after_position_reads -= 1
            if self.collision_after_position_reads == 0:
                self.collision_response = (
                    1,
                    [self.ARM_SHAPES[0], self.ENVIRONMENT_SHAPE],
                )
                self.collision_after_position_reads = None
        return self.positions[handle]

    def createCollection(self, options: int = 0) -> int:
        del options
        handle = self.collection_counter
        self.collection_counter += 1
        return handle

    def addItemToCollection(
        self,
        collection: int,
        what: int,
        object_handle: int,
        options: int,
    ) -> None:
        self.collection_items.append((collection, what, object_handle, options))

    def destroyCollection(self, handle: int) -> None:
        self.destroyed_collections.append(handle)

    def checkCollision(self, first: int, second: int) -> tuple[int, list[int]]:
        del first, second
        if self.fail_next_collision:
            self.fail_next_collision = False
            raise TimeoutError("fallo RPC en checkCollision")
        return self.collision_response

    def setJointTargetPosition(
        self,
        handle: int,
        target: float,
        motion_params: list[float],
    ) -> None:
        should_block = False
        with self._command_lock:
            if self._block_next_target:
                self._block_next_target = False
                should_block = True
        if should_block:
            self.block_entered.set()
            if not self.block_release.wait(2.0):
                raise TimeoutError("fake RPC bloqueada")

        with self._command_lock:
            self.joint_commands.append((handle, target, tuple(motion_params)))
            if self.apply_joint_targets:
                self.positions[handle] = target
            if handle == self.JOINTS[-1]:
                self.hold_event.set()
            shoulder_commands = sum(
                command[0] == self.JOINTS[1] for command in self.joint_commands
            )
            if shoulder_commands >= 2:
                self.second_shoulder_event.set()

    def setIntProperty(self, scene: int, name: str, value: int) -> None:
        self.signal_commands.append((scene, name, value))
        self.signal_event.set()

    def stopSimulation(self, wait: bool) -> None:
        self.stop_calls += 1
        self.stop_attempt_event.set()
        if self.block_stop:
            self.stop_block_entered.set()
            if not self.stop_block_release.wait(2.0):
                raise TimeoutError("stopSimulation bloqueado")
        if self.alias_failed_after_preflight:
            raise RuntimeError("socket REQ inválido después del alias")
        if self.fail_stop:
            raise TimeoutError("stopSimulation sin respuesta")
        self.simulation_state = self.simulation_stopped
        self.stop_event.set()
        self.last_stop_wait = wait


class FakeClient:
    def __init__(self, sim: FakeSim, log: list[tuple]) -> None:
        self.sim = sim
        self.log = log
        self.socket = FakeSocket(log)

    def require(self, name: str) -> FakeSim:
        self.log.append(("require", name))
        return self.sim


class CoppeliaRobotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.sim = FakeSim()
        self.client_log: list[tuple] = []
        self.robot = self.make_robot(self.sim)
        self.addCleanup(self.robot.close, False)

    def make_robot(self, sim: FakeSim) -> CoppeliaRobot:
        return CoppeliaRobot(
            client_factory=lambda **_: FakeClient(sim, self.client_log),
            clock=self.clock,
        )

    def command(
        self,
        sequence: int = 1,
        shoulder: float = 30.0,
        elbow: float = 120.0,
        aperture: float = 0.75,
    ) -> MimicCommand:
        return MimicCommand(
            sequence=sequence,
            created_at=self.clock(),
            arm=ArmCommand(shoulder, elbow),
            gripper=GripperCommand(aperture),
        )

    def connect_and_start(self, require_arm: bool = True) -> None:
        self.assertTrue(self.robot.connect())
        self.assertTrue(self.robot.start_imitation(require_arm=require_arm))

    def wait_for_state(self, state: SafetyState, timeout: float = 1.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.robot.snapshot.state is state:
                return True
            time.sleep(0.001)
        return self.robot.snapshot.state is state

    def test_map_arm_angles_preserves_expected_mapping(self) -> None:
        cases = (
            ((30.0, 120.0), (-10.0, -20.0)),
            ((150.0, 30.0), (-50.0, -50.0)),
            ((220.0, -20.0), (-60.0, -60.0)),
            ((0.0, 180.0), (0.0, 0.0)),
        )
        for inputs, expected in cases:
            with self.subTest(inputs=inputs):
                self.assertEqual(CoppeliaRobot.map_arm_angles(*inputs), expected)

    def test_sub_hz_frequencies_are_respected_without_clamping(self) -> None:
        with patch.object(settings, "COPPELIASIM_UPDATE_HZ", 0.5), patch.object(
            settings,
            "COPPELIASIM_COLLISION_CHECK_HZ",
            0.25,
        ):
            robot = self.make_robot(self.sim)
        self.addCleanup(robot.close, False)

        self.assertEqual(robot._update_interval, 2.0)
        self.assertEqual(robot._collision_interval, 4.0)

    def test_connect_runs_preflight_and_configures_socket_after_require(self) -> None:
        fake_zmq = SimpleNamespace(RCVTIMEO=1, SNDTIMEO=2, LINGER=3)
        with patch.dict(sys.modules, {"zmq": fake_zmq}):
            self.assertTrue(self.robot.connect())

        self.assertEqual(self.robot.snapshot.state, SafetyState.READY)
        self.assertTrue(self.robot.snapshot.connected)
        require_index = self.client_log.index(("require", "sim"))
        before_require = self.client_log[:require_index]
        after_require = self.client_log[require_index + 1 :]
        self.assertIn(
            ("sockopt", fake_zmq.RCVTIMEO, settings.COPPELIASIM_CONNECTION_TIMEOUT_MS),
            before_require,
        )
        self.assertIn(
            ("sockopt", fake_zmq.SNDTIMEO, settings.COPPELIASIM_CONNECTION_TIMEOUT_MS),
            before_require,
        )
        self.assertIn(
            ("sockopt", fake_zmq.RCVTIMEO, settings.COPPELIASIM_REQUEST_TIMEOUT_MS),
            after_require,
        )
        self.assertIn(
            ("sockopt", fake_zmq.SNDTIMEO, settings.COPPELIASIM_REQUEST_TIMEOUT_MS),
            after_require,
        )
        self.assertEqual(
            self.sim.collection_items,
            [
                (1000, self.sim.handle_single, self.sim.ARM_SHAPES[0], 0),
                (1000, self.sim.handle_single, self.sim.ARM_SHAPES[1], 0),
                (1000, self.sim.handle_single, self.sim.RG2_SHAPE, 0),
                (
                    1001,
                    self.sim.handle_single,
                    self.sim.ENVIRONMENT_SHAPE,
                    0,
                ),
            ],
        )

    def test_missing_vision_sensor_does_not_block_safe_connection(self) -> None:
        self.assertTrue(self.robot.connect())

        self.assertEqual(self.robot.snapshot.state, SafetyState.READY)
        self.assertIsNone(self.robot.simulator_frame.rgb)
        self.assertIn("Vision Sensor", self.robot.simulator_frame.message)

    def test_vision_sensor_transport_failure_fails_preflight(self) -> None:
        self.sim.fail_vision_lookup = True

        self.assertFalse(self.robot.connect())

        self.assertEqual(self.robot.snapshot.state, SafetyState.DISCONNECTED)
        self.assertIn("Vision Sensor", self.robot.snapshot.message)

    def test_vision_frame_is_rate_limited_validated_and_flipped(self) -> None:
        self.sim.vision_available = True
        with patch.object(settings, "COPPELIASIM_VIEW_WIDTH", 2), patch.object(
            settings,
            "COPPELIASIM_VIEW_HEIGHT",
            2,
        ):
            robot = self.make_robot(self.sim)
        self.addCleanup(robot.close, False)

        self.assertTrue(robot.connect())
        deadline = time.monotonic() + 1.0
        while robot.simulator_frame.sequence == 0 and time.monotonic() < deadline:
            time.sleep(0.001)

        frame = robot.simulator_frame
        self.assertEqual((frame.width, frame.height), (2, 2))
        self.assertEqual(frame.rgb, bytes(range(6, 12)) + bytes(range(6)))
        self.assertEqual(self.sim.vision_calls, 1)

        robot._wake_event.set()
        time.sleep(0.02)
        self.assertEqual(self.sim.vision_calls, 1)

    def test_invalid_vision_buffer_keeps_controller_ready(self) -> None:
        self.sim.vision_available = True
        self.sim.vision_image = b"short"
        with patch.object(settings, "COPPELIASIM_VIEW_WIDTH", 2), patch.object(
            settings,
            "COPPELIASIM_VIEW_HEIGHT",
            2,
        ):
            robot = self.make_robot(self.sim)
        self.addCleanup(robot.close, False)

        self.assertTrue(robot.connect())
        deadline = time.monotonic() + 1.0
        while "tamano" not in robot.simulator_frame.message and time.monotonic() < deadline:
            time.sleep(0.001)

        self.assertEqual(robot.snapshot.state, SafetyState.READY)
        self.assertIsNone(robot.simulator_frame.rgb)
        self.assertIn("tamano", robot.simulator_frame.message)

    def test_preflight_rejects_empty_effective_interval(self) -> None:
        self.sim.intervals[self.sim.JOINTS[1]] = (
            False,
            [math.radians(10.0), math.radians(20.0)],
        )

        self.assertFalse(self.robot.connect())

        self.assertEqual(self.robot.snapshot.state, SafetyState.DISCONNECTED)
        self.assertIn("no intersecta", self.robot.snapshot.message)

    def test_preflight_rejects_confirmed_incompatible_mode(self) -> None:
        self.sim.joint_modes[self.sim.JOINTS[2]] = 999

        self.assertFalse(self.robot.connect())

        self.assertIn("cinemático/dinámico", self.robot.snapshot.message)

    def test_preflight_fails_closed_when_joint_query_is_missing(self) -> None:
        self.sim.getJointInterval = None  # type: ignore[method-assign]

        self.assertFalse(self.robot.connect())

        self.assertIn("intervalos", self.robot.snapshot.message)

    def test_dynamic_joint_requires_position_motion_profile(self) -> None:
        handle = self.sim.JOINTS[1]
        self.sim.joint_modes[handle] = self.sim.jointmode_dynamic
        self.sim.dynamic_profile_modes[handle] = 0

        self.assertFalse(self.robot.connect())

        self.assertIn("perfil de movimiento", self.robot.snapshot.message)

    def test_dynamic_joint_uses_documented_legacy_parameter_fallback(self) -> None:
        handle = self.sim.JOINTS[1]
        self.sim.joint_modes[handle] = self.sim.jointmode_dynamic
        self.sim.force_legacy_dynamic_properties = True

        self.assertTrue(self.robot.connect())

        self.assertEqual(self.robot.snapshot.state, SafetyState.READY)

    def test_preflight_rejects_joint_inside_rg2_subtree(self) -> None:
        self.sim.aliases[self.sim.JOINTS[-1]] = "/UR5/RG2/extraJoint"

        self.assertFalse(self.robot.connect())

        self.assertIn("subárbol RG2", self.robot.snapshot.message)

    def test_submit_validates_types_finiteness_ranges_and_age(self) -> None:
        self.connect_and_start()
        cases = (
            (
                MimicCommand(1, self.clock(), ArmCommand(True, 90.0)),
                "no finitos",
            ),
            (
                MimicCommand(2, self.clock(), ArmCommand(math.nan, 90.0)),
                "no finitos",
            ),
            (
                MimicCommand(3, self.clock(), ArmCommand(181.0, 90.0)),
                "fuera",
            ),
            (
                MimicCommand(4, self.clock(), gripper=GripperCommand(math.inf)),
                "no finita",
            ),
        )
        for command, reason_fragment in cases:
            with self.subTest(command=command):
                report = self.robot.submit(command)
                reasons = (report.arm.reason or "") + (report.gripper.reason or "")
                self.assertIn(reason_fragment, reasons)

        self.clock.advance(1.0)
        stale = MimicCommand(5, self.clock() - 1.0, ArmCommand(30.0, 120.0))
        report = self.robot.submit(stale)
        self.assertFalse(report.arm.accepted)
        self.assertFalse(report.gripper.accepted)
        self.assertIn("caducado", report.arm.reason or "")

    def test_arm_batch_and_gripper_are_validated_independently(self) -> None:
        self.connect_and_start()
        command = MimicCommand(
            sequence=1,
            created_at=self.clock(),
            arm=ArmCommand(200.0, 120.0),
            gripper=GripperCommand(0.9),
        )

        report = self.robot.submit(command)

        self.assertFalse(report.arm.accepted)
        self.assertTrue(report.gripper.accepted)
        self.assertTrue(self.sim.signal_event.wait(1.0))
        self.assertEqual(
            self.sim.signal_commands,
            [(self.sim.handle_scene, CoppeliaRobot.GRIPPER_SIGNAL, 1)],
        )
        self.assertFalse(
            any(
                handle in self.sim.JOINTS[1:3] and target < -1e-6
                for handle, target, _ in self.sim.joint_commands
            )
        )

    def test_pause_requests_hold_without_rpc_on_caller(self) -> None:
        self.connect_and_start()

        self.assertTrue(self.robot.pause())

        self.assertEqual(self.robot.snapshot.state, SafetyState.PAUSED)
        self.assertTrue(self.sim.hold_event.wait(1.0))
        self.assertEqual(len(self.sim.joint_commands), 6)

    def test_emergency_stop_is_nonblocking_while_rpc_is_blocked(self) -> None:
        self.connect_and_start()
        self.sim.block_next_target()
        report = self.robot.submit(self.command())
        self.assertTrue(report.arm.accepted)
        self.assertTrue(self.sim.block_entered.wait(1.0))

        self.robot.emergency_stop("tecla E")

        self.assertEqual(self.robot.snapshot.state, SafetyState.ESTOP)
        self.assertFalse(self.sim.stop_event.is_set())
        self.sim.block_release.set()
        self.assertTrue(self.sim.stop_event.wait(1.0))
        self.assertFalse(self.sim.last_stop_wait)
        self.assertNotIn(
            (self.sim.JOINTS[2], math.radians(-20.0)),
            [(handle, target) for handle, target, _ in self.sim.joint_commands],
        )
        self.robot.emergency_stop("segunda pulsación")
        self.assertEqual(self.sim.stop_calls, 1)

    def test_pause_during_blocked_rpc_cancels_remaining_arm_batch(self) -> None:
        self.connect_and_start()
        self.sim.block_next_target()
        self.robot.submit(self.command())
        self.assertTrue(self.sim.block_entered.wait(1.0))

        self.assertTrue(self.robot.pause())
        self.sim.block_release.set()

        self.assertTrue(self.sim.hold_event.wait(1.0))
        elbow_targets = [
            target
            for handle, target, _ in self.sim.joint_commands
            if handle == self.sim.JOINTS[2]
        ]
        self.assertNotIn(math.radians(-20.0), elbow_targets)
        self.assertEqual(self.robot.snapshot.state, SafetyState.PAUSED)

    def test_invalid_arm_cancels_an_inflight_older_arm_batch(self) -> None:
        self.connect_and_start()
        self.sim.block_next_target()
        self.robot.submit(self.command())
        self.assertTrue(self.sim.block_entered.wait(1.0))

        invalid = MimicCommand(
            sequence=2,
            created_at=self.clock(),
            arm_error="pose perdida",
        )
        self.assertFalse(self.robot.submit(invalid).arm.accepted)
        self.sim.block_release.set()

        self.assertTrue(self.sim.hold_event.wait(1.0))
        elbow_targets = [
            target
            for handle, target, _ in self.sim.joint_commands
            if handle == self.sim.JOINTS[2]
        ]
        self.assertNotIn(math.radians(-20.0), elbow_targets)

    def test_fully_invalid_sample_removes_an_older_pending_command(self) -> None:
        self.connect_and_start()
        with self.robot._lock:
            self.robot._latest_command = self.command()

        invalid = MimicCommand(
            sequence=2,
            created_at=self.clock(),
            arm_error="pose perdida",
            gripper_error="mano perdida",
        )
        report = self.robot.submit(invalid)

        self.assertFalse(report.arm.accepted)
        self.assertFalse(report.gripper.accepted)
        with self.robot._lock:
            self.assertIsNone(self.robot._latest_command)

    def test_latest_only_queue_discards_intermediate_command(self) -> None:
        self.connect_and_start()
        self.sim.block_next_target()
        self.robot.submit(self.command(sequence=1, shoulder=30.0))
        self.assertTrue(self.sim.block_entered.wait(1.0))

        self.clock.advance(0.1)
        self.robot.submit(self.command(sequence=2, shoulder=60.0))
        self.robot.submit(self.command(sequence=3, shoulder=120.0))
        self.sim.block_release.set()

        self.assertTrue(self.sim.second_shoulder_event.wait(1.0))
        shoulder_targets = [
            target
            for handle, target, _ in self.sim.joint_commands
            if handle == self.sim.JOINTS[1]
        ]
        self.assertIn(math.radians(-10.0), shoulder_targets)
        self.assertIn(math.radians(-40.0), shoulder_targets)
        self.assertNotIn(math.radians(-20.0), shoulder_targets)

    def test_collision_latches_estop_and_records_aliases(self) -> None:
        self.connect_and_start()
        self.sim.collision_response = (
            1,
            [self.sim.ARM_SHAPES[0], self.sim.ENVIRONMENT_SHAPE],
        )

        self.robot.submit(self.command())

        self.assertTrue(self.sim.stop_event.wait(1.0))
        snapshot = self.robot.snapshot
        self.assertEqual(snapshot.state, SafetyState.ESTOP)
        self.assertEqual(snapshot.collision_pair, ("/UR5/link1", "/Obstacle"))
        self.assertEqual(self.sim.stop_calls, 1)

    def test_collision_stops_before_any_optional_alias_rpc(self) -> None:
        self.connect_and_start()
        self.sim.fail_alias_after_preflight = True
        self.sim.collision_response = (
            1,
            [self.sim.ARM_SHAPES[0], self.sim.ENVIRONMENT_SHAPE],
        )

        self.robot.submit(self.command())

        self.assertTrue(self.sim.stop_event.wait(1.0))
        self.assertFalse(self.sim.alias_failed_after_preflight)
        self.assertEqual(self.robot.snapshot.state, SafetyState.ESTOP)
        self.assertEqual(
            self.robot.snapshot.collision_pair,
            ("/UR5/link1", "/Obstacle"),
        )
        self.assertEqual(self.sim.stop_calls, 1)

    def test_collision_is_checked_immediately_before_command_actuation(self) -> None:
        self.connect_and_start()
        self.robot._last_collision_check = self.clock() + 1_000.0
        self.sim.collision_response = (
            1,
            [self.sim.ARM_SHAPES[0], self.sim.ENVIRONMENT_SHAPE],
        )

        self.robot.submit(self.command())

        self.assertTrue(self.sim.stop_event.wait(1.0))
        self.assertEqual(self.robot.snapshot.state, SafetyState.ESTOP)
        self.assertEqual(self.sim.joint_commands, [])
        self.assertEqual(self.sim.signal_commands, [])

    def test_runtime_rpc_failure_enters_fault_and_stops_once(self) -> None:
        self.connect_and_start()
        self.sim.fail_next_collision = True
        self.clock.advance(0.1)

        self.assertTrue(self.sim.stop_event.wait(1.0))

        self.assertEqual(self.robot.snapshot.state, SafetyState.FAULT)
        self.assertIn("Fallo RPC", self.robot.snapshot.message)
        self.assertEqual(self.sim.stop_calls, 1)

    def test_pose_watchdog_holds_then_estops(self) -> None:
        self.connect_and_start()
        self.clock.advance(0.3)

        report = self.robot.submit(
            MimicCommand(
                sequence=1,
                created_at=self.clock(),
                gripper=GripperCommand(0.5),
                arm_error="pose perdida",
            )
        )

        self.assertFalse(report.arm.accepted)
        self.assertTrue(report.gripper.accepted)
        self.assertTrue(self.sim.hold_event.wait(1.0))
        self.clock.advance(0.5)
        self.robot.submit(
            MimicCommand(
                sequence=2,
                created_at=self.clock(),
                gripper=GripperCommand(0.5),
                arm_error="pose perdida",
            )
        )
        self.assertTrue(self.sim.stop_event.wait(1.0))
        self.assertEqual(self.robot.snapshot.state, SafetyState.ESTOP)
        self.assertIn("Watchdog", self.robot.snapshot.message)

    def test_repeated_invalid_samples_do_not_repeat_hold(self) -> None:
        self.connect_and_start()
        first = MimicCommand(
            sequence=1,
            created_at=self.clock(),
            gripper=GripperCommand(0.2),
            arm_error="pose perdida",
        )
        self.robot.submit(first)
        self.assertTrue(self.sim.hold_event.wait(1.0))
        self.assertTrue(self.sim.signal_event.wait(1.0))
        hold_command_count = len(self.sim.joint_commands)

        self.sim.signal_event.clear()
        self.clock.advance(0.1)
        second = MimicCommand(
            sequence=2,
            created_at=self.clock(),
            gripper=GripperCommand(0.8),
            arm_error="pose aún perdida",
        )
        self.robot.submit(second)

        self.assertTrue(self.sim.signal_event.wait(1.0))
        self.assertEqual(len(self.sim.joint_commands), hold_command_count)

    def test_gripper_only_mode_disables_arm_watchdog(self) -> None:
        self.connect_and_start(require_arm=False)
        self.clock.advance(10.0)

        report = self.robot.submit(
            MimicCommand(
                sequence=1,
                created_at=self.clock(),
                gripper=GripperCommand(0.2),
            )
        )

        self.assertTrue(report.gripper.accepted)
        self.assertTrue(self.sim.signal_event.wait(1.0))
        self.assertEqual(self.robot.snapshot.state, SafetyState.RUNNING)
        self.assertEqual(self.sim.stop_calls, 0)

    def test_estop_close_never_returns_home(self) -> None:
        self.connect_and_start()
        self.robot.emergency_stop("prueba")
        self.assertTrue(self.sim.stop_event.wait(1.0))
        command_count = len(self.sim.joint_commands)

        self.robot.close(normal_exit=True)

        self.assertEqual(len(self.sim.joint_commands), command_count)
        self.assertEqual(self.sim.stop_calls, 1)
        self.assertCountEqual(self.sim.destroyed_collections, [1000, 1001])

    def test_immediate_close_still_services_pending_estop(self) -> None:
        self.connect_and_start()

        self.robot.emergency_stop("prueba")
        self.robot.close(normal_exit=False)

        self.assertEqual(self.sim.stop_calls, 1)

    def test_failed_stop_cannot_be_masked_by_estop_or_reconnect(self) -> None:
        self.connect_and_start()
        self.sim.fail_stop = True

        self.robot.emergency_stop("primera parada")

        self.assertTrue(self.sim.stop_attempt_event.wait(1.0))
        self.assertTrue(self.wait_for_state(SafetyState.FAULT))
        original_message = self.robot.snapshot.message
        self.assertIn("no pudo detener", original_message)

        self.robot.emergency_stop("segunda parada")

        self.assertEqual(self.robot.snapshot.state, SafetyState.FAULT)
        self.assertEqual(self.robot.snapshot.message, original_message)
        self.assertEqual(self.sim.stop_calls, 1)
        self.assertFalse(self.robot.reconnect())
        self.assertEqual(self.robot.snapshot.state, SafetyState.FAULT)
        self.robot.close(normal_exit=False)
        self.assertFalse(self.robot.connect())

    def test_reconnect_is_rejected_while_stop_is_unconfirmed(self) -> None:
        self.connect_and_start()
        self.sim.block_stop = True
        self.sim.fail_stop = True
        self.robot.emergency_stop("parada bloqueada")
        self.assertTrue(self.sim.stop_block_entered.wait(1.0))
        reconnect_result: list[bool] = []
        reconnect_thread = threading.Thread(
            target=lambda: reconnect_result.append(self.robot.reconnect())
        )

        reconnect_thread.start()
        reconnect_thread.join(0.1)

        self.assertFalse(reconnect_thread.is_alive())
        self.assertEqual(reconnect_result, [False])
        self.sim.stop_block_release.set()
        self.assertTrue(self.wait_for_state(SafetyState.FAULT))
        self.assertFalse(self.robot.reconnect())
        self.assertEqual(self.sim.stop_calls, 1)

    def test_estop_forbids_home_even_after_successful_reconnect(self) -> None:
        self.connect_and_start()
        self.robot.emergency_stop("prueba")
        self.assertTrue(self.sim.stop_event.wait(1.0))
        self.sim.simulation_state = self.sim.simulation_advancing_running
        self.assertTrue(self.robot.reconnect())
        command_count = len(self.sim.joint_commands)

        self.robot.close(normal_exit=True)

        self.assertEqual(len(self.sim.joint_commands), command_count)

    def test_normal_close_does_not_send_home_if_simulation_was_stopped(self) -> None:
        self.assertTrue(self.robot.connect())
        self.sim.simulation_state = self.sim.simulation_stopped

        self.robot.close(normal_exit=True)

        self.assertEqual(self.sim.joint_commands, [])

    def test_normal_close_returns_home_and_verifies_readback(self) -> None:
        self.connect_and_start()
        self.robot.submit(self.command())
        self.assertTrue(self.sim.signal_event.wait(1.0))

        self.robot.close(normal_exit=True)

        home_commands = self.sim.joint_commands[-6:]
        self.assertEqual(
            [(handle, target) for handle, target, _ in home_commands],
            [(handle, 0.0) for handle in self.sim.JOINTS],
        )
        self.assertEqual(self.robot.snapshot.state, SafetyState.DISCONNECTED)

    def test_home_checks_collision_after_the_final_readback(self) -> None:
        self.connect_and_start()
        self.sim.collision_after_position_reads = len(self.sim.JOINTS)

        self.robot.close(normal_exit=True)

        self.assertEqual(self.robot.snapshot.state, SafetyState.ESTOP)
        self.assertEqual(
            self.robot.snapshot.collision_pair,
            ("/UR5/link1", "/Obstacle"),
        )
        self.assertFalse(self.robot._home_succeeded)
        self.assertEqual(self.sim.stop_calls, 1)

    def test_home_readback_timeout_enters_fault_and_stops(self) -> None:
        self.connect_and_start()
        for handle in self.sim.JOINTS:
            self.sim.positions[handle] = 0.5
        self.sim.apply_joint_targets = False
        self.robot._home_timeout = 0.05

        self.robot.close(normal_exit=True)

        self.assertEqual(self.robot.snapshot.state, SafetyState.FAULT)
        self.assertFalse(self.robot.snapshot.connected)
        self.assertIn("Timeout esperando readback de home", self.robot.snapshot.message)
        self.assertEqual(self.sim.stop_calls, 1)

    def test_reconnect_requires_running_simulation_and_returns_ready(self) -> None:
        self.connect_and_start()
        self.robot.emergency_stop("prueba")
        self.assertTrue(self.sim.stop_event.wait(1.0))

        self.assertFalse(self.robot.reconnect())
        self.sim.simulation_state = self.sim.simulation_advancing_running
        self.assertTrue(self.robot.reconnect())

        self.assertEqual(self.robot.snapshot.state, SafetyState.READY)

    def test_reconnect_is_rejected_while_running(self) -> None:
        self.connect_and_start()

        self.assertFalse(self.robot.reconnect())

        self.assertEqual(self.robot.snapshot.state, SafetyState.RUNNING)


if __name__ == "__main__":
    unittest.main()
