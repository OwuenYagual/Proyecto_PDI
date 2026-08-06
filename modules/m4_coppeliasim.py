"""Módulo M4: control cinemático del UR5 y el gripper RG2."""

from __future__ import annotations

import math
import time
from typing import Any

from config.settings import (
    COPPELIASIM_ENABLED,
    COPPELIASIM_HOST,
    COPPELIASIM_PORT,
    COPPELIASIM_CONNECTION_TIMEOUT_MS,
    COPPELIASIM_UPDATE_HZ,
    COPPELIASIM_JOINT_COUNT,
    COPPELIASIM_SHOULDER_INDEX,
    COPPELIASIM_ELBOW_INDEX,
    COPPELIASIM_SHOULDER_MIN_DEG,
    COPPELIASIM_SHOULDER_MAX_DEG,
    COPPELIASIM_ELBOW_MIN_DEG,
    COPPELIASIM_ELBOW_MAX_DEG,
    COPPELIASIM_GRIPPER_THRESHOLD,
    COPPELIASIM_MAX_VELOCITY_DEG,
    COPPELIASIM_MAX_ACCELERATION_DEG,
    COPPELIASIM_MAX_JERK_DEG,
)


class CoppeliaRobot:
    """Adapta las mediciones de M3 a las articulaciones del UR5."""

    GRIPPER_SIGNAL = "signal.RG2_open"

    def __init__(self) -> None:
        self._client: Any | None = None
        self._sim: Any | None = None
        self._joint_handles: list[int] = []
        self._home_positions: list[float] = []
        self._connected = False
        self._last_update = 0.0
        self._last_gripper_command: int | None = None
        self._update_interval = 1.0 / max(COPPELIASIM_UPDATE_HZ, 1.0)
        self._motion_params = [
            math.radians(COPPELIASIM_MAX_VELOCITY_DEG),
            math.radians(COPPELIASIM_MAX_ACCELERATION_DEG),
            math.radians(COPPELIASIM_MAX_JERK_DEG),
        ]

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Conecta con la escena activa y localiza los seis joints del UR5."""
        if not COPPELIASIM_ENABLED:
            print("M4 CoppeliaSim deshabilitado en settings.py.")
            return False

        try:
            import zmq
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient

            self._client = RemoteAPIClient(
                host=COPPELIASIM_HOST,
                port=COPPELIASIM_PORT,
            )
            self._client.socket.setsockopt(
                zmq.RCVTIMEO,
                COPPELIASIM_CONNECTION_TIMEOUT_MS,
            )
            self._client.socket.setsockopt(
                zmq.SNDTIMEO,
                COPPELIASIM_CONNECTION_TIMEOUT_MS,
            )
            self._client.socket.setsockopt(zmq.LINGER, 0)
            self._sim = self._client.require("sim")

            if self._sim.getSimulationState() == self._sim.simulation_stopped:
                print(
                    "M4 sin activar: inicia la simulacion en CoppeliaSim "
                    "antes de ejecutar main.py."
                )
                self._reset_connection()
                return False

            ur5_handle = self._sim.getObject("/UR5")
            self._joint_handles = [
                self._sim.getObject(
                    "./joint",
                    {"proxy": ur5_handle, "index": index},
                )
                for index in range(COPPELIASIM_JOINT_COUNT)
            ]
            self._validate_arm_joints()
            self._home_positions = [
                self._sim.getJointPosition(handle)
                for handle in self._joint_handles
            ]
        except Exception as exc:
            print(f"M4 sin conexion con CoppeliaSim: {exc}")
            print("La camara continuara funcionando sin el robot virtual.")
            self._reset_connection()
            return False

        self._connected = True
        self._last_update = 0.0
        print("M4 conectado: UR5 y RG2 listos.")
        return True

    def update(self, angles: dict) -> bool:
        """Envía hombro, codo y apertura binaria sin bloquear cada frame."""
        if not self._connected or self._sim is None:
            return False
        if not angles.get("valid", False):
            return False

        now = time.monotonic()
        if now - self._last_update < self._update_interval:
            return False

        shoulder_deg, elbow_deg = self.map_arm_angles(
            shoulder_deg=float(angles["shoulder"]),
            elbow_deg=float(angles["elbow"]),
        )

        try:
            self._set_joint_target(
                COPPELIASIM_SHOULDER_INDEX,
                math.radians(shoulder_deg),
            )
            self._set_joint_target(
                COPPELIASIM_ELBOW_INDEX,
                math.radians(elbow_deg),
            )

            if angles.get("hand_detected", False):
                gripper_command = int(
                    float(angles["gripper"])
                    >= COPPELIASIM_GRIPPER_THRESHOLD
                )
                if gripper_command != self._last_gripper_command:
                    self._sim.setIntProperty(
                        self._sim.handle_scene,
                        self.GRIPPER_SIGNAL,
                        gripper_command,
                    )
                    self._last_gripper_command = gripper_command
        except Exception as exc:
            print(f"M4 perdio la conexion con CoppeliaSim: {exc}")
            self._reset_connection()
            return False

        self._last_update = now
        return True

    @staticmethod
    def map_arm_angles(
        shoulder_deg: float,
        elbow_deg: float,
    ) -> tuple[float, float]:
        """Escala el rango humano completo al rango seguro del UR5."""
        shoulder_deg = max(0.0, min(180.0, shoulder_deg))
        elbow_deg = max(0.0, min(180.0, elbow_deg))

        shoulder_ratio = shoulder_deg / 180.0
        elbow_flexion_ratio = (180.0 - elbow_deg) / 180.0

        shoulder_target = (
            COPPELIASIM_SHOULDER_MAX_DEG
            + shoulder_ratio
            * (
                COPPELIASIM_SHOULDER_MIN_DEG
                - COPPELIASIM_SHOULDER_MAX_DEG
            )
        )
        elbow_target = (
            COPPELIASIM_ELBOW_MAX_DEG
            + elbow_flexion_ratio
            * (
                COPPELIASIM_ELBOW_MIN_DEG
                - COPPELIASIM_ELBOW_MAX_DEG
            )
        )
        return shoulder_target, elbow_target

    def move_home(self) -> None:
        """Solicita el retorno a la postura que tenía al conectarse."""
        if not self._connected or self._sim is None:
            return
        try:
            for handle, target in zip(
                self._joint_handles,
                self._home_positions,
            ):
                self._sim.setJointTargetPosition(
                    handle,
                    target,
                    self._motion_params,
                )
        except Exception as exc:
            print(f"M4 no pudo restaurar la postura inicial: {exc}")

    def close(self, restore_home: bool = True) -> None:
        """Finaliza el cliente; opcionalmente restaura la postura inicial."""
        if restore_home:
            self.move_home()
        self._reset_connection()

    def _set_joint_target(self, index: int, target: float) -> None:
        assert self._sim is not None
        self._sim.setJointTargetPosition(
            self._joint_handles[index],
            target,
            self._motion_params,
        )

    def _validate_arm_joints(self) -> None:
        assert self._sim is not None
        if len(self._joint_handles) != COPPELIASIM_JOINT_COUNT:
            raise RuntimeError("No se encontraron los seis joints del UR5.")

        paths = [
            self._sim.getObjectAlias(handle, 2)
            for handle in self._joint_handles
        ]
        if any("/RG2/" in path for path in paths):
            raise RuntimeError(
                "La busqueda de joints alcanzo el RG2; revisa la escena."
            )

    def _reset_connection(self) -> None:
        self._connected = False
        self._joint_handles = []
        self._home_positions = []
        self._last_gripper_command = None
        self._sim = None
        self._client = None

    def __enter__(self) -> CoppeliaRobot:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False
