"""Configuracion central y validada de Robot Mimic."""

from __future__ import annotations

import math
from numbers import Real


# Camara
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_MAX_CONSECUTIVE_FAILURES = 3

# Preprocesamiento (M1)
APPLY_GAUSSIAN = False
GAUSSIAN_KERNEL = (5, 5)

# MediaPipe Pose y Hands (M2)
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
POSE_MODEL_COMPLEXITY = 0
HAND_MODEL_COMPLEXITY = 0
HANDS_PROCESS_INTERVAL = 2
HAND_WRIST_MATCH_MAX_RATIO = 0.15

# Calculo de angulos (M3)
SMOOTHING_WINDOW = 5
VISIBILITY_THRESHOLD = 0.5
MIN_ARM_SEGMENT_LENGTH_PX = 5.0
GRIPPER_D_MIN = 20
GRIPPER_D_MAX = 120

# CoppeliaSim (M4)
COPPELIASIM_ENABLED = True
COPPELIASIM_HOST = "localhost"
COPPELIASIM_PORT = 23000
COPPELIASIM_CONNECTION_TIMEOUT_MS = 10000
COPPELIASIM_REQUEST_TIMEOUT_MS = 500
COPPELIASIM_COMMAND_MAX_AGE_MS = 250
COPPELIASIM_POSE_LOSS_ESTOP_MS = 750
COPPELIASIM_UPDATE_HZ = 20.0
COPPELIASIM_COLLISION_CHECK_HZ = 20.0
COPPELIASIM_VIEW_ENABLED = True
COPPELIASIM_VIEW_SENSOR_PATH = "/RobotMimicVisionSensor"
COPPELIASIM_VIEW_HZ = 10.0
COPPELIASIM_VIEW_WIDTH = 640
COPPELIASIM_VIEW_HEIGHT = 480
COPPELIASIM_JOINT_COUNT = 6
COPPELIASIM_SHOULDER_INDEX = 1
COPPELIASIM_ELBOW_INDEX = 2
COPPELIASIM_SHOULDER_MIN_DEG = -60.0
COPPELIASIM_SHOULDER_MAX_DEG = 0.0
COPPELIASIM_ELBOW_MIN_DEG = -60.0
COPPELIASIM_ELBOW_MAX_DEG = 0.0
COPPELIASIM_GRIPPER_THRESHOLD = 0.5
COPPELIASIM_MAX_VELOCITY_DEG = 45.0
COPPELIASIM_MAX_ACCELERATION_DEG = 90.0
COPPELIASIM_MAX_JERK_DEG = 360.0
COPPELIASIM_HOME_TOLERANCE_DEG = 1.0
COPPELIASIM_HOME_TIMEOUT_S = 5.0

# Visualizacion y metricas (M5)
FPS_SMOOTHING_WINDOW = 30


def _require_integer(
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = globals()[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} debe ser un entero; recibido {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} debe ser mayor o igual que {minimum}; recibido {value!r}."
        )
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{name} debe ser menor o igual que {maximum}; recibido {value!r}."
        )
    return value


def _require_finite_number(name: str) -> float:
    value = globals()[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} debe ser un numero real finito; recibido {value!r}.")
    return float(value)


def _require_positive_number(name: str) -> float:
    value = _require_finite_number(name)
    if value <= 0.0:
        raise ValueError(f"{name} debe ser mayor que cero; recibido {value!r}.")
    return value


def _require_unit_interval(name: str) -> float:
    value = _require_finite_number(name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} debe estar en el intervalo [0, 1]; recibido {value!r}.")
    return value


def _require_ordered_pair(minimum_name: str, maximum_name: str) -> None:
    minimum = _require_finite_number(minimum_name)
    maximum = _require_finite_number(maximum_name)
    if minimum >= maximum:
        raise ValueError(
            f"{minimum_name} debe ser menor que {maximum_name}; "
            f"recibidos {minimum!r} y {maximum!r}."
        )


def validate_settings() -> None:
    """Falla al inicio si una constante no cumple el contrato del pipeline."""

    _require_integer("CAMERA_INDEX", minimum=0)
    _require_integer("FRAME_WIDTH", minimum=1)
    _require_integer("FRAME_HEIGHT", minimum=1)
    _require_positive_number("CAMERA_FPS")
    _require_integer("CAMERA_MAX_CONSECUTIVE_FAILURES", minimum=1)

    if not isinstance(APPLY_GAUSSIAN, bool):
        raise ValueError(
            f"APPLY_GAUSSIAN debe ser booleano; recibido {APPLY_GAUSSIAN!r}."
        )
    if (
        not isinstance(GAUSSIAN_KERNEL, tuple)
        or len(GAUSSIAN_KERNEL) != 2
        or any(
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size % 2 == 0
            for size in GAUSSIAN_KERNEL
        )
    ):
        raise ValueError(
            "GAUSSIAN_KERNEL debe ser una tupla de dos enteros positivos impares; "
            f"recibido {GAUSSIAN_KERNEL!r}."
        )

    for name in (
        "MIN_DETECTION_CONFIDENCE",
        "MIN_TRACKING_CONFIDENCE",
        "VISIBILITY_THRESHOLD",
    ):
        _require_unit_interval(name)

    pose_complexity = _require_integer("POSE_MODEL_COMPLEXITY")
    if pose_complexity not in {0, 1, 2}:
        raise ValueError(
            "POSE_MODEL_COMPLEXITY debe ser 0, 1 o 2; "
            f"recibido {pose_complexity!r}."
        )
    hand_complexity = _require_integer("HAND_MODEL_COMPLEXITY")
    if hand_complexity not in {0, 1}:
        raise ValueError(
            "HAND_MODEL_COMPLEXITY debe ser 0 o 1; "
            f"recibido {hand_complexity!r}."
        )
    _require_integer("HANDS_PROCESS_INTERVAL", minimum=1)
    wrist_ratio = _require_positive_number("HAND_WRIST_MATCH_MAX_RATIO")
    if wrist_ratio > 1.0:
        raise ValueError(
            "HAND_WRIST_MATCH_MAX_RATIO debe estar en el intervalo (0, 1]; "
            f"recibido {wrist_ratio!r}."
        )

    _require_integer("SMOOTHING_WINDOW", minimum=1)
    _require_integer("FPS_SMOOTHING_WINDOW", minimum=1)
    _require_positive_number("MIN_ARM_SEGMENT_LENGTH_PX")
    gripper_minimum = _require_finite_number("GRIPPER_D_MIN")
    gripper_maximum = _require_finite_number("GRIPPER_D_MAX")
    if gripper_minimum < 0.0:
        raise ValueError(
            f"GRIPPER_D_MIN debe ser mayor o igual que cero; recibido {gripper_minimum!r}."
        )
    if gripper_minimum >= gripper_maximum:
        raise ValueError(
            "GRIPPER_D_MIN debe ser menor que GRIPPER_D_MAX; "
            f"recibidos {gripper_minimum!r} y {gripper_maximum!r}."
        )

    if not isinstance(COPPELIASIM_ENABLED, bool):
        raise ValueError(
            f"COPPELIASIM_ENABLED debe ser booleano; recibido {COPPELIASIM_ENABLED!r}."
        )
    if not isinstance(COPPELIASIM_HOST, str) or not COPPELIASIM_HOST.strip():
        raise ValueError(
            "COPPELIASIM_HOST debe ser una cadena no vacia; "
            f"recibido {COPPELIASIM_HOST!r}."
        )
    _require_integer("COPPELIASIM_PORT", minimum=1, maximum=65535)

    for name in (
        "COPPELIASIM_CONNECTION_TIMEOUT_MS",
        "COPPELIASIM_REQUEST_TIMEOUT_MS",
        "COPPELIASIM_COMMAND_MAX_AGE_MS",
        "COPPELIASIM_POSE_LOSS_ESTOP_MS",
    ):
        _require_integer(name, minimum=1)
    if COPPELIASIM_POSE_LOSS_ESTOP_MS <= COPPELIASIM_COMMAND_MAX_AGE_MS:
        raise ValueError(
            "COPPELIASIM_POSE_LOSS_ESTOP_MS debe ser mayor que "
            "COPPELIASIM_COMMAND_MAX_AGE_MS para permitir hold antes del ESTOP."
        )

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
        _require_positive_number(name)

    if not isinstance(COPPELIASIM_VIEW_ENABLED, bool):
        raise ValueError(
            "COPPELIASIM_VIEW_ENABLED debe ser booleano; "
            f"recibido {COPPELIASIM_VIEW_ENABLED!r}."
        )
    if (
        not isinstance(COPPELIASIM_VIEW_SENSOR_PATH, str)
        or not COPPELIASIM_VIEW_SENSOR_PATH.strip()
        or not COPPELIASIM_VIEW_SENSOR_PATH.startswith("/")
    ):
        raise ValueError(
            "COPPELIASIM_VIEW_SENSOR_PATH debe ser una ruta absoluta no vacia; "
            f"recibido {COPPELIASIM_VIEW_SENSOR_PATH!r}."
        )
    _require_integer("COPPELIASIM_VIEW_WIDTH", minimum=1)
    _require_integer("COPPELIASIM_VIEW_HEIGHT", minimum=1)

    joint_count = _require_integer("COPPELIASIM_JOINT_COUNT", minimum=1)
    if joint_count != 6:
        raise ValueError(
            "COPPELIASIM_JOINT_COUNT debe ser exactamente 6 para el UR5; "
            f"recibido {joint_count}."
        )
    shoulder_index = _require_integer("COPPELIASIM_SHOULDER_INDEX", minimum=0)
    elbow_index = _require_integer("COPPELIASIM_ELBOW_INDEX", minimum=0)
    if shoulder_index >= joint_count:
        raise ValueError(
            "COPPELIASIM_SHOULDER_INDEX debe ser menor que "
            f"COPPELIASIM_JOINT_COUNT ({joint_count}); recibido {shoulder_index}."
        )
    if elbow_index >= joint_count:
        raise ValueError(
            "COPPELIASIM_ELBOW_INDEX debe ser menor que "
            f"COPPELIASIM_JOINT_COUNT ({joint_count}); recibido {elbow_index}."
        )
    if shoulder_index == elbow_index:
        raise ValueError(
            "COPPELIASIM_SHOULDER_INDEX y COPPELIASIM_ELBOW_INDEX "
            "deben ser distintos."
        )

    _require_ordered_pair(
        "COPPELIASIM_SHOULDER_MIN_DEG",
        "COPPELIASIM_SHOULDER_MAX_DEG",
    )
    _require_ordered_pair(
        "COPPELIASIM_ELBOW_MIN_DEG",
        "COPPELIASIM_ELBOW_MAX_DEG",
    )
    _require_unit_interval("COPPELIASIM_GRIPPER_THRESHOLD")


validate_settings()
