"""M3: validación, cálculo y suavizado de comandos humanos."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from numbers import Real

from config.settings import (
    GRIPPER_D_MAX,
    GRIPPER_D_MIN,
    MIN_ARM_SEGMENT_LENGTH_PX,
    SMOOTHING_WINDOW,
    VISIBILITY_THRESHOLD,
)
from modules.commands import ArmCommand, GripperCommand, MimicCommand
from modules.m2_mediapipe import ELBOW, INDEX_TIP, SHOULDER, THUMB_TIP, WRIST_POSE


class AngleCalculator:
    """Convierte landmarks en comandos tipados con canales independientes."""

    JOINT_SHOULDER = "shoulder"
    JOINT_ELBOW = "elbow"
    JOINT_GRIPPER = "gripper"

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._sequence = 0
        self._buffers: dict[str, deque[float]] = {
            self.JOINT_SHOULDER: deque(maxlen=SMOOTHING_WINDOW),
            self.JOINT_ELBOW: deque(maxlen=SMOOTHING_WINDOW),
            self.JOINT_GRIPPER: deque(maxlen=SMOOTHING_WINDOW),
        }

    def compute(self, result: dict | None) -> MimicCommand:
        """Produce siempre un comando, incluyendo los motivos de cada rechazo.

        Un canal inválido no contamina los buffers ni impide usar el otro. Si
        desaparece la mano no se envía una apertura nueva; el controlador puede
        conservar así el último estado seguro del gripper.
        """
        self._sequence += 1
        created_at = float(self._clock())

        if isinstance(result, dict):
            arm_landmarks = result.get("arm")
            hand_landmarks = result.get("hand")
        else:
            arm_landmarks = None
            hand_landmarks = None

        arm, arm_error = self._compute_arm(arm_landmarks)
        gripper, gripper_error = self._compute_gripper(hand_landmarks)
        return MimicCommand(
            sequence=self._sequence,
            created_at=created_at,
            arm=arm,
            gripper=gripper,
            arm_error=arm_error,
            gripper_error=gripper_error,
        )

    def reset_buffers(self) -> None:
        """Descarta todo el historial de suavizado sin reiniciar la secuencia."""
        for buffer in self._buffers.values():
            buffer.clear()

    def _compute_arm(self, arm: object) -> tuple[ArmCommand | None, str | None]:
        error = self._validate_arm(arm)
        if error is not None:
            self._clear_arm_buffers()
            return None, error

        # ``_validate_arm`` garantiza la estructura de estos landmarks.
        raw_shoulder = self._signed_shoulder_elevation(  # type: ignore[index]
            arm[SHOULDER],  # type: ignore[index]
            arm[ELBOW],  # type: ignore[index]
        )
        raw_elbow = self._elbow_angle(  # type: ignore[index]
            arm[SHOULDER],  # type: ignore[index]
            arm[ELBOW],  # type: ignore[index]
            arm[WRIST_POSE],  # type: ignore[index]
        )
        if not (math.isfinite(raw_shoulder) and math.isfinite(raw_elbow)):
            self._clear_arm_buffers()
            return None, "arm_angle_non_finite"

        shoulder = round(
            self._smooth_circular(self.JOINT_SHOULDER, raw_shoulder),
            2,
        )
        elbow = round(self._smooth(self.JOINT_ELBOW, raw_elbow), 2)
        return ArmCommand(shoulder_deg=shoulder, elbow_deg=elbow), None

    def _compute_gripper(
        self,
        hand: object,
    ) -> tuple[GripperCommand | None, str | None]:
        error = self._validate_hand(hand)
        if error is not None:
            self._buffers[self.JOINT_GRIPPER].clear()
            return None, error

        thumb = hand[THUMB_TIP]  # type: ignore[index]
        index = hand[INDEX_TIP]  # type: ignore[index]
        distance = math.hypot(index["x"] - thumb["x"], index["y"] - thumb["y"])
        span = GRIPPER_D_MAX - GRIPPER_D_MIN
        if not math.isfinite(distance) or span <= 0:
            self._buffers[self.JOINT_GRIPPER].clear()
            return None, "gripper_distance_invalid"

        raw_aperture = min(1.0, max(0.0, (distance - GRIPPER_D_MIN) / span))
        aperture = round(self._smooth(self.JOINT_GRIPPER, raw_aperture), 3)
        return GripperCommand(aperture=aperture), None

    @staticmethod
    def _angle_between(
        p_proximal: dict,
        p_central: dict,
        p_distal: dict,
    ) -> float:
        """Calcula el ángulo entre dos vectores en grados."""
        center_x = float(p_central["x"])
        center_y = float(p_central["y"])
        vector_one = (
            float(p_proximal["x"]) - center_x,
            float(p_proximal["y"]) - center_y,
        )
        vector_two = (
            float(p_distal["x"]) - center_x,
            float(p_distal["y"]) - center_y,
        )
        norm_one = math.hypot(*vector_one)
        norm_two = math.hypot(*vector_two)
        if norm_one == 0.0 or norm_two == 0.0:
            return math.nan
        cosine = (
            vector_one[0] * vector_two[0] + vector_one[1] * vector_two[1]
        ) / (norm_one * norm_two)
        cosine = min(1.0, max(-1.0, cosine))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _signed_shoulder_elevation(shoulder: dict, elbow: dict) -> float:
        """Ángulo firmado desde abajo, coherente con la vista espejada del HUD."""
        display_dx = float(shoulder["x"]) - float(elbow["x"])
        down_dy = float(elbow["y"]) - float(shoulder["y"])
        if display_dx == 0.0 and down_dy == 0.0:
            return math.nan
        return math.degrees(math.atan2(display_dx, down_dy))

    @classmethod
    def _elbow_angle(cls, shoulder: dict, elbow: dict, wrist: dict) -> float:
        """Prefiere MediaPipe World para evitar distorsión por profundidad."""
        world_keys = ("world_x", "world_y", "world_z")
        if all(
            all(cls._is_finite_real(point.get(key)) for key in world_keys)
            for point in (shoulder, elbow, wrist)
        ):
            proximal = tuple(
                float(shoulder[key]) - float(elbow[key]) for key in world_keys
            )
            distal = tuple(
                float(wrist[key]) - float(elbow[key]) for key in world_keys
            )
            norm_proximal = math.sqrt(sum(value * value for value in proximal))
            norm_distal = math.sqrt(sum(value * value for value in distal))
            if norm_proximal == 0.0 or norm_distal == 0.0:
                return math.nan
            cosine = sum(
                first * second for first, second in zip(proximal, distal)
            ) / (norm_proximal * norm_distal)
            return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))
        return cls._angle_between(shoulder, elbow, wrist)

    def _smooth(self, joint: str, value: float) -> float:
        buffer = self._buffers[joint]
        buffer.append(value)
        return sum(buffer) / len(buffer)

    def _smooth_circular(self, joint: str, value: float) -> float:
        buffer = self._buffers[joint]
        buffer.append(value)
        sine = sum(math.sin(math.radians(sample)) for sample in buffer)
        cosine = sum(math.cos(math.radians(sample)) for sample in buffer)
        return math.degrees(math.atan2(sine, cosine))

    def _clear_arm_buffers(self) -> None:
        self._buffers[self.JOINT_SHOULDER].clear()
        self._buffers[self.JOINT_ELBOW].clear()

    @classmethod
    def _validate_arm(cls, arm: object) -> str | None:
        if not isinstance(arm, dict):
            return "arm_missing"
        required = (SHOULDER, ELBOW, WRIST_POSE)
        if not all(idx in arm for idx in required):
            return "arm_landmark_missing"

        for idx in required:
            landmark = arm[idx]
            if not isinstance(landmark, dict):
                return "arm_landmark_invalid"
            if not all(cls._is_finite_real(landmark.get(axis)) for axis in ("x", "y")):
                return "arm_landmark_non_finite"
            visibility = landmark.get("visibility")
            if not cls._is_finite_real(visibility):
                return "arm_visibility_non_finite"
            if float(visibility) < VISIBILITY_THRESHOLD:
                return "arm_not_visible"

        upper_arm_length = cls._distance(arm[SHOULDER], arm[ELBOW])
        forearm_length = cls._distance(arm[ELBOW], arm[WRIST_POSE])
        if (
            upper_arm_length < MIN_ARM_SEGMENT_LENGTH_PX
            or forearm_length < MIN_ARM_SEGMENT_LENGTH_PX
        ):
            return "arm_segment_too_short"
        return None

    @classmethod
    def _validate_hand(cls, hand: object) -> str | None:
        if hand is None:
            return "gripper_hand_missing"
        if not isinstance(hand, dict):
            return "gripper_hand_invalid"
        if not all(idx in hand for idx in (THUMB_TIP, INDEX_TIP)):
            return "gripper_landmark_missing"
        for idx in (THUMB_TIP, INDEX_TIP):
            landmark = hand[idx]
            if not isinstance(landmark, dict):
                return "gripper_landmark_invalid"
            if not all(cls._is_finite_real(landmark.get(axis)) for axis in ("x", "y")):
                return "gripper_landmark_non_finite"
        return None

    @staticmethod
    def _is_finite_real(value: object) -> bool:
        return (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    @staticmethod
    def _distance(first: dict, second: dict) -> float:
        return math.hypot(
            float(second["x"]) - float(first["x"]),
            float(second["y"]) - float(first["y"]),
        )
