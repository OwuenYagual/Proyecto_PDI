import numpy as np
from collections import deque

from config.settings import (
    SMOOTHING_WINDOW,
    VISIBILITY_THRESHOLD,
    GRIPPER_D_MIN,
    GRIPPER_D_MAX,
)
from modules.m2_mediapipe import SHOULDER, ELBOW, WRIST_POSE, THUMB_TIP, INDEX_TIP


class AngleCalculator:
    """
    Módulo M3: calcula ángulos articulares del brazo derecho y apertura
    del gripper a partir del resultado fusionado de M2.

    Ángulos calculados:
        - Hombro : entre eje vertical y vector hombro→codo        (Pose)
        - Codo   : entre vector hombro→codo y vector codo→muñeca  (Pose)
        - Gripper: distancia euclidiana normalizada índice-pulgar  (Hands)
    """

    JOINT_SHOULDER = "shoulder"
    JOINT_ELBOW    = "elbow"
    JOINT_GRIPPER  = "gripper"

    def __init__(self):
        self._buffers: dict[str, deque] = {
            self.JOINT_SHOULDER: deque(maxlen=SMOOTHING_WINDOW),
            self.JOINT_ELBOW:    deque(maxlen=SMOOTHING_WINDOW),
            self.JOINT_GRIPPER:  deque(maxlen=SMOOTHING_WINDOW),
        }

    # ------------------------------------------------------------------ #
    # Interfaz pública                                                     #
    # ------------------------------------------------------------------ #

    def compute(self, result: dict | None) -> dict | None:
        """
        Calcula y suaviza los ángulos a partir del resultado de M2.

        Args:
            result: diccionario {"arm": {...}, "hand": {...}} de M2.process().

        Returns:
            {
              "shoulder": float,  # ángulo hombro  [0°, 180°]
              "elbow":    float,  # ángulo codo    [0°, 180°]
              "gripper":  float,  # apertura       [0.0, 1.0]
              "valid":    bool,
            }
            None si no hay landmarks del brazo disponibles.
        """
        if result is None:
            return None

        arm  = result.get("arm")
        hand = result.get("hand")

        if not self._arm_available(arm):
            return None

        # ── Ángulos desde Pose ────────────────────────────────────────
        raw_shoulder = self._angle_between(
            arm[SHOULDER],
            arm[SHOULDER],
            arm[ELBOW],
            reference="vertical",
        )
        raw_elbow = self._angle_between(
            arm[SHOULDER],
            arm[ELBOW],
            arm[WRIST_POSE],
        )

        # ── Apertura gripper desde Hands ──────────────────────────────
        raw_gripper = self._gripper_aperture(hand)
        if raw_gripper is None:
            gripper = self._current_smoothed(self.JOINT_GRIPPER, default=0.0)
        else:
            gripper = self._smooth(self.JOINT_GRIPPER, raw_gripper)

        # ── Media móvil ───────────────────────────────────────────────
        return {
            self.JOINT_SHOULDER: round(self._smooth(self.JOINT_SHOULDER, raw_shoulder), 2),
            self.JOINT_ELBOW:    round(self._smooth(self.JOINT_ELBOW,    raw_elbow),    2),
            self.JOINT_GRIPPER:  round(gripper, 3),
            "valid":             self._arm_valid(arm),
            "hand_detected":     hand is not None,
        }

    def reset_buffers(self) -> None:
        for buf in self._buffers.values():
            buf.clear()

    # ------------------------------------------------------------------ #
    # Cálculo de ángulo                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _angle_between(
        p_proximal: dict,
        p_central:  dict,
        p_distal:   dict,
        reference:  str | None = None,
    ) -> float:
        """
        θ = arccos( v1·v2 / (|v1||v2|) )

        reference="vertical"   → v1 = (0, -1)  eje hacia arriba
        reference=None         → v1 = proximal - central
        """
        cx, cy = p_central["x"], p_central["y"]

        v1 = (np.array([0.0, -1.0]) if reference == "vertical"
              else np.array([p_proximal["x"] - cx, p_proximal["y"] - cy]))

        v2 = np.array([p_distal["x"] - cx, p_distal["y"] - cy])

        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0

        cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    # ------------------------------------------------------------------ #
    # Apertura del gripper (desde Hands)                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gripper_aperture(hand: dict | None) -> float | None:
        """
        Distancia euclidiana entre punta del índice (lm 8) y punta
        del pulgar (lm 4), sincronizadas al espacio de Pose por M2.

        Returns:
            Apertura normalizada [0.0, 1.0].
            None si no hay detección de mano.
        """
        if hand is None:
            return None

        if THUMB_TIP not in hand or INDEX_TIP not in hand:
            return None

        dx = hand[INDEX_TIP]["x"] - hand[THUMB_TIP]["x"]
        dy = hand[INDEX_TIP]["y"] - hand[THUMB_TIP]["y"]
        d  = np.sqrt(dx ** 2 + dy ** 2)

        apertura = (d - GRIPPER_D_MIN) / (GRIPPER_D_MAX - GRIPPER_D_MIN)
        return float(np.clip(apertura, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    # Media móvil                                                          #
    # ------------------------------------------------------------------ #

    def _smooth(self, joint: str, value: float) -> float:
        self._buffers[joint].append(value)
        return float(np.mean(self._buffers[joint]))

    def _current_smoothed(self, joint: str, default: float) -> float:
        buffer = self._buffers[joint]
        return float(np.mean(buffer)) if buffer else default

    # ------------------------------------------------------------------ #
    # Validación                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _arm_available(arm: dict | None) -> bool:
        return arm is not None and all(
            idx in arm for idx in [SHOULDER, ELBOW, WRIST_POSE]
        )

    @staticmethod
    def _arm_valid(arm: dict) -> bool:
        return all(
            arm[idx]["visibility"] >= VISIBILITY_THRESHOLD
            for idx in [SHOULDER, ELBOW, WRIST_POSE]
        )
