import cv2
import mediapipe as mp
import numpy as np

from config.settings import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    VISIBILITY_THRESHOLD,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    POSE_MODEL_COMPLEXITY,
    HAND_MODEL_COMPLEXITY,
    HANDS_PROCESS_INTERVAL,
)

# Índices MediaPipe Pose
SHOULDER   = 11
ELBOW      = 13
WRIST_POSE = 15   # muñeca según Pose (ancla de sincronización)

# Índices MediaPipe Hands
WRIST_HAND = 0    # muñeca según Hands (ancla de sincronización)
THUMB_TIP  = 4    # punta del pulgar
INDEX_TIP  = 8    # punta del índice


class PoseDetector:
    """
    Módulo M2: corre MediaPipe Pose y MediaPipe Hands
    sobre el mismo frame y sincroniza sus landmarks usando la muñeca
    como punto de referencia común.

    Retorna un diccionario con dos secciones:
        "arm"  : landmarks del brazo en píxeles
                 {SHOULDER: {...}, ELBOW: {...}, WRIST_POSE: {...}}
        "hand" : landmarks de dedos sincronizados con Pose
                 {WRIST_HAND: {...}, THUMB_TIP: {...}, INDEX_TIP: {...}}
                 None si no se detectó ninguna mano.
    """

    def __init__(self):
        self.mp_pose    = mp.solutions.pose
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=POSE_MODEL_COMPLEXITY,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,              # solo una mano necesaria
            model_complexity=HAND_MODEL_COMPLEXITY,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self._cached_hand = None
        self._hands_initialized = False
        self._hands_frames_since_process = HANDS_PROCESS_INTERVAL

    # Interfaz pública

    def process(
        self,
        rgb_frame: np.ndarray,
        process_hands: bool = True,
    ) -> dict | None:
        """
        Ejecuta Pose y Hands sobre el frame RGB y retorna los landmarks
        fusionados y sincronizados.

        Args:
            rgb_frame: imagen RGB uint8 (H, W, 3).
            process_hands: activa Hands; si es False solo procesa Pose.

        Returns:
            {
              "arm": {
                  SHOULDER:   {"x": px, "y": px, "visibility": float},
                  ELBOW:      {"x": px, "y": px, "visibility": float},
                  WRIST_POSE: {"x": px, "y": px, "visibility": float},
              },
              "hand": {
                  WRIST_HAND: {"x": px, "y": px, "visibility": float},
                  THUMB_TIP:  {"x": px, "y": px, "visibility": float},
                  INDEX_TIP:  {"x": px, "y": px, "visibility": float},
              } | None
            }
            None si no se detectó el brazo.
        """
        rgb_frame.flags.writeable = False
        try:
            pose_results = self.pose.process(rgb_frame)

            # Sin pose no se necesita ejecutar el modelo adicional de manos.
            if not pose_results.pose_landmarks:
                self._reset_hand_cache()
                return None

            arm_landmarks = self._extract_arm(pose_results.pose_landmarks)
            hand_landmarks = self._process_hand_if_needed(
                rgb_frame,
                enabled=process_hands,
            )
        finally:
            rgb_frame.flags.writeable = True

        # Sincronizar el resultado reutilizado de Hands con la muñeca actual.
        if hand_landmarks is not None:
            hand_landmarks = self._sync_wrists(arm_landmarks, hand_landmarks)

        return {
            "arm":  arm_landmarks,
            "hand": hand_landmarks,
        }

    def draw_skeleton(self, bgr_frame: np.ndarray, result: dict | None) -> np.ndarray:
        """
        Dibuja el skeleton del brazo (verde) y los landmarks de dedos
        (naranja) sobre el frame BGR.

        Args:
            bgr_frame: frame BGR de entrada.
            result   : diccionario retornado por process().

        Returns:
            Frame BGR con overlays dibujados.
        """
        if result is None:
            return bgr_frame

        frame = bgr_frame.copy()

        # ── Brazo (verde) ─────────────────────────────────────────────
        arm = result["arm"]
        arm_connections = [
            (SHOULDER, ELBOW),
            (ELBOW,    WRIST_POSE),
        ]
        for start_idx, end_idx in arm_connections:
            lm_s = arm[start_idx]
            lm_e = arm[end_idx]
            if (lm_s["visibility"] < VISIBILITY_THRESHOLD or
                    lm_e["visibility"] < VISIBILITY_THRESHOLD):
                continue
            cv2.line(
                frame,
                (int(lm_s["x"]), int(lm_s["y"])),
                (int(lm_e["x"]), int(lm_e["y"])),
                (34, 197, 94), 2,
            )
        for idx, lm in arm.items():
            if lm["visibility"] < VISIBILITY_THRESHOLD:
                continue
            cx, cy = int(lm["x"]), int(lm["y"])
            cv2.circle(frame, (cx, cy), 5, (34, 197, 94), -1)
            cv2.putText(
                frame, str(idx),
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (34, 197, 94), 1, cv2.LINE_AA,
            )

        # ── Mano (naranja) ────────────────────────────────────────────
        hand = result.get("hand")
        if hand is not None:
            # Línea pulgar-índice (representa la apertura del gripper)
            lm_thumb = hand[THUMB_TIP]
            lm_index = hand[INDEX_TIP]
            cv2.line(
                frame,
                (int(lm_thumb["x"]), int(lm_thumb["y"])),
                (int(lm_index["x"]), int(lm_index["y"])),
                (0, 165, 255), 2,
            )
            # Conexión muñeca → pulgar y muñeca → índice
            lm_wrist = hand[WRIST_HAND]
            for lm_tip in [lm_thumb, lm_index]:
                cv2.line(
                    frame,
                    (int(lm_wrist["x"]), int(lm_wrist["y"])),
                    (int(lm_tip["x"]),   int(lm_tip["y"])),
                    (0, 165, 255), 1,
                )
            # Puntos de los dedos
            for idx, lm in hand.items():
                cx, cy = int(lm["x"]), int(lm["y"])
                cv2.circle(frame, (cx, cy), 4, (0, 165, 255), -1)
                label = {WRIST_HAND: "W", THUMB_TIP: "4", INDEX_TIP: "8"}
                cv2.putText(
                    frame, label[idx],
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 165, 255), 1, cv2.LINE_AA,
                )

        return frame

    def is_arm_visible(self, result: dict | None) -> bool:
        """True si hombro, codo y muñeca superan el umbral de visibilidad."""
        if result is None:
            return False
        arm = result["arm"]
        return all(
            arm[idx]["visibility"] >= VISIBILITY_THRESHOLD
            for idx in [SHOULDER, ELBOW, WRIST_POSE]
        )

    def release(self) -> None:
        self.pose.close()
        self.hands.close()

    # Context manager

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False

    # Métodos privados

    def _process_hand_if_needed(
        self,
        rgb_frame: np.ndarray,
        enabled: bool,
    ) -> dict | None:
        if not enabled:
            self._reset_hand_cache()
            return None

        self._hands_frames_since_process += 1
        should_process = (
            not self._hands_initialized
            or self._hands_frames_since_process >= HANDS_PROCESS_INTERVAL
        )
        if should_process:
            hands_results = self.hands.process(rgb_frame)
            self._cached_hand = self._extract_hand(hands_results)
            self._hands_initialized = True
            self._hands_frames_since_process = 0

        return self._cached_hand

    def _reset_hand_cache(self) -> None:
        self._cached_hand = None
        self._hands_initialized = False
        self._hands_frames_since_process = HANDS_PROCESS_INTERVAL

    def _extract_arm(self, pose_landmarks) -> dict:
        """
        Extrae hombro, codo y muñeca de Pose y los convierte a píxeles.
        """
        extracted = {}
        for idx in [SHOULDER, ELBOW, WRIST_POSE]:
            lm = pose_landmarks.landmark[idx]
            extracted[idx] = {
                "x":          lm.x * FRAME_WIDTH,
                "y":          lm.y * FRAME_HEIGHT,
                "visibility": lm.visibility,
            }
        return extracted

    def _extract_hand(self, hands_results) -> dict | None:
        """
        Extrae muñeca (0), pulgar (4) e índice (8) de Hands.
        Retorna None si no hay detección de mano.
        """
        if not hands_results.multi_hand_landmarks:
            return None

        # Tomar la primera mano detectada
        hand_lms = hands_results.multi_hand_landmarks[0]

        extracted = {}
        for idx in [WRIST_HAND, THUMB_TIP, INDEX_TIP]:
            lm = hand_lms.landmark[idx]
            extracted[idx] = {
                "x":          lm.x * FRAME_WIDTH,
                "y":          lm.y * FRAME_HEIGHT,
                "visibility": 1.0,   # Hands no provee visibility, asumimos 1.0
            }
        return extracted

    def _sync_wrists(self, arm: dict, hand: dict) -> dict:
        """
        Sincroniza los landmarks de Hands al espacio de coordenadas
        de Pose usando la muñeca como ancla común.

        Calcula el offset entre:
            muñeca_pose (lm 15)  y  muñeca_hand (lm 0)
        y lo aplica a todos los landmarks de la mano.

        Args:
            arm : landmarks del brazo en píxeles (de Pose).
            hand: landmarks de la mano en píxeles (de Hands).

        Returns:
            Diccionario de landmarks de mano corregidos.
        """
        wrist_pose_x = arm[WRIST_POSE]["x"]
        wrist_pose_y = arm[WRIST_POSE]["y"]

        wrist_hand_x = hand[WRIST_HAND]["x"]
        wrist_hand_y = hand[WRIST_HAND]["y"]

        offset_x = wrist_pose_x - wrist_hand_x
        offset_y = wrist_pose_y - wrist_hand_y

        synced = {}
        for idx, lm in hand.items():
            synced[idx] = {
                "x":          lm["x"] + offset_x,
                "y":          lm["y"] + offset_y,
                "visibility": lm["visibility"],
            }

        return synced
