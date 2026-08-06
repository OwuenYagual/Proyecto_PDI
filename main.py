import cv2
import time
import numpy as np
from collections import deque

from config.settings import FPS_SMOOTHING_WINDOW
from modules.m1_capture import CaptureModule
from modules.m2_mediapipe import PoseDetector
from modules.m3_angles import AngleCalculator
from modules.m4_coppeliasim import CoppeliaRobot


def draw_hud(
    frame: np.ndarray,
    angles: dict | None,
    fps: float,
    imitando: bool,
) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}",
        (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 136), 2, cv2.LINE_AA)

    if not imitando:
        cv2.putText(frame, "En espera",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return

    if angles is None:
        cv2.putText(frame, "Sin deteccion",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return

    color = (0, 255, 0) if angles["valid"] else (0, 165, 255)
    cv2.putText(frame,
        f"Hombro:{angles['shoulder']:5.1f}  Codo:{angles['elbow']:5.1f}",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    hand_txt = (f"Gripper: {angles['gripper']*100:.1f}%"
                if angles["hand_detected"] else "Gripper: sin mano")
    cv2.putText(frame, hand_txt,
        (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (0, 165, 255), 1, cv2.LINE_AA)


def main() -> None:
    print("Iniciando pipeline M1+M2+M3+M4 ...")
    print("Presiona ESPACIO para iniciar la imitacion.")
    print("Presiona 'q' para salir.")

    cv2.namedWindow("pose2robot - Camara")
    cv2.moveWindow("pose2robot - Camara", 0, 30)

    frame_durations = deque(maxlen=FPS_SMOOTHING_WINDOW)
    fps = 0.0
    imitando = False
    robot = CoppeliaRobot()
    robot.connect()

    try:
        with CaptureModule() as cam, PoseDetector() as detector:
            calculator = AngleCalculator()
            prev_time = time.perf_counter()

            while True:
                # M1
                ok, bgr, rgb = cam.read()
                if not ok:
                    break

                # M2
                result = detector.process(rgb, process_hands=imitando)

                # M3
                measured_angles = calculator.compute(result) if imitando else None
                if result is None:
                    calculator.reset_buffers()

                # M4
                if measured_angles is not None:
                    robot.update(measured_angles)

                # Ventana cámara con skeleton
                cam_frame = detector.draw_skeleton(bgr, result)

                draw_hud(cam_frame, measured_angles, fps, imitando)

                # Mensaje de espera si aún no ha iniciado
                if not imitando:
                    cv2.putText(cam_frame,
                        "Presiona ESPACIO para iniciar",
                        (10, cam_frame.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("pose2robot - Camara", cam_frame)

                key = cv2.waitKey(1) & 0xFF

                curr_time = time.perf_counter()
                frame_durations.append(curr_time - prev_time)
                prev_time = curr_time
                total_duration = sum(frame_durations)
                if total_duration > 0:
                    fps = len(frame_durations) / total_duration

                if key == ord(" "):
                    imitando = not imitando
                    calculator.reset_buffers()
                    estado = "INICIADA" if imitando else "PAUSADA"
                    print(f"Imitacion {estado}.")

                elif key == ord("q"):
                    print("Saliendo...")
                    break
    finally:
        robot.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
