"""Prueba manual del brazo usando el mismo controlador seguro de la aplicacion."""

from __future__ import annotations

import time

from modules.commands import ArmCommand, MimicCommand
from modules.m4_coppeliasim import CoppeliaRobot, SafetyState


STEP_DEG = 10.0
SETTLE_TIME_S = 0.15


def main() -> None:
    robot = CoppeliaRobot()
    unsafe_session = False
    sequence = 0
    shoulder_deg = 90.0
    elbow_deg = 90.0

    print("Conectando mediante el controlador seguro...")
    if not robot.connect():
        print("No se pudo completar el preflight. Revisa el estado mostrado.")
        robot.close(normal_exit=False)
        return

    print("Esta prueba envia un lote coordinado y hace hold tras cada paso.")
    confirmation = input("Continuar? [s/N]: ").strip().lower()
    if confirmation != "s":
        robot.close(normal_exit=True)
        print("Prueba cancelada sin enviar movimientos.")
        return

    print("Comandos:")
    print("  h- / h+ : variar hombro en 10 grados humanos")
    print("  c- / c+ : variar codo en 10 grados humanos")
    print("  e        : parada de emergencia enclavada")
    print("  r        : rearmar despues de pulsar Play en CoppeliaSim")
    print("  q        : salida normal; home solo si nunca hubo ESTOP/FAULT")

    try:
        while True:
            current_state = robot.snapshot.state
            if current_state in {SafetyState.ESTOP, SafetyState.FAULT}:
                unsafe_session = True

            text = input("> ").strip().lower()
            if text in {"h-", "h+", "c-", "c+"}:
                candidate_shoulder = shoulder_deg
                candidate_elbow = elbow_deg
                if text == "h-":
                    candidate_shoulder -= STEP_DEG
                elif text == "h+":
                    candidate_shoulder += STEP_DEG
                elif text == "c-":
                    candidate_elbow -= STEP_DEG
                else:
                    candidate_elbow += STEP_DEG

                if not robot.start_imitation(require_arm=True):
                    print(
                        f"Movimiento rechazado en estado {robot.snapshot.state.value}."
                    )
                    continue

                sequence += 1
                report = robot.submit(
                    MimicCommand(
                        sequence=sequence,
                        created_at=time.monotonic(),
                        arm=ArmCommand(
                            shoulder_deg=candidate_shoulder,
                            elbow_deg=candidate_elbow,
                        ),
                    )
                )
                time.sleep(SETTLE_TIME_S)
                robot.pause()

                if report.arm.accepted:
                    shoulder_deg = candidate_shoulder
                    elbow_deg = candidate_elbow
                    print(
                        f"Lote aceptado: hombro={shoulder_deg:.1f}, "
                        f"codo={elbow_deg:.1f}; hold solicitado."
                    )
                else:
                    print(f"Lote rechazado: {report.arm.reason}.")

            elif text == "e":
                unsafe_session = True
                robot.emergency_stop("ESTOP solicitado desde test_ur5.py.")
                print("ESTOP solicitado. Pulsa Play en CoppeliaSim antes de R.")

            elif text == "r":
                if robot.snapshot.state in {SafetyState.ESTOP, SafetyState.FAULT}:
                    unsafe_session = True
                rearmed = robot.reconnect()
                print("Rearme OK; usa otro comando para mover." if rearmed else "Rearme rechazado.")

            elif text == "q":
                break

            else:
                print("Usa h-, h+, c-, c+, e, r o q.")
    except (EOFError, KeyboardInterrupt):
        unsafe_session = True
        robot.emergency_stop("Prueba UR5 interrumpida.")
        print("\nPrueba interrumpida; ESTOP solicitado.")
    finally:
        if robot.snapshot.state in {SafetyState.ESTOP, SafetyState.FAULT}:
            unsafe_session = True
        robot.close(normal_exit=not unsafe_session)


if __name__ == "__main__":
    main()
