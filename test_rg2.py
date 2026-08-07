"""Prueba manual del RG2 mediante el controlador seguro de CoppeliaSim."""

from __future__ import annotations

import time

from modules.commands import GripperCommand, MimicCommand
from modules.m4_coppeliasim import CoppeliaRobot, SafetyState


def main() -> None:
    robot = CoppeliaRobot()
    unsafe_session = False
    sequence = 0

    print("Conectando mediante el controlador seguro...")
    if not robot.connect():
        print("No se pudo completar el preflight. Revisa el estado mostrado.")
        robot.close(normal_exit=False)
        return

    print("Comandos:")
    print("  s : iniciar/pausar el canal manual (equivale a ESPACIO)")
    print("  a : abrir RG2")
    print("  c : cerrar RG2")
    print("  e : parada de emergencia")
    print("  r : rearmar despues de pulsar Play en CoppeliaSim")
    print("  q : salir")

    try:
        while True:
            current_state = robot.snapshot.state
            if current_state in {SafetyState.ESTOP, SafetyState.FAULT}:
                unsafe_session = True

            text = input("> ").strip().lower()
            if text == "s":
                if current_state is SafetyState.RUNNING:
                    changed = robot.pause()
                    action = "Pausa/hold"
                else:
                    changed = robot.start_imitation(require_arm=False)
                    action = "Inicio manual RG2"
                print(f"{action}: {'OK' if changed else 'rechazado'}.")

            elif text in {"a", "c"}:
                sequence += 1
                report = robot.submit(
                    MimicCommand(
                        sequence=sequence,
                        created_at=time.monotonic(),
                        gripper=GripperCommand(
                            aperture=1.0 if text == "a" else 0.0
                        ),
                    )
                )
                if report.gripper.accepted:
                    print("Orden aceptada: ABRIR" if text == "a" else "Orden aceptada: CERRAR")
                else:
                    print(f"Orden rechazada: {report.gripper.reason}.")

            elif text == "e":
                unsafe_session = True
                robot.emergency_stop("ESTOP solicitado desde test_rg2.py.")
                print("ESTOP solicitado. Pulsa Play en CoppeliaSim antes de R.")

            elif text == "r":
                if robot.snapshot.state in {SafetyState.ESTOP, SafetyState.FAULT}:
                    unsafe_session = True
                rearmed = robot.reconnect()
                print("Rearme OK; pulsa s para continuar." if rearmed else "Rearme rechazado.")

            elif text == "q":
                break

            else:
                print("Usa s, a, c, e, r o q.")
    except (EOFError, KeyboardInterrupt):
        unsafe_session = True
        robot.emergency_stop("Prueba RG2 interrumpida.")
        print("\nPrueba interrumpida; ESTOP solicitado.")
    finally:
        if robot.snapshot.state in {SafetyState.ESTOP, SafetyState.FAULT}:
            unsafe_session = True
        robot.close(normal_exit=not unsafe_session)


if __name__ == "__main__":
    main()
