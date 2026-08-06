"""Prueba manual y limitada de hombro y codo del UR5 en CoppeliaSim."""

import math

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


JOINT_COUNT = 6
SHOULDER_INDEX = 1
ELBOW_INDEX = 2
STEP_DEG = 10.0
MAX_OFFSET_DEG = 30.0
MOTION_PARAMS = [
    math.radians(45.0),
    math.radians(90.0),
    math.radians(360.0),
]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def find_arm_joints(sim) -> list[int]:
    ur5 = sim.getObject("/UR5")
    return [
        sim.getObject("./joint", {"proxy": ur5, "index": index})
        for index in range(JOINT_COUNT)
    ]


def set_target(sim, handle: int, target: float) -> None:
    sim.setJointTargetPosition(handle, target, MOTION_PARAMS)


def main() -> None:
    print("Conectando con CoppeliaSim en localhost:23000...")

    try:
        client = RemoteAPIClient()
        sim = client.require("sim")
        joints = find_arm_joints(sim)
    except Exception as exc:
        print(f"No se pudo preparar la prueba: {exc}")
        return

    if sim.getSimulationState() == sim.simulation_stopped:
        print("La simulacion esta detenida. Presiona Play y vuelve a ejecutar.")
        return

    print("Articulaciones encontradas:")
    for index, handle in enumerate(joints, start=1):
        path = sim.getObjectAlias(handle, 2)
        position_deg = math.degrees(sim.getJointPosition(handle))
        print(f"  {index}: {path} ({position_deg:.1f} deg)")

    confirmation = input("Continuar con movimientos limitados? [s/N]: ").strip().lower()
    if confirmation != "s":
        print("Prueba cancelada sin mover el robot.")
        return

    initial = [sim.getJointPosition(handle) for handle in joints]
    targets = initial.copy()
    max_offset = math.radians(MAX_OFFSET_DEG)
    step = math.radians(STEP_DEG)

    print("Comandos:")
    print("  h- / h+ : mover hombro -10 / +10 grados")
    print("  e- / e+ : mover codo    -10 / +10 grados")
    print("  r        : restaurar postura inicial")
    print("  q        : restaurar y salir")

    while True:
        command = input("> ").strip().lower()

        if command in {"h-", "h+"}:
            delta = -step if command == "h-" else step
            index = SHOULDER_INDEX
        elif command in {"e-", "e+"}:
            delta = -step if command == "e-" else step
            index = ELBOW_INDEX
        elif command == "r":
            targets = initial.copy()
            for handle, target in zip(joints, targets):
                set_target(sim, handle, target)
            print("Restaurando postura inicial.")
            continue
        elif command == "q":
            for handle, target in zip(joints, initial):
                set_target(sim, handle, target)
            print("Restaurando postura inicial y finalizando.")
            break
        else:
            print("Comando no reconocido: usa h-, h+, e-, e+, r o q.")
            continue

        targets[index] = clamp(
            targets[index] + delta,
            initial[index] - max_offset,
            initial[index] + max_offset,
        )
        set_target(sim, joints[index], targets[index])
        print(f"Objetivo: {math.degrees(targets[index]):.1f} grados")


if __name__ == "__main__":
    main()
