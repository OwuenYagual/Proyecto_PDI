"""Prueba manual de apertura y cierre del RG2 en CoppeliaSim."""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


SIGNAL_NAME = "signal.RG2_open"


def set_gripper_open(sim, is_open: bool) -> None:
    """Envía al script original del RG2 la orden binaria de apertura."""
    sim.setIntProperty(sim.handle_scene, SIGNAL_NAME, int(is_open))


def main() -> None:
    print("Conectando con CoppeliaSim en localhost:23000...")

    try:
        client = RemoteAPIClient()
        sim = client.require("sim")
    except Exception as exc:
        print(f"No se pudo conectar con CoppeliaSim: {exc}")
        print("Abre CoppeliaSim, carga la escena y vuelve a intentarlo.")
        return

    if sim.getSimulationState() == sim.simulation_stopped:
        print("Conexión correcta, pero la simulación está detenida.")
        print("Presiona Play en CoppeliaSim y ejecuta nuevamente esta prueba.")
        return

    print("Conexión correcta.")
    print("Comandos: [a] abrir, [c] cerrar, [q] salir")

    while True:
        command = input("> ").strip().lower()

        if command == "a":
            set_gripper_open(sim, True)
            print("Orden enviada: ABRIR")
        elif command == "c":
            set_gripper_open(sim, False)
            print("Orden enviada: CERRAR")
        elif command == "q":
            print("Prueba finalizada.")
            break
        else:
            print("Comando no reconocido. Usa a, c o q.")


if __name__ == "__main__":
    main()
