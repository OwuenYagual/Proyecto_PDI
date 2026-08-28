"""Control seguro y no bloqueante del UR5/RG2 en CoppeliaSim.

El cliente ZeroMQ pertenece exclusivamente al hilo worker.  Los métodos
públicos sólo validan datos y publican intenciones, de modo que la captura de
cámara y el teclado no quedan bloqueados por una RPC lenta.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
import threading
import time
from typing import Any, Callable

from config import settings
from modules.commands import (
    ArmCommand,
    ChannelValidation,
    GripperCommand,
    MimicCommand,
    ValidationReport,
)


class SafetyState(Enum):
    """Estados explícitos del controlador de seguridad."""

    DISCONNECTED = "disconnected"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ESTOP = "estop"
    FAULT = "fault"


@dataclass(frozen=True)
class SafetySnapshot:
    """Vista atómica del estado para el HUD y la aplicación principal."""

    state: SafetyState
    connected: bool
    message: str
    collision_pair: tuple[str, str] | None


@dataclass(frozen=True)
class SimulatorFrameSnapshot:
    """Ultimo frame RGB inmutable producido por el Vision Sensor."""

    rgb: bytes | None
    width: int
    height: int
    sequence: int
    captured_at: float | None
    message: str


class CoppeliaRobot:
    """Controla un UR5 simulado aplicando validación y vigilancia reactiva."""

    GRIPPER_SIGNAL = "signal.RG2_open"

    def __init__(
        self,
        client_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client_factory = client_factory
        self._clock = clock

        self._lock = threading.RLock()
        self._snapshot = SafetySnapshot(
            SafetyState.DISCONNECTED,
            False,
            "CoppeliaSim desconectado.",
            None,
        )
        self._simulator_frame = SimulatorFrameSnapshot(
            None,
            0,
            0,
            0,
            None,
            "Vista de CoppeliaSim no disponible.",
        )

        self._client: Any | None = None
        self._sim: Any | None = None
        self._worker: threading.Thread | None = None
        self._joint_handles: list[int] = []
        self._home_positions: list[float] = []
        self._scene_limits: dict[int, tuple[float, float]] = {}
        self._effective_limits: dict[int, tuple[float, float]] = {}
        self._robot_collection: int | None = None
        self._environment_collection: int | None = None
        self._collision_aliases: dict[int, str] = {}
        self._rg2_handle: int | None = None
        self._vision_sensor_handle: int | None = None

        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._wake_event = threading.Event()
        self._home_done = threading.Event()

        self._latest_command: MimicCommand | None = None
        self._hold_requested = False
        self._home_requested = False
        self._home_succeeded = False
        self._estop_request: str | None = None
        self._simulation_stop_attempted = False
        self._simulation_stop_succeeded = False
        self._simulation_stop_error: str | None = None
        self._home_forbidden = False
        self._last_gripper_command: int | None = None
        self._last_valid_arm_at: float | None = None
        self._arm_hold_sent = False
        self._require_arm = True
        self._last_command_at = -math.inf
        self._last_collision_check = -math.inf
        self._last_view_capture = -math.inf

        self._update_interval = 1.0 / float(settings.COPPELIASIM_UPDATE_HZ)
        self._collision_interval = 1.0 / float(
            settings.COPPELIASIM_COLLISION_CHECK_HZ
        )
        self._view_interval = 1.0 / float(settings.COPPELIASIM_VIEW_HZ)
        self._view_enabled = bool(settings.COPPELIASIM_VIEW_ENABLED)
        self._view_sensor_path = str(settings.COPPELIASIM_VIEW_SENSOR_PATH)
        self._view_width = int(settings.COPPELIASIM_VIEW_WIDTH)
        self._view_height = int(settings.COPPELIASIM_VIEW_HEIGHT)
        self._view_camera_position = tuple(
            float(value) for value in settings.COPPELIASIM_VIEW_CAMERA_POSITION
        )
        self._view_target_position = tuple(
            float(value) for value in settings.COPPELIASIM_VIEW_TARGET_POSITION
        )
        self._view_angle_deg = float(settings.COPPELIASIM_VIEW_ANGLE_DEG)
        self._base_yaw_deg = float(
            getattr(settings, "COPPELIASIM_BASE_YAW_DEG", 0.0)
        )
        self._command_max_age = (
            float(getattr(settings, "COPPELIASIM_COMMAND_MAX_AGE_MS", 250))
            / 1000.0
        )
        self._pose_loss_estop = (
            float(getattr(settings, "COPPELIASIM_POSE_LOSS_ESTOP_MS", 750))
            / 1000.0
        )
        self._request_timeout_ms = int(
            getattr(settings, "COPPELIASIM_REQUEST_TIMEOUT_MS", 500)
        )
        self._home_tolerance = math.radians(
            float(getattr(settings, "COPPELIASIM_HOME_TOLERANCE_DEG", 1.0))
        )
        self._home_timeout = float(
            getattr(settings, "COPPELIASIM_HOME_TIMEOUT_S", 5.0)
        )
        self._motion_params = [
            math.radians(float(settings.COPPELIASIM_MAX_VELOCITY_DEG)),
            math.radians(float(settings.COPPELIASIM_MAX_ACCELERATION_DEG)),
            math.radians(float(settings.COPPELIASIM_MAX_JERK_DEG)),
        ]

    @property
    def snapshot(self) -> SafetySnapshot:
        with self._lock:
            return self._snapshot

    @property
    def simulator_frame(self) -> SimulatorFrameSnapshot:
        """Devuelve el ultimo frame sin realizar ninguna RPC."""
        with self._lock:
            return self._simulator_frame

    @property
    def is_connected(self) -> bool:
        """Compatibilidad de lectura con la interfaz anterior."""
        return self.snapshot.connected

    def connect(self) -> bool:
        """Inicia el worker y espera de forma acotada a que complete preflight."""
        if not bool(settings.COPPELIASIM_ENABLED):
            self._set_state(
                SafetyState.DISCONNECTED,
                False,
                "M4 CoppeliaSim deshabilitado en settings.py.",
            )
            return False

        with self._lock:
            if (
                self._simulation_stop_attempted
                and not self._simulation_stop_succeeded
            ):
                return False
            if self._worker is not None and self._worker.is_alive():
                return self._snapshot.state in {
                    SafetyState.READY,
                    SafetyState.RUNNING,
                    SafetyState.PAUSED,
                }
            if self._snapshot.state is not SafetyState.DISCONNECTED:
                return False
            self._prepare_worker_locked()
            self._set_state_locked(
                SafetyState.DISCONNECTED,
                False,
                "Conectando con CoppeliaSim...",
                None,
            )
            self._worker = threading.Thread(
                target=self._worker_main,
                name="coppeliasim-safety-worker",
                daemon=True,
            )
            self._worker.start()

        wait_seconds = max(
            float(settings.COPPELIASIM_CONNECTION_TIMEOUT_MS) / 1000.0,
            self._request_timeout_ms / 1000.0,
            0.1,
        ) + 0.25
        if not self._ready_event.wait(wait_seconds):
            self._stop_event.set()
            self._wake_event.set()
            self._set_state(
                SafetyState.DISCONNECTED,
                False,
                "Tiempo de conexión con CoppeliaSim agotado.",
            )
            return False
        return self.snapshot.state is SafetyState.READY

    def submit(self, command: MimicCommand) -> ValidationReport:
        """Valida y reemplaza el comando pendiente sin realizar ninguna RPC."""
        now = self._clock()
        with self._lock:
            report = self._validate_command_locked(command, now)
            wake_for_hold = False

            if self._snapshot.state is SafetyState.RUNNING and self._require_arm:
                if report.arm.accepted:
                    self._last_valid_arm_at = now
                    self._arm_hold_sent = False
                elif not self._arm_hold_sent:
                    self._hold_requested = True
                    wake_for_hold = True

            if report.arm.accepted or report.gripper.accepted:
                self._latest_command = command
            else:
                # Una muestra inválida es también la muestra más reciente: no
                # debe sobrevivir en la cola un objetivo válido anterior.
                self._latest_command = None
            if wake_for_hold:
                self._wake_event.set()
            return report

    def start_imitation(self, require_arm: bool = True) -> bool:
        """Pasa READY/PAUSED a RUNNING y activa el watchdog del brazo."""
        if not isinstance(require_arm, bool):
            return False
        with self._lock:
            if self._snapshot.state not in {
                SafetyState.READY,
                SafetyState.PAUSED,
            }:
                return False
            self._require_arm = require_arm
            self._last_valid_arm_at = self._clock() if require_arm else None
            self._arm_hold_sent = False
            self._hold_requested = False
            self._set_state_locked(
                SafetyState.RUNNING,
                True,
                "Imitación activa.",
                None,
            )
            self._wake_event.set()
            return True

    def pause(self) -> bool:
        """Pausa la imitación y solicita hold inmediato al worker."""
        with self._lock:
            if self._snapshot.state is not SafetyState.RUNNING:
                return False
            self._latest_command = None
            self._hold_requested = True
            self._set_state_locked(
                SafetyState.PAUSED,
                True,
                "Imitación pausada; hold solicitado.",
                None,
            )
            self._wake_event.set()
            return True

    def emergency_stop(self, reason: str) -> None:
        """Enclava ESTOP sin bloquear; el worker detiene la simulación."""
        safe_reason = str(reason).strip() or "Parada de emergencia solicitada."
        with self._lock:
            if self._snapshot.state is SafetyState.ESTOP:
                return
            if (
                self._simulation_stop_attempted
                and not self._simulation_stop_succeeded
            ):
                self._set_state_locked(
                    SafetyState.FAULT,
                    False,
                    self._snapshot.message,
                    self._snapshot.collision_pair,
                )
                return
            self._latest_command = None
            self._hold_requested = False
            self._estop_request = safe_reason
            self._set_state_locked(
                SafetyState.ESTOP,
                self._snapshot.connected,
                safe_reason,
                self._snapshot.collision_pair,
            )
            self._wake_event.set()

    def reconnect(self) -> bool:
        """Cierra el runtime anterior y repite conexión/preflight desde cero."""
        with self._lock:
            if (
                self._simulation_stop_attempted
                and not self._simulation_stop_succeeded
            ):
                return False
            if self._snapshot.state not in {
                SafetyState.DISCONNECTED,
                SafetyState.ESTOP,
                SafetyState.FAULT,
            }:
                return False
        if not self._shutdown_worker():
            return False
        with self._lock:
            if (
                self._simulation_stop_attempted
                and not self._simulation_stop_succeeded
            ):
                return False
        self._set_state(
            SafetyState.DISCONNECTED,
            False,
            "Rearmando conexión con CoppeliaSim...",
            collision_pair=None,
        )
        return self.connect()

    def close(self, normal_exit: bool = True) -> None:
        """Cierra el worker; sólo una salida normal puede solicitar home."""
        worker = self._worker
        state = self.snapshot.state
        if (
            normal_exit
            and worker is not None
            and worker.is_alive()
            and state not in {SafetyState.ESTOP, SafetyState.FAULT}
            and self.snapshot.connected
            and not self._home_forbidden
        ):
            with self._lock:
                self._home_requested = True
                self._home_succeeded = False
                self._home_done.clear()
                self._latest_command = None
                self._wake_event.set()
            self._home_done.wait(self._home_timeout + 1.0)

        if not self._shutdown_worker():
            return
        final_snapshot = self.snapshot
        if final_snapshot.state in {SafetyState.ESTOP, SafetyState.FAULT}:
            self._set_state(
                final_snapshot.state,
                False,
                final_snapshot.message,
                collision_pair=final_snapshot.collision_pair,
            )
        else:
            self._set_state(
                SafetyState.DISCONNECTED,
                False,
                "Controlador CoppeliaSim cerrado.",
                collision_pair=None,
            )

    @staticmethod
    def map_arm_angles(
        shoulder_deg: float,
        elbow_deg: float,
    ) -> tuple[float, float]:
        """Aplica offsets y signos calibrados sin comprimir los ángulos."""
        shoulder_delta = (
            shoulder_deg
            - float(settings.COPPELIASIM_SHOULDER_HUMAN_NEUTRAL_DEG)
        )
        shoulder_deadband = float(
            settings.COPPELIASIM_SHOULDER_NEUTRAL_DEADBAND_DEG
        )
        shoulder_delta = math.copysign(
            max(0.0, abs(shoulder_delta) - shoulder_deadband),
            shoulder_delta,
        )
        shoulder_target = (
            float(settings.COPPELIASIM_SHOULDER_ROBOT_NEUTRAL_DEG)
            + float(settings.COPPELIASIM_SHOULDER_DIRECTION)
            * shoulder_delta
        )
        elbow_target = (
            float(settings.COPPELIASIM_ELBOW_ROBOT_STRAIGHT_DEG)
            + float(settings.COPPELIASIM_ELBOW_DIRECTION)
            * (
                float(settings.COPPELIASIM_ELBOW_HUMAN_STRAIGHT_DEG)
                - elbow_deg
            )
        )
        return shoulder_target, elbow_target

    def _prepare_worker_locked(self) -> None:
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._wake_event = threading.Event()
        self._home_done = threading.Event()
        self._latest_command = None
        self._hold_requested = False
        self._home_requested = False
        self._home_succeeded = False
        self._estop_request = None
        self._simulation_stop_attempted = False
        self._simulation_stop_succeeded = False
        self._simulation_stop_error = None
        self._last_gripper_command = None
        self._last_valid_arm_at = None
        self._arm_hold_sent = False
        self._last_command_at = -math.inf
        self._last_collision_check = -math.inf
        self._last_view_capture = -math.inf
        self._vision_sensor_handle = None
        self._simulator_frame = SimulatorFrameSnapshot(
            None,
            0,
            0,
            0,
            None,
            "Esperando la vista de CoppeliaSim...",
        )

    def _shutdown_worker(self) -> bool:
        worker = self._worker
        if worker is None:
            return True
        self._stop_event.set()
        self._wake_event.set()
        if worker is not threading.current_thread():
            worker.join(max(1.0, self._request_timeout_ms / 1000.0 * 3.0))
        with self._lock:
            if worker.is_alive():
                self._set_state_locked(
                    SafetyState.FAULT,
                    False,
                    "El worker de CoppeliaSim no respondió al cierre.",
                    self._snapshot.collision_pair,
                )
                return False
            elif self._worker is worker:
                self._worker = None
        return True

    def _worker_main(self) -> None:
        try:
            self._open_runtime()
            pair = self._collision_pair()
            if pair is not None:
                self._enter_collision_estop(pair)
                self._ready_event.set()
            else:
                if self._stop_event.is_set():
                    return
                self._set_state(
                    SafetyState.READY,
                    True,
                    "UR5 y RG2 listos; preflight superado.",
                    collision_pair=None,
                )
                self._ready_event.set()

            loop_interval = min(
                self._update_interval,
                self._collision_interval,
                self._view_interval,
            )
            while not self._stop_event.is_set():
                try:
                    self._worker_cycle()
                except Exception as exc:  # una RPC fallida es un fallo fatal
                    self._enter_fault_worker(f"Fallo RPC de CoppeliaSim: {exc}")
                self._wake_event.wait(loop_interval)
                self._wake_event.clear()
        except Exception as exc:
            self._set_state(
                SafetyState.DISCONNECTED,
                False,
                f"M4 sin conexión con CoppeliaSim: {exc}",
                collision_pair=None,
            )
            self._ready_event.set()
        finally:
            # close(False) puede llegar inmediatamente después de E. Aunque el
            # evento de cierre ya esté activo, la RPC de parada pendiente debe
            # ejecutarse desde este mismo worker antes de soltar el runtime.
            pending_estop = self._take_estop_request()
            if pending_estop is not None and self._sim is not None:
                self._enter_estop_worker(pending_estop)
            self._home_done.set()
            self._destroy_collections()
            with self._lock:
                self._client = None
                self._sim = None
                self._joint_handles = []
                self._home_positions = []
                self._scene_limits = {}
                self._effective_limits = {}
                self._collision_aliases = {}
                self._rg2_handle = None
                self._vision_sensor_handle = None
                current = self._simulator_frame
                self._simulator_frame = SimulatorFrameSnapshot(
                    None,
                    0,
                    0,
                    current.sequence,
                    current.captured_at,
                    "Vista de CoppeliaSim desconectada.",
                )

    def _open_runtime(self) -> None:
        if self._client_factory is None:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient

            factory: Callable[..., Any] = RemoteAPIClient
        else:
            factory = self._client_factory

        client = factory(
            host=settings.COPPELIASIM_HOST,
            port=settings.COPPELIASIM_PORT,
        )
        self._configure_socket(
            client,
            int(settings.COPPELIASIM_CONNECTION_TIMEOUT_MS),
        )
        sim = client.require("sim")
        # require() realiza una recepción y puede restaurar opciones del socket;
        # por eso los timeouts operativos se configuran después.
        self._configure_socket(client, self._request_timeout_ms)
        self._client = client
        self._sim = sim
        self._preflight()

    def _configure_socket(self, client: Any, timeout_ms: int) -> None:
        socket = getattr(client, "socket", None)
        if socket is None or not callable(getattr(socket, "setsockopt", None)):
            raise RuntimeError("el cliente no permite configurar timeout RPC")
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError("pyzmq es obligatorio para configurar timeout RPC") from exc
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.LINGER, 0)

    def _preflight(self) -> None:
        sim = self._require_sim()
        if not self._simulation_is_running():
            raise RuntimeError("inicia la simulación con Play antes de conectar")

        ur5_handle = sim.getObject("/UR5")
        if not self._valid_handle(ur5_handle):
            raise RuntimeError("no se encontró /UR5")
        self._orient_ur5_base(ur5_handle)

        count = int(settings.COPPELIASIM_JOINT_COUNT)
        shoulder_index = int(settings.COPPELIASIM_SHOULDER_INDEX)
        elbow_index = int(settings.COPPELIASIM_ELBOW_INDEX)
        if (
            count != 6
            or shoulder_index == elbow_index
            or not 0 <= shoulder_index < count
            or not 0 <= elbow_index < count
        ):
            raise RuntimeError("índices o cantidad de joints inválidos")

        joints = [
            sim.getObject("./joint", {"proxy": ur5_handle, "index": index})
            for index in range(count)
        ]
        if any(not self._valid_handle(handle) for handle in joints):
            raise RuntimeError("no se encontraron los seis joints del UR5")
        if len(set(joints)) != count:
            raise RuntimeError("los handles de los joints del UR5 no son únicos")

        aliases = [str(sim.getObjectAlias(handle, 2)) for handle in joints]
        if any(
            "RG2" in {part.upper() for part in alias.split("/") if part}
            for alias in aliases
        ):
            raise RuntimeError(
                "la búsqueda de joints alcanzó el subárbol RG2"
            )

        self._joint_handles = joints
        self._validate_joint_kinds_and_modes()
        self._load_joint_limits()
        self._rg2_handle = self._find_rg2(ur5_handle)
        self._home_positions = [sim.getJointPosition(handle) for handle in joints]
        for index, value in enumerate(self._home_positions):
            if not self._is_number(value):
                raise RuntimeError(f"home no finito en joint {index}")
            if not self._target_in_bounds(index, float(value)):
                raise RuntimeError(
                    f"home fuera del límite efectivo en joint {index}"
                )
        self._create_collision_collections(ur5_handle)
        self._resolve_vision_sensor()
        self._configure_vision_sensor(ur5_handle)

    def _orient_ur5_base(self, ur5_handle: int) -> None:
        """Orienta el modelo completo sin cambiar los ejes locales de joints."""
        if not math.isfinite(self._base_yaw_deg):
            raise RuntimeError("COPPELIASIM_BASE_YAW_DEG no es finito")

        sim = self._require_sim()
        get_orientation = getattr(sim, "getObjectOrientation", None)
        set_orientation = getattr(sim, "setObjectOrientation", None)
        if not callable(get_orientation) or not callable(set_orientation):
            raise RuntimeError(
                "la API no permite orientar el root /UR5 alrededor del eje vertical"
            )

        orientation = get_orientation(ur5_handle, -1)
        if (
            not isinstance(orientation, (tuple, list))
            or len(orientation) != 3
            or not all(self._is_number(value) for value in orientation)
        ):
            raise RuntimeError("orientación inválida para el root /UR5")

        target_orientation = [0.0, 0.0, math.radians(self._base_yaw_deg)]
        set_orientation(ur5_handle, -1, target_orientation)

    def _resolve_vision_sensor(self) -> None:
        """Localiza el sensor opcional sin convertirlo en requisito de seguridad."""
        if not self._view_enabled:
            self._set_simulator_frame_message(
                "Vista de CoppeliaSim deshabilitada en settings.py."
            )
            return

        sim = self._require_sim()
        path = self._view_sensor_path
        # ``noError`` convierte un objeto ausente en -1. Cualquier excepcion
        # restante es de transporte/API y debe conservar la politica FAULT.
        handle = sim.getObject(path, {"noError": True})
        if not self._valid_handle(handle):
            self._vision_sensor_handle = None
            self._set_simulator_frame_message(
                f"No se encontro el Vision Sensor {path}."
            )
            return

        self._vision_sensor_handle = int(handle)
        self._set_simulator_frame_message("Esperando imagen del simulador...")

    def _configure_vision_sensor(self, ur5_handle: int) -> None:
        """Encuadra lateralmente el robot completo y conserva Z vertical."""
        handle = self._vision_sensor_handle
        if handle is None:
            return

        camera = self._view_camera_position
        target = self._view_target_position
        forward = [target[index] - camera[index] for index in range(3)]
        forward_norm = math.sqrt(sum(value * value for value in forward))
        if forward_norm <= 1e-9:
            raise RuntimeError("la camara y su objetivo no pueden coincidir")
        forward = [value / forward_norm for value in forward]

        # El Vision Sensor mira por +Z y usa +Y como arriba. Mantener +Y tan
        # alineado como sea posible con el eje vertical del UR5 hace que el
        # movimiento del hombro/codo se lea en un plano vertical estable.
        world_up = [0.0, 0.0, 1.0]
        left = [
            world_up[1] * forward[2] - world_up[2] * forward[1],
            world_up[2] * forward[0] - world_up[0] * forward[2],
            world_up[0] * forward[1] - world_up[1] * forward[0],
        ]
        left_norm = math.sqrt(sum(value * value for value in left))
        if left_norm <= 1e-9:
            raise RuntimeError("la vista del simulador debe tener componente horizontal")
        left = [value / left_norm for value in left]
        up = [
            forward[1] * left[2] - forward[2] * left[1],
            forward[2] * left[0] - forward[0] * left[2],
            forward[0] * left[1] - forward[1] * left[0],
        ]

        matrix = [
            left[0], up[0], forward[0], camera[0],
            left[1], up[1], forward[1], camera[1],
            left[2], up[2], forward[2], camera[2],
        ]
        sim = self._require_sim()
        set_matrix = getattr(sim, "setObjectMatrix", None)
        if not callable(set_matrix):
            raise RuntimeError("la API no permite encuadrar el Vision Sensor")
        set_matrix(handle, ur5_handle, matrix)

        view_angle = math.radians(self._view_angle_deg)
        set_float_property = getattr(sim, "setFloatProperty", None)
        if callable(set_float_property):
            set_float_property(handle, "viewAngle", view_angle)
            return

        set_float_param = getattr(sim, "setObjectFloatParam", None)
        parameter = getattr(sim, "visionfloatparam_perspective_angle", None)
        if not callable(set_float_param) or parameter is None:
            raise RuntimeError("la API no permite aplicar zoom al Vision Sensor")
        set_float_param(handle, parameter, view_angle)

    def _validate_joint_kinds_and_modes(self) -> None:
        sim = self._require_sim()
        get_object_type = getattr(sim, "getObjectType", None)
        object_joint_type = getattr(
            sim,
            "sceneobject_joint",
            getattr(sim, "object_joint_type", None),
        )
        get_joint_type = getattr(sim, "getJointType", None)
        revolute_type = getattr(
            sim,
            "joint_revolute",
            getattr(sim, "joint_revolute_subtype", None),
        )
        get_joint_mode = getattr(sim, "getJointMode", None)
        kinematic_mode = getattr(sim, "jointmode_kinematic", None)
        dynamic_mode = getattr(sim, "jointmode_dynamic", None)

        if not callable(get_object_type) or object_joint_type is None:
            raise RuntimeError("no se puede confirmar el tipo de los joints")
        if not callable(get_joint_type) or revolute_type is None:
            raise RuntimeError("no se puede confirmar que los joints sean revolutos")
        if (
            not callable(get_joint_mode)
            or kinematic_mode is None
            or dynamic_mode is None
        ):
            raise RuntimeError("no se puede confirmar el modo de los joints")

        for index, handle in enumerate(self._joint_handles):
            if get_object_type(handle) != object_joint_type:
                raise RuntimeError(f"el objeto {index} no es una articulación")
            joint_type = get_joint_type(handle)
            if isinstance(joint_type, (tuple, list)):
                joint_type = joint_type[0]
            if joint_type != revolute_type:
                raise RuntimeError(f"el joint {index} no es revoluto")

            mode = get_joint_mode(handle)
            if isinstance(mode, (tuple, list)):
                mode = mode[0]
            if mode not in {kinematic_mode, dynamic_mode}:
                raise RuntimeError(
                    f"joint {index} no está en modo cinemático/dinámico"
                )
            if mode == dynamic_mode:
                self._validate_dynamic_position_control(index, handle)

    def _validate_dynamic_position_control(self, index: int, handle: int) -> None:
        sim = self._require_sim()
        expected = getattr(sim, "jointdynctrl_position", None)
        if expected is None:
            raise RuntimeError(
                f"no se puede confirmar el control dinámico del joint {index}"
            )
        try:
            control_mode = self._read_joint_int_setting(
                handle,
                property_name="dynCtrlMode",
                parameter_name="jointintparam_dynctrlmode",
            )
            profile_enabled = self._read_joint_int_setting(
                handle,
                property_name="dynPosMode",
                parameter_name="jointintparam_dynposctrltype",
            )
        except Exception as exc:
            raise RuntimeError(
                f"no se puede leer el control dinámico del joint {index}: {exc}"
            ) from exc
        if control_mode != expected:
            raise RuntimeError(
                f"joint dinámico {index} no usa control de posición"
            )
        if profile_enabled != 1:
            raise RuntimeError(
                f"joint dinámico {index} no tiene perfil de movimiento activo"
            )

    def _read_joint_int_setting(
        self,
        handle: int,
        *,
        property_name: str,
        parameter_name: str,
    ) -> int:
        """Lee una propiedad actual o su parámetro entero documentado legado."""
        sim = self._require_sim()
        property_error: Exception | None = None
        property_getter = getattr(sim, "getIntProperty", None)
        if callable(property_getter):
            try:
                value = property_getter(handle, property_name)
                if isinstance(value, int) and not isinstance(value, bool):
                    return value
                raise RuntimeError(f"{property_name} no devolvió un entero")
            except Exception as exc:
                property_error = exc

        parameter = getattr(sim, parameter_name, None)
        parameter_getter = getattr(sim, "getObjectInt32Param", None)
        if parameter is not None and callable(parameter_getter):
            value = parameter_getter(handle, parameter)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            raise RuntimeError(f"{parameter_name} no devolvió un entero")

        detail = f": {property_error}" if property_error is not None else ""
        raise RuntimeError(
            f"no se puede consultar {property_name}/{parameter_name}{detail}"
        )

    def _load_joint_limits(self) -> None:
        sim = self._require_sim()
        getter = getattr(sim, "getJointInterval", None)
        if not callable(getter):
            raise RuntimeError("no se pueden consultar los intervalos de joints")
        scene_limits: dict[int, tuple[float, float]] = {}
        effective: dict[int, tuple[float, float]] = {}
        shoulder_index = int(settings.COPPELIASIM_SHOULDER_INDEX)
        elbow_index = int(settings.COPPELIASIM_ELBOW_INDEX)

        for index, handle in enumerate(self._joint_handles):
            cyclic, interval = getter(handle)
            if cyclic:
                low, high = -math.pi, math.pi
            else:
                if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                    raise RuntimeError(f"intervalo inválido en joint {index}")
                low = float(interval[0])
                high = low + float(interval[1])
                if not all(math.isfinite(value) for value in (low, high)):
                    raise RuntimeError(f"intervalo no finito en joint {index}")
                if high < low:
                    raise RuntimeError(f"intervalo invertido en joint {index}")
            scene_limits[index] = (low, high)

            config_limit: tuple[float, float] | None = None
            if index == shoulder_index:
                config_limit = (
                    math.radians(float(settings.COPPELIASIM_SHOULDER_MIN_DEG)),
                    math.radians(float(settings.COPPELIASIM_SHOULDER_MAX_DEG)),
                )
            elif index == elbow_index:
                config_limit = (
                    math.radians(float(settings.COPPELIASIM_ELBOW_MIN_DEG)),
                    math.radians(float(settings.COPPELIASIM_ELBOW_MAX_DEG)),
                )

            if config_limit is None:
                effective[index] = (low, high)
            else:
                intersection = (
                    max(low, config_limit[0]),
                    min(high, config_limit[1]),
                )
                if intersection[0] > intersection[1]:
                    raise RuntimeError(
                        f"límite configurado no intersecta la escena en joint {index}"
                    )
                effective[index] = intersection

        self._scene_limits = scene_limits
        self._effective_limits = effective

    def _find_rg2(self, ur5_handle: int) -> int:
        sim = self._require_sim()
        tree: list[int] | None = None
        get_tree = getattr(sim, "getObjectsInTree", None)
        if callable(get_tree):
            handle_all = getattr(sim, "handle_all", -1)
            for args in (
                (ur5_handle, handle_all, 0),
                (ur5_handle, handle_all),
            ):
                try:
                    tree = list(get_tree(*args))
                    break
                except TypeError:
                    continue

        candidates: list[int] = []
        for path, options in (
            ("/UR5/RG2", None),
            ("./RG2", {"proxy": ur5_handle}),
        ):
            try:
                handle = sim.getObject(path) if options is None else sim.getObject(path, options)
            except Exception:
                continue
            if self._valid_handle(handle):
                candidates.append(handle)

        if tree is not None:
            for handle in tree:
                try:
                    alias = str(sim.getObjectAlias(handle, 2))
                except Exception:
                    alias = ""
                if alias.rstrip("/").split("/")[-1].upper() == "RG2":
                    candidates.append(handle)
            candidates = [handle for handle in candidates if handle in tree]

        if not candidates:
            raise RuntimeError("RG2 no está presente como descendiente de /UR5")
        return candidates[0]

    def _create_collision_collections(self, ur5_handle: int) -> None:
        sim = self._require_sim()
        required = (
            "createCollection",
            "addItemToCollection",
            "checkCollision",
            "getObjectsInTree",
        )
        if any(not callable(getattr(sim, name, None)) for name in required):
            raise RuntimeError("la API no soporta colecciones/colisiones")
        shape_type = getattr(
            sim,
            "sceneobject_shape",
            getattr(sim, "object_shape_type", None),
        )
        handle_single = getattr(sim, "handle_single", None)
        if shape_type is None or handle_single is None:
            raise RuntimeError("la API no expone constantes para formas")

        robot_shapes = set(
            sim.getObjectsInTree(self._joint_handles[0], shape_type, 0)
        )
        ur5_shapes = set(sim.getObjectsInTree(ur5_handle, shape_type, 0))
        scene_shapes = set(
            sim.getObjectsInTree(getattr(sim, "handle_scene"), shape_type, 0)
        )
        rg2_shapes = set(
            sim.getObjectsInTree(self._rg2_handle, shape_type, 0)
        )
        environment_shapes = scene_shapes - ur5_shapes
        all_shapes = robot_shapes | ur5_shapes | scene_shapes | rg2_shapes
        if any(not self._valid_handle(handle) for handle in all_shapes):
            raise RuntimeError("la escena devolvió handles de formas inválidos")
        if not robot_shapes:
            raise RuntimeError("el subárbol móvil del UR5 no contiene formas")
        if not environment_shapes:
            raise RuntimeError("la escena no contiene formas de entorno")
        if not robot_shapes.intersection(rg2_shapes):
            raise RuntimeError("las formas del RG2 no pertenecen al robot móvil")
        if not robot_shapes.issubset(ur5_shapes):
            raise RuntimeError("formas móviles fuera del árbol /UR5")

        # Los aliases se precargan en el preflight. La ruta crítica de una
        # colisión no puede hacer RPC opcionales antes de stopSimulation(False):
        # un timeout dejaría el socket REQ sin posibilidad de enviar la parada.
        collision_aliases: dict[int, str] = {}
        for shape in sorted(robot_shapes | environment_shapes):
            try:
                collision_aliases[shape] = str(sim.getObjectAlias(shape, 2))
            except Exception as exc:
                raise RuntimeError(
                    f"no se pudo precargar el alias de la forma {shape}: {exc}"
                ) from exc

        try:
            robot_collection = sim.createCollection(0)
            environment_collection = sim.createCollection(0)
        except TypeError:
            robot_collection = sim.createCollection()
            environment_collection = sim.createCollection()

        self._robot_collection = robot_collection
        self._environment_collection = environment_collection
        self._collision_aliases = collision_aliases
        for shape in sorted(robot_shapes):
            sim.addItemToCollection(
                robot_collection,
                handle_single,
                shape,
                0,
            )
        for shape in sorted(environment_shapes):
            sim.addItemToCollection(
                environment_collection,
                handle_single,
                shape,
                0,
            )

    def _worker_cycle(self) -> None:
        reason = self._take_estop_request()
        if reason is not None:
            self._enter_estop_worker(reason)
            return

        state = self.snapshot.state
        if state in {SafetyState.ESTOP, SafetyState.FAULT}:
            return

        if self._take_home_request():
            try:
                self._home_succeeded = self._move_home_worker()
            finally:
                self._home_done.set()
            return

        if self._take_hold_request():
            if not self._hold_worker():
                return
            self._arm_hold_sent = True

        if not self._simulation_is_running():
            self._enter_fault_worker("La simulación fue detenida fuera del controlador.")
            return
        if self._service_pending_estop_worker():
            return

        now = self._clock()
        if now - self._last_collision_check >= self._collision_interval:
            self._last_collision_check = now
            pair = self._collision_pair()
            if pair is not None:
                self._enter_collision_estop(pair)
                return
            if self._service_pending_estop_worker():
                return

        state = self.snapshot.state
        if state is not SafetyState.RUNNING:
            self._service_simulator_view_worker(now)
            return

        if self._require_arm:
            last_arm = self._last_valid_arm_at
            elapsed = math.inf if last_arm is None else now - last_arm
            if elapsed >= self._command_max_age and not self._arm_hold_sent:
                if not self._hold_worker():
                    return
                self._arm_hold_sent = True
            if elapsed >= self._pose_loss_estop:
                self._enter_estop_worker("Watchdog: brazo inválido o perdido.")
                return

        if now - self._last_command_at < self._update_interval:
            self._service_simulator_view_worker(now)
            return
        command = self._take_latest_command()
        if command is None:
            self._service_simulator_view_worker(now)
            return
        self._last_command_at = now
        if self._running_motion_cancelled_worker():
            return
        current_state = self.snapshot.state
        report = self._validate_command(command, now, current_state)
        if report.arm.accepted or report.gripper.accepted:
            # La vigilancia periódica no sustituye esta barrera: cada comando
            # se contrasta con el entorno inmediatamente antes de actuar.
            pair = self._collision_pair()
            self._last_collision_check = self._clock()
            if pair is not None:
                self._enter_collision_estop(pair)
                return
            if self._service_pending_estop_worker():
                return
        sent = False
        if report.arm.accepted:
            shoulder_deg, elbow_deg = report.arm.targets
            if self._running_motion_cancelled_worker():
                return
            self._set_joint_target(
                int(settings.COPPELIASIM_SHOULDER_INDEX),
                math.radians(shoulder_deg),
            )
            if self._running_motion_cancelled_worker():
                return
            self._set_joint_target(
                int(settings.COPPELIASIM_ELBOW_INDEX),
                math.radians(elbow_deg),
            )
            if self._running_motion_cancelled_worker():
                return
            self._last_valid_arm_at = now
            self._arm_hold_sent = False
            sent = True
        elif self._require_arm and not self._arm_hold_sent:
            if not self._hold_worker():
                return
            self._arm_hold_sent = True

        if report.gripper.accepted:
            if self._running_motion_cancelled_worker():
                return
            gripper_target = int(report.gripper.targets[0])
            if gripper_target != self._last_gripper_command:
                sim = self._require_sim()
                sim.setIntProperty(
                    sim.handle_scene,
                    self.GRIPPER_SIGNAL,
                    gripper_target,
                )
                if self._running_motion_cancelled_worker():
                    return
                self._last_gripper_command = gripper_target
                sent = True

        if sent:
            pair = self._collision_pair()
            if pair is not None:
                self._enter_collision_estop(pair)
                return
            else:
                self._service_pending_estop_worker()

        if self.snapshot.state not in {SafetyState.ESTOP, SafetyState.FAULT}:
            self._service_simulator_view_worker(self._clock())

    def _service_simulator_view_worker(self, now: float) -> None:
        """Captura al final del ciclo, despues de todas las barreras de seguridad."""
        handle = self._vision_sensor_handle
        if handle is None or now - self._last_view_capture < self._view_interval:
            return
        self._last_view_capture = now

        sim = self._require_sim()
        image, resolution = sim.getVisionSensorImg(handle)
        if (
            not isinstance(resolution, (tuple, list))
            or len(resolution) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in resolution
            )
        ):
            self._set_simulator_frame_message(
                "El Vision Sensor devolvio una resolucion invalida.",
                clear=True,
            )
            return

        width, height = int(resolution[0]), int(resolution[1])
        expected = (self._view_width, self._view_height)
        if (width, height) != expected:
            self._set_simulator_frame_message(
                "Resolucion inesperada del Vision Sensor: "
                f"{width}x{height}; se esperaba {expected[0]}x{expected[1]}.",
                clear=True,
            )
            return

        try:
            source = bytes(image)
        except (TypeError, ValueError):
            self._set_simulator_frame_message(
                "El Vision Sensor devolvio un buffer de imagen invalido.",
                clear=True,
            )
            return
        row_size = width * 3
        if len(source) != row_size * height:
            self._set_simulator_frame_message(
                "El Vision Sensor devolvio un buffer de tamano invalido.",
                clear=True,
            )
            return

        # CoppeliaSim entrega la primera fila desde la parte inferior.
        view = memoryview(source)
        rgb = b"".join(
            view[row * row_size : (row + 1) * row_size]
            for row in range(height - 1, -1, -1)
        )
        with self._lock:
            sequence = self._simulator_frame.sequence + 1
            self._simulator_frame = SimulatorFrameSnapshot(
                rgb=rgb,
                width=width,
                height=height,
                sequence=sequence,
                captured_at=now,
                message="Vista del simulador activa.",
            )

    def _set_simulator_frame_message(self, message: str, *, clear: bool = False) -> None:
        with self._lock:
            current = self._simulator_frame
            self._simulator_frame = SimulatorFrameSnapshot(
                rgb=None if clear else current.rgb,
                width=0 if clear else current.width,
                height=0 if clear else current.height,
                sequence=current.sequence,
                captured_at=current.captured_at,
                message=message,
            )

    def _validate_command_locked(
        self,
        command: MimicCommand,
        now: float,
    ) -> ValidationReport:
        return self._validate_command(command, now, self._snapshot.state)

    def _validate_command(
        self,
        command: MimicCommand,
        now: float,
        state: SafetyState,
    ) -> ValidationReport:
        if state is not SafetyState.RUNNING:
            reason = f"controlador no está RUNNING ({state.value})"
            return self._reject_both(reason)
        if not isinstance(command, MimicCommand):
            return self._reject_both("se esperaba MimicCommand")
        if (
            isinstance(command.sequence, bool)
            or not isinstance(command.sequence, int)
            or command.sequence < 0
        ):
            return self._reject_both("sequence debe ser entero no negativo")
        if not self._is_number(command.created_at):
            return self._reject_both("created_at debe ser real y finito")
        age = now - float(command.created_at)
        if age > self._command_max_age:
            return self._reject_both("comando caducado")
        if age < -self._command_max_age:
            return self._reject_both("created_at está en el futuro")

        arm = self._validate_arm(command)
        gripper = self._validate_gripper(command)
        return ValidationReport(arm=arm, gripper=gripper)

    def _validate_arm(self, command: MimicCommand) -> ChannelValidation:
        if command.arm_error is not None:
            return ChannelValidation(False, str(command.arm_error))
        if command.arm is None:
            return ChannelValidation(False, "brazo no disponible")
        if not isinstance(command.arm, ArmCommand):
            return ChannelValidation(False, "arm debe ser ArmCommand")
        values = (command.arm.shoulder_deg, command.arm.elbow_deg)
        if not all(self._is_number(value) for value in values):
            return ChannelValidation(False, "ángulos del brazo no finitos")
        shoulder, elbow = (float(value) for value in values)
        if not -180.0 <= shoulder <= 180.0:
            return ChannelValidation(False, "hombro humano fuera de [-180, 180]")
        if not 0.0 <= elbow <= 180.0:
            return ChannelValidation(False, "codo humano fuera de [0, 180]")

        targets = self.map_arm_angles(shoulder, elbow)
        for index, target_deg in zip(
            (
                int(settings.COPPELIASIM_SHOULDER_INDEX),
                int(settings.COPPELIASIM_ELBOW_INDEX),
            ),
            targets,
        ):
            if not self._target_in_bounds(index, math.radians(target_deg)):
                return ChannelValidation(
                    False,
                    f"objetivo del joint {index} fuera del límite efectivo",
                )
        return ChannelValidation(True, targets=tuple(float(x) for x in targets))

    def _validate_gripper(self, command: MimicCommand) -> ChannelValidation:
        if command.gripper_error is not None:
            return ChannelValidation(False, str(command.gripper_error))
        if command.gripper is None:
            return ChannelValidation(False, "gripper no disponible")
        if not isinstance(command.gripper, GripperCommand):
            return ChannelValidation(False, "gripper debe ser GripperCommand")
        aperture = command.gripper.aperture
        if not self._is_number(aperture):
            return ChannelValidation(False, "apertura del gripper no finita")
        aperture = float(aperture)
        if not 0.0 <= aperture <= 1.0:
            return ChannelValidation(False, "apertura fuera de [0, 1]")
        target = float(aperture >= float(settings.COPPELIASIM_GRIPPER_THRESHOLD))
        return ChannelValidation(True, targets=(target,))

    @staticmethod
    def _reject_both(reason: str) -> ValidationReport:
        rejected = ChannelValidation(False, reason)
        return ValidationReport(arm=rejected, gripper=rejected)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    def _target_in_bounds(self, index: int, target: float) -> bool:
        bounds = self._effective_limits.get(index)
        if bounds is None:
            return False
        return bounds[0] - 1e-9 <= target <= bounds[1] + 1e-9

    def _set_joint_target(self, index: int, target: float) -> None:
        sim = self._require_sim()
        sim.setJointTargetPosition(
            self._joint_handles[index],
            target,
            self._motion_params,
        )

    def _hold_worker(self) -> bool:
        sim = self._require_sim()
        positions: list[float] = []
        for handle in self._joint_handles:
            positions.append(sim.getJointPosition(handle))
            if self._service_pending_estop_worker():
                return False
        for index, position in enumerate(positions):
            if not self._is_number(position) or not self._target_in_bounds(index, float(position)):
                raise RuntimeError(f"posición inválida al hacer hold en joint {index}")
        for handle, position in zip(self._joint_handles, positions):
            sim.setJointTargetPosition(handle, float(position), self._motion_params)
            if self._service_pending_estop_worker():
                return False
        return True

    def _move_home_worker(self) -> bool:
        if self.snapshot.state in {SafetyState.ESTOP, SafetyState.FAULT}:
            return False
        sim = self._require_sim()
        if not self._simulation_is_running():
            self._set_state(
                SafetyState.FAULT,
                False,
                "Home cancelado: la simulación está detenida.",
            )
            return False
        if len(self._home_positions) != len(self._joint_handles):
            self._enter_fault_worker("Home rechazado: readback inicial incompleto.")
            return False
        for index, target in enumerate(self._home_positions):
            if not self._is_number(target) or not self._target_in_bounds(index, float(target)):
                self._enter_fault_worker(
                    f"Home rechazado por límite efectivo en joint {index}.",
                )
                return False
        pair = self._collision_pair()
        if pair is not None:
            self._enter_collision_estop(pair)
            return False
        if self._service_pending_estop_worker():
            return False

        for handle, target in zip(self._joint_handles, self._home_positions):
            sim.setJointTargetPosition(handle, float(target), self._motion_params)
            if self._service_pending_estop_worker():
                return False

        deadline = self._clock() + self._home_timeout
        wall_deadline = time.monotonic() + self._home_timeout
        while not self._stop_event.is_set():
            reason = self._take_estop_request()
            if reason is not None:
                self._enter_estop_worker(reason)
                return False
            pair = self._collision_pair()
            if pair is not None:
                self._enter_collision_estop(pair)
                return False
            if self._service_pending_estop_worker():
                return False
            if not self._simulation_is_running():
                self._set_state(
                    SafetyState.FAULT,
                    False,
                    "Home cancelado: la simulación se detuvo.",
                )
                return False
            positions = [sim.getJointPosition(handle) for handle in self._joint_handles]
            at_home = all(
                self._is_number(value)
                and abs(float(value) - target) <= self._home_tolerance
                for value, target in zip(positions, self._home_positions)
            )
            if at_home:
                # El movimiento puede entrar en contacto durante los últimos
                # readbacks. No declarar home hasta cerrar esa ventana.
                pair = self._collision_pair()
                self._last_collision_check = self._clock()
                if pair is not None:
                    self._enter_collision_estop(pair)
                    return False
                if self._service_pending_estop_worker():
                    return False
                self._set_state(
                    self.snapshot.state,
                    True,
                    "Retorno a home verificado.",
                )
                return True
            if self._clock() >= deadline or time.monotonic() >= wall_deadline:
                self._enter_fault_worker(
                    "Timeout esperando readback de home.",
                )
                return False
            self._wake_event.wait(min(self._collision_interval, 0.05))
            self._wake_event.clear()
        return False

    def _collision_pair(self) -> tuple[Any, Any] | None:
        sim = self._require_sim()
        if self._robot_collection is None or self._environment_collection is None:
            raise RuntimeError("colecciones de colisión no inicializadas")
        response = sim.checkCollision(
            self._robot_collection,
            self._environment_collection,
        )
        if isinstance(response, (tuple, list)):
            result = int(response[0])
            handles = response[1] if len(response) > 1 else None
        else:
            result = int(response)
            handles = None
        if result == 0:
            return None
        if isinstance(handles, (tuple, list)) and len(handles) >= 2:
            return (handles[0], handles[1])
        return ("colección robot", "colección entorno")

    def _format_collision_pair(self, pair: tuple[Any, Any]) -> tuple[str, str]:
        first, second = pair

        def format_handle(handle: Any) -> str:
            if isinstance(handle, int) and not isinstance(handle, bool):
                return self._collision_aliases.get(handle, str(handle))
            return str(handle)

        return format_handle(first), format_handle(second)

    def _enter_collision_estop(self, raw_pair: tuple[Any, Any]) -> None:
        # Enclavar, vaciar y detener preceden incluso al formateo local de los
        # aliases. No se ejecuta ninguna RPC entre checkCollision y la parada.
        self._set_state(
            SafetyState.ESTOP,
            True,
            "Colisión detectada; deteniendo simulación.",
            collision_pair=None,
        )
        with self._lock:
            self._latest_command = None
        try:
            self._stop_simulation_once()
        except Exception as exc:
            pair = self._format_collision_pair(raw_pair)
            self._set_state(
                SafetyState.FAULT,
                False,
                f"La colisión no pudo detener la simulación: {exc}",
                collision_pair=pair,
            )
            return
        pair = self._format_collision_pair(raw_pair)
        self._set_state(
            SafetyState.ESTOP,
            True,
            f"Colisión detectada: {pair[0]} <-> {pair[1]}",
            collision_pair=pair,
        )

    def _enter_estop_worker(self, reason: str) -> None:
        self._set_state(
            SafetyState.ESTOP,
            self._sim is not None,
            reason,
            collision_pair=self.snapshot.collision_pair,
        )
        with self._lock:
            self._latest_command = None
        try:
            self._stop_simulation_once()
        except Exception as exc:
            self._set_state(
                SafetyState.FAULT,
                False,
                f"ESTOP no pudo detener la simulación: {exc}",
            )

    def _enter_fault_worker(self, message: str) -> None:
        self._set_state(
            SafetyState.FAULT,
            False,
            message,
            collision_pair=self.snapshot.collision_pair,
        )
        with self._lock:
            self._latest_command = None
        try:
            self._stop_simulation_once()
        except Exception as exc:
            self._set_state(
                SafetyState.FAULT,
                False,
                f"{message} Parada no confirmada: {exc}",
                collision_pair=self.snapshot.collision_pair,
            )

    def _stop_simulation_once(self) -> None:
        with self._lock:
            if self._simulation_stop_succeeded:
                return
            if self._simulation_stop_attempted:
                detail = self._simulation_stop_error or "resultado desconocido"
                raise RuntimeError(f"parada anterior no confirmada: {detail}")
            self._simulation_stop_attempted = True
        try:
            self._require_sim().stopSimulation(False)
        except Exception as exc:
            with self._lock:
                self._simulation_stop_error = str(exc)
            raise
        else:
            with self._lock:
                self._simulation_stop_succeeded = True

    def _simulation_is_running(self) -> bool:
        sim = self._require_sim()
        getter = getattr(sim, "getSimulationState", None)
        stopped = getattr(sim, "simulation_stopped", None)
        if not callable(getter) or stopped is None:
            raise RuntimeError("no se puede confirmar el estado de simulación")
        state = getter()
        paused = getattr(sim, "simulation_paused", None)
        if state == stopped or (paused is not None and state == paused):
            return False

        advancing_names = (
            "simulation_advancing_firstafterstop",
            "simulation_advancing_running",
            "simulation_advancing_firstafterpause",
        )
        advancing = {
            getattr(sim, name)
            for name in advancing_names
            if getattr(sim, name, None) is not None
        }
        # APIs antiguas pueden no publicar las subfases, pero sí stopped.
        return state in advancing if advancing else state != stopped

    def _take_latest_command(self) -> MimicCommand | None:
        with self._lock:
            command = self._latest_command
            self._latest_command = None
            return command

    def _take_hold_request(self) -> bool:
        with self._lock:
            requested = self._hold_requested
            self._hold_requested = False
            return requested

    def _take_home_request(self) -> bool:
        with self._lock:
            requested = self._home_requested
            self._home_requested = False
            return requested

    def _take_estop_request(self) -> str | None:
        with self._lock:
            reason = self._estop_request
            self._estop_request = None
            return reason

    def _service_pending_estop_worker(self) -> bool:
        reason = self._take_estop_request()
        if reason is None:
            return False
        self._enter_estop_worker(reason)
        return True

    def _running_motion_cancelled_worker(self) -> bool:
        if self._service_pending_estop_worker():
            return True
        with self._lock:
            hold_pending = self._hold_requested
            state = self._snapshot.state
        return hold_pending or state is not SafetyState.RUNNING

    def _destroy_collections(self) -> None:
        sim = self._sim
        if sim is None:
            return
        destroy = getattr(sim, "destroyCollection", None)
        if callable(destroy):
            for handle in (self._robot_collection, self._environment_collection):
                if handle is not None:
                    try:
                        destroy(handle)
                    except Exception:
                        pass
        self._robot_collection = None
        self._environment_collection = None
        self._collision_aliases = {}

    def _require_sim(self) -> Any:
        if self._sim is None:
            raise RuntimeError("cliente CoppeliaSim no inicializado")
        return self._sim

    @staticmethod
    def _valid_handle(handle: Any) -> bool:
        return isinstance(handle, int) and not isinstance(handle, bool) and handle >= 0

    def _set_state(
        self,
        state: SafetyState,
        connected: bool,
        message: str,
        collision_pair: tuple[str, str] | None | object = ...,
    ) -> None:
        with self._lock:
            if collision_pair is ...:
                pair = self._snapshot.collision_pair
            else:
                pair = collision_pair
            self._set_state_locked(state, connected, message, pair)

    def _set_state_locked(
        self,
        state: SafetyState,
        connected: bool,
        message: str,
        collision_pair: tuple[str, str] | None,
    ) -> None:
        if state in {SafetyState.ESTOP, SafetyState.FAULT}:
            self._home_forbidden = True
        self._snapshot = SafetySnapshot(
            state=state,
            connected=connected,
            message=message,
            collision_pair=collision_pair,
        )

    def __enter__(self) -> CoppeliaRobot:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close(normal_exit=exc_type is None)
        return False
