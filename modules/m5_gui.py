"""M5: dashboard Tkinter y runtime no bloqueante de Robot Mimic."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from enum import Enum, auto
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable

from config.settings import CAMERA_MAX_CONSECUTIVE_FAILURES, FPS_SMOOTHING_WINDOW
from modules.commands import MimicCommand, ValidationReport
from modules.m1_capture import CaptureModule
from modules.m2_mediapipe import PoseDetector
from modules.m3_angles import AngleCalculator
from modules.m4_coppeliasim import (
    CoppeliaRobot,
    SafetySnapshot,
    SafetyState,
    SimulatorFrameSnapshot,
)


BACKGROUND = "#0b1220"
SURFACE = "#111c2e"
SURFACE_ALT = "#17243a"
BORDER = "#263750"
TEXT = "#e7edf6"
MUTED = "#8fa2bb"
GREEN = "#34d399"
YELLOW = "#fbbf24"
ORANGE = "#fb923c"
RED = "#f05252"
BLUE = "#38bdf8"
GRAY = "#64748b"


class RuntimeAction(Enum):
    RECONNECT = auto()


@dataclass(frozen=True)
class CameraFrameSnapshot:
    rgb: bytes | None
    width: int
    height: int
    sequence: int


@dataclass(frozen=True)
class RuntimeSnapshot:
    safety: SafetySnapshot
    simulator: SimulatorFrameSnapshot
    camera: CameraFrameSnapshot
    camera_available: bool
    camera_fatal: bool
    fps: float
    command: MimicCommand | None
    report: ValidationReport | None
    message: str


@dataclass(frozen=True)
class PresentationState:
    badge_text: str
    badge_color: str
    toggle_text: str
    toggle_enabled: bool
    reconnect_enabled: bool
    estop_enabled: bool


def presentation_for(snapshot: RuntimeSnapshot, *, closing: bool = False) -> PresentationState:
    """Convierte el estado funcional en propiedades puras de la interfaz."""
    state = snapshot.safety.state
    colors = {
        SafetyState.DISCONNECTED: GRAY,
        SafetyState.READY: YELLOW,
        SafetyState.RUNNING: GREEN,
        SafetyState.PAUSED: ORANGE,
        SafetyState.ESTOP: RED,
        SafetyState.FAULT: RED,
    }
    names = {
        SafetyState.DISCONNECTED: "DESCONECTADO",
        SafetyState.READY: "LISTO",
        SafetyState.RUNNING: "IMITANDO",
        SafetyState.PAUSED: "PAUSADO",
        SafetyState.ESTOP: "EMERGENCIA",
        SafetyState.FAULT: "FALLO",
    }
    toggle_text = "Pausar" if state is SafetyState.RUNNING else "Iniciar"
    can_start = (
        state in {SafetyState.READY, SafetyState.PAUSED}
        and snapshot.camera_available
        and not snapshot.camera_fatal
    )
    return PresentationState(
        badge_text="CERRANDO" if closing else names[state],
        badge_color=GRAY if closing else colors[state],
        toggle_text=toggle_text,
        toggle_enabled=not closing and (state is SafetyState.RUNNING or can_start),
        reconnect_enabled=(
            not closing
            and not snapshot.camera_fatal
            and state in {SafetyState.DISCONNECTED, SafetyState.ESTOP, SafetyState.FAULT}
        ),
        estop_enabled=not closing and state is not SafetyState.ESTOP,
    )


def metric_values(snapshot: RuntimeSnapshot) -> tuple[str, str, str]:
    command = snapshot.command
    shoulder = "--"
    elbow = "--"
    gripper = "--"
    if command is not None and command.arm is not None:
        shoulder = f"{command.arm.shoulder_deg:.1f}°"
        elbow = f"{command.arm.elbow_deg:.1f}°"
    if command is not None and command.gripper is not None:
        gripper = f"{command.gripper.aperture * 100:.0f}%"
    return shoulder, elbow, gripper


class MimicRuntime:
    """Posee captura/MediaPipe y publica snapshots latest-only para Tkinter."""

    def __init__(
        self,
        *,
        robot_factory: Callable[[], CoppeliaRobot] = CoppeliaRobot,
        capture_factory: Callable[[], CaptureModule] = CaptureModule,
        detector_factory: Callable[[], PoseDetector] = PoseDetector,
        calculator_factory: Callable[[], AngleCalculator] = AngleCalculator,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._robot = robot_factory()
        self._capture_factory = capture_factory
        self._detector_factory = detector_factory
        self._calculator_factory = calculator_factory
        self._clock = clock
        self._lock = threading.RLock()
        self._actions: queue.SimpleQueue[RuntimeAction] = queue.SimpleQueue()
        self._stop_event = threading.Event()
        self._reset_buffers = threading.Event()
        self._done_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._unsafe_session = False
        self._camera_fatal = False
        self._normal_exit = False
        self._connection_lock = threading.Lock()
        self._connection_thread: threading.Thread | None = None
        self._camera_sequence = 0
        self._snapshot = RuntimeSnapshot(
            safety=self._robot.snapshot,
            simulator=self._robot.simulator_frame,
            camera=CameraFrameSnapshot(None, 0, 0, 0),
            camera_available=False,
            camera_fatal=False,
            fps=0.0,
            command=None,
            report=None,
            message="Inicializando cámara y MediaPipe...",
        )

    @property
    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def done(self) -> bool:
        return self._done_event.is_set()

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._run,
            name="robot-mimic-vision-runtime",
            daemon=True,
        )
        self._worker.start()

    def toggle_imitation(self) -> bool:
        snapshot = self.snapshot
        if self._camera_fatal:
            return False
        state = snapshot.safety.state
        if state is SafetyState.RUNNING:
            succeeded = self._robot.pause()
        elif snapshot.camera_available and state in {SafetyState.READY, SafetyState.PAUSED}:
            succeeded = self._robot.start_imitation(require_arm=True)
        else:
            return False
        if succeeded:
            self._reset_buffers.set()
            self._refresh_control_snapshot(self._robot.snapshot.message)
        return succeeded

    def emergency_stop(self, reason: str = "ESTOP solicitado por el usuario.") -> None:
        self._unsafe_session = True
        self._reset_buffers.set()
        self._robot.emergency_stop(reason)
        self._refresh_control_snapshot(reason)

    def reconnect(self) -> bool:
        snapshot = self.snapshot
        if self._camera_fatal or snapshot.safety.state not in {
            SafetyState.DISCONNECTED,
            SafetyState.ESTOP,
            SafetyState.FAULT,
        }:
            return False
        self._reset_buffers.set()
        self._actions.put(RuntimeAction.RECONNECT)
        return True

    def request_close(self) -> None:
        if self._stop_event.is_set():
            return
        state = self._robot.snapshot.state
        self._normal_exit = not self._unsafe_session and state not in {
            SafetyState.ESTOP,
            SafetyState.FAULT,
        }
        self._stop_event.set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done_event.wait(timeout)

    def _run(self) -> None:
        frame_durations: deque[float] = deque(maxlen=FPS_SMOOTHING_WINDOW)
        previous_time = self._clock()
        consecutive_failures = 0
        calculator: AngleCalculator | None = None
        try:
            with self._capture_factory() as camera, self._detector_factory() as detector:
                calculator = self._calculator_factory()
                self._launch_connection(reconnect=False)
                while not self._stop_event.is_set():
                    self._drain_actions()
                    if self._reset_buffers.is_set():
                        calculator.reset_buffers()
                        self._reset_buffers.clear()

                    ok, bgr_frame, rgb_frame = camera.read()
                    if not ok or bgr_frame is None or rgb_frame is None:
                        consecutive_failures += 1
                        calculator.reset_buffers()
                        if consecutive_failures == 1:
                            self._robot.pause()
                        message = (
                            "Lectura de cámara perdida; hold solicitado."
                            if consecutive_failures < CAMERA_MAX_CONSECUTIVE_FAILURES
                            else "Fallo fatal de cámara; reinicia la aplicación."
                        )
                        if consecutive_failures >= CAMERA_MAX_CONSECUTIVE_FAILURES:
                            self._camera_fatal = True
                            self._unsafe_session = True
                            self._robot.emergency_stop(
                                "Fallo fatal: "
                                f"{CAMERA_MAX_CONSECUTIVE_FAILURES} lecturas "
                                "consecutivas de camara."
                            )
                        self._publish(
                            camera=None,
                            camera_available=False,
                            fps=self.snapshot.fps,
                            command=None,
                            report=None,
                            message=message,
                        )
                        if self._camera_fatal:
                            while not self._stop_event.wait(0.05):
                                self._drain_actions()
                            break
                        continue

                    recovered = consecutive_failures > 0
                    consecutive_failures = 0
                    safety = self._robot.snapshot
                    if safety.state in {SafetyState.ESTOP, SafetyState.FAULT}:
                        self._unsafe_session = True
                    running = safety.state is SafetyState.RUNNING
                    result = detector.process(rgb_frame, process_hands=running)
                    command = calculator.compute(result) if running else None
                    report = self._robot.submit(command) if command is not None else None
                    annotated = detector.draw_skeleton(bgr_frame, result)
                    displayed = annotated[:, ::-1]
                    display_rgb = displayed[:, :, ::-1].copy()
                    height, width = display_rgb.shape[:2]
                    self._camera_sequence += 1
                    camera_snapshot = CameraFrameSnapshot(
                        rgb=display_rgb.tobytes(),
                        width=int(width),
                        height=int(height),
                        sequence=self._camera_sequence,
                    )

                    current_time = self._clock()
                    frame_durations.append(current_time - previous_time)
                    previous_time = current_time
                    duration = sum(frame_durations)
                    fps = len(frame_durations) / duration if duration > 0.0 else 0.0
                    self._publish(
                        camera=camera_snapshot,
                        camera_available=True,
                        fps=fps,
                        command=command,
                        report=report,
                        message=(
                            "Cámara recuperada; pulsa Iniciar para continuar."
                            if recovered
                            else self._robot.snapshot.message
                        ),
                    )
        except Exception as exc:
            self._unsafe_session = True
            self._robot.emergency_stop(f"Fallo de aplicación: {exc}")
            self._publish(
                camera=None,
                camera_available=False,
                fps=self.snapshot.fps,
                command=None,
                report=None,
                message=f"Fallo del runtime: {exc}",
            )
        finally:
            connector = self._connection_thread
            if connector is not None and connector.is_alive():
                connector.join(0.75)
            self._robot.close(normal_exit=self._normal_exit and not self._unsafe_session)
            self._publish(
                camera=None,
                camera_available=False,
                fps=self.snapshot.fps,
                command=None,
                report=None,
                message=self.snapshot.message,
            )
            self._done_event.set()

    def _drain_actions(self) -> None:
        while True:
            try:
                action = self._actions.get_nowait()
            except queue.Empty:
                return
            if action is RuntimeAction.RECONNECT and not self._camera_fatal:
                self._launch_connection(reconnect=True)

    def _launch_connection(self, *, reconnect: bool) -> None:
        with self._connection_lock:
            current = self._connection_thread
            if current is not None and current.is_alive():
                return

            operation = self._robot.reconnect if reconnect else self._robot.connect
            thread = threading.Thread(
                target=operation,
                name="robot-mimic-coppeliasim-connect",
                daemon=True,
            )
            self._connection_thread = thread
            thread.start()

    def _publish(
        self,
        *,
        camera: CameraFrameSnapshot | None,
        camera_available: bool,
        fps: float,
        command: MimicCommand | None,
        report: ValidationReport | None,
        message: str,
    ) -> None:
        with self._lock:
            current_camera = camera or CameraFrameSnapshot(
                None,
                0,
                0,
                self._snapshot.camera.sequence,
            )
            self._snapshot = RuntimeSnapshot(
                safety=self._robot.snapshot,
                simulator=self._robot.simulator_frame,
                camera=current_camera,
                camera_available=camera_available,
                camera_fatal=self._camera_fatal,
                fps=float(fps),
                command=command,
                report=report,
                message=str(message),
            )

    def _refresh_control_snapshot(self, message: str) -> None:
        """Hace visibles de inmediato las acciones que no realizan RPC."""
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                safety=self._robot.snapshot,
                simulator=self._robot.simulator_frame,
                message=str(message),
            )


class RobotMimicApp:
    """Dashboard oscuro; todas las operaciones de widgets viven en Tk."""

    REFRESH_MS = 33

    def __init__(self, root: tk.Tk, runtime: MimicRuntime) -> None:
        self.root = root
        self.runtime = runtime
        self._closing = False
        self._camera_photo = None
        self._simulator_photo = None
        self._camera_render_key: tuple[int, int, int] | None = None
        self._simulator_render_key: tuple[int, int, int] | None = None
        self._build_window()
        self._bind_actions()
        self.runtime.start()
        self.root.after(self.REFRESH_MS, self._refresh)

    def _build_window(self) -> None:
        self.root.title("Robot Mimic · UR5 + RG2")
        self.root.configure(bg=BACKGROUND)
        self.root.geometry("1440x900")
        self.root.minsize(1024, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Root.TFrame", background=BACKGROUND)
        style.configure("Card.TFrame", background=SURFACE, relief="flat")
        style.configure("Title.TLabel", background=BACKGROUND, foreground=TEXT, font=("Segoe UI Semibold", 19))
        style.configure("Subtitle.TLabel", background=BACKGROUND, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 11))
        style.configure("MetricName.TLabel", background=SURFACE_ALT, foreground=MUTED, font=("Segoe UI", 8))
        style.configure("MetricValue.TLabel", background=SURFACE_ALT, foreground=TEXT, font=("Segoe UI Semibold", 14))
        shell = ttk.Frame(self.root, style="Root.TFrame", padding=(16, 10))
        shell.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="ROBOT MIMIC", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Imitación visual segura · UR5 + RG2", style="Subtitle.TLabel").grid(row=1, column=0, sticky="w")
        self.badge = tk.Label(header, text="INICIALIZANDO", bg=GRAY, fg="white", font=("Segoe UI Semibold", 9), padx=12, pady=5)
        self.badge.grid(row=0, column=1, rowspan=2, sticky="e")

        views = ttk.Frame(shell, style="Root.TFrame")
        views.grid(row=1, column=0, sticky="nsew")
        views.columnconfigure(0, weight=1, uniform="view")
        views.columnconfigure(1, weight=1, uniform="view")
        views.rowconfigure(0, weight=1)
        self.camera_label = self._make_view(views, 0, "Cámara humana", "Inicializando cámara...")
        self.simulator_label = self._make_view(views, 1, "Simulación UR5", "Esperando Vision Sensor...")

        metrics = ttk.Frame(shell, style="Root.TFrame")
        metrics.grid(row=2, column=0, sticky="ew", pady=(8, 6))
        for column in range(5):
            metrics.columnconfigure(column, weight=1, uniform="metric")
        self.fps_value = self._make_metric(metrics, 0, "FPS CÁMARA")
        self.shoulder_value = self._make_metric(metrics, 1, "HOMBRO")
        self.elbow_value = self._make_metric(metrics, 2, "CODO")
        self.gripper_value = self._make_metric(metrics, 3, "GRIPPER")
        self.connection_value = self._make_metric(metrics, 4, "CONEXIÓN")

        status_card = tk.Frame(shell, bg=SURFACE_ALT, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=5)
        status_card.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        status_card.columnconfigure(0, weight=3)
        status_card.columnconfigure(1, weight=2)
        self.status_label = tk.Label(status_card, text="Inicializando...", bg=SURFACE_ALT, fg=TEXT, anchor="w", justify="left", font=("Segoe UI", 9))
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.sim_status_label = tk.Label(status_card, text="", bg=SURFACE_ALT, fg=MUTED, anchor="e", justify="right", font=("Segoe UI", 8))
        self.sim_status_label.grid(row=0, column=1, sticky="ew", padx=(12, 0))

        controls = ttk.Frame(shell, style="Root.TFrame")
        controls.grid(row=4, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, text="ESPACIO iniciar/pausar  ·  E emergencia  ·  R rearmar  ·  Q salir", style="Subtitle.TLabel").grid(row=0, column=0, sticky="e")

    def _make_view(self, parent: ttk.Frame, column: int, title: str, placeholder: str) -> tk.Label:
        card = tk.Frame(parent, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=0, column=column, sticky="nsew", padx=(0, 8) if column == 0 else (8, 0))
        card.rowconfigure(1, weight=1)
        card.columnconfigure(0, weight=1)
        tk.Label(card, text=title, bg=SURFACE, fg=TEXT, anchor="w", font=("Segoe UI Semibold", 11), padx=12, pady=6).grid(row=0, column=0, sticky="ew")
        label = tk.Label(card, text=placeholder, bg="#050a12", fg=MUTED, font=("Segoe UI", 10), compound="center")
        label.grid(row=1, column=0, sticky="nsew")
        return label

    def _make_metric(self, parent: ttk.Frame, column: int, title: str) -> tk.Label:
        card = tk.Frame(parent, bg=SURFACE_ALT, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=4)
        card.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0 if column == 4 else 5))
        tk.Label(card, text=title, bg=SURFACE_ALT, fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
        value = tk.Label(card, text="--", bg=SURFACE_ALT, fg=TEXT, font=("Segoe UI Semibold", 14))
        value.pack(anchor="w")
        return value

    def _bind_actions(self) -> None:
        self.root.bind("<space>", lambda _event: self._on_toggle())
        self.root.bind("<KeyPress-e>", lambda _event: self._on_estop())
        self.root.bind("<KeyPress-E>", lambda _event: self._on_estop())
        self.root.bind("<KeyPress-r>", lambda _event: self._on_reconnect())
        self.root.bind("<KeyPress-R>", lambda _event: self._on_reconnect())
        self.root.bind("<KeyPress-q>", lambda _event: self._on_close())
        self.root.bind("<KeyPress-Q>", lambda _event: self._on_close())

    def _on_toggle(self) -> None:
        if not self._closing:
            self.runtime.toggle_imitation()

    def _on_estop(self) -> None:
        if not self._closing:
            self.runtime.emergency_stop()

    def _on_reconnect(self) -> None:
        if not self._closing:
            self.runtime.reconnect()

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.runtime.request_close()

    def _refresh(self) -> None:
        snapshot = self.runtime.snapshot
        presentation = presentation_for(snapshot, closing=self._closing)
        self.badge.configure(text=presentation.badge_text, bg=presentation.badge_color)

        shoulder, elbow, gripper = metric_values(snapshot)
        self.fps_value.configure(text=f"{snapshot.fps:.1f}")
        self.shoulder_value.configure(text=shoulder)
        self.elbow_value.configure(text=elbow)
        self.gripper_value.configure(text=gripper)
        self.connection_value.configure(text="ONLINE" if snapshot.safety.connected else "OFFLINE", fg=GREEN if snapshot.safety.connected else MUTED)
        detail = snapshot.message or snapshot.safety.message
        if snapshot.safety.collision_pair:
            detail += f"  ·  Colisión: {snapshot.safety.collision_pair[0]} ↔ {snapshot.safety.collision_pair[1]}"
        self.status_label.configure(text=detail)
        self.sim_status_label.configure(text=snapshot.simulator.message)

        self._render_camera(snapshot.camera, snapshot.camera_available)
        self._render_simulator(snapshot.simulator)
        if self._closing and self.runtime.done:
            self.root.destroy()
            return
        self.root.after(self.REFRESH_MS, self._refresh)

    def _render_camera(self, frame: CameraFrameSnapshot, available: bool) -> None:
        if not available or frame.rgb is None:
            self.camera_label.configure(image="", text="Cámara no disponible")
            self._camera_photo = None
            self._camera_render_key = None
            return
        key = (frame.sequence, self.camera_label.winfo_width(), self.camera_label.winfo_height())
        if key != self._camera_render_key:
            self._camera_photo = self._photo_for(frame.rgb, frame.width, frame.height, key[1], key[2])
            self.camera_label.configure(image=self._camera_photo, text="")
            self._camera_render_key = key

    def _render_simulator(self, frame: SimulatorFrameSnapshot) -> None:
        if frame.rgb is None:
            self.simulator_label.configure(image="", text=frame.message)
            self._simulator_photo = None
            self._simulator_render_key = None
            return
        key = (frame.sequence, self.simulator_label.winfo_width(), self.simulator_label.winfo_height())
        if key != self._simulator_render_key:
            self._simulator_photo = self._photo_for(frame.rgb, frame.width, frame.height, key[1], key[2])
            self.simulator_label.configure(image=self._simulator_photo, text="")
            self._simulator_render_key = key

    @staticmethod
    def _photo_for(rgb: bytes, width: int, height: int, target_width: int, target_height: int):
        from PIL import Image, ImageTk

        image = Image.frombytes("RGB", (width, height), rgb)
        target_width = max(1, target_width)
        target_height = max(1, target_height)
        image.thumbnail((target_width, target_height), Image.Resampling.BILINEAR)
        return ImageTk.PhotoImage(image)

def run_app() -> None:
    root = tk.Tk()
    runtime = MimicRuntime()
    RobotMimicApp(root, runtime)
    root.mainloop()
