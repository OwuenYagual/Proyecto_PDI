"""Contratos inmutables entre la visión (M3) y el controlador (M4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmCommand:
    """Pose humana: elevación firmada de hombro y ángulo interno de codo."""

    shoulder_deg: float
    elbow_deg: float


@dataclass(frozen=True)
class GripperCommand:
    """Apertura normalizada del gripper, entre cero y uno."""

    aperture: float


@dataclass(frozen=True)
class MimicCommand:
    """Muestra tipada producida por M3.

    Los canales son independientes: un error en el brazo no impide emitir un
    objetivo de gripper válido, y viceversa.
    """

    sequence: int
    created_at: float
    arm: ArmCommand | None = None
    gripper: GripperCommand | None = None
    arm_error: str | None = None
    gripper_error: str | None = None


@dataclass(frozen=True)
class ChannelValidation:
    """Resultado de validar un canal de un comando."""

    accepted: bool
    reason: str | None = None
    targets: tuple[float, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """Resultado independiente de los canales de brazo y gripper."""

    arm: ChannelValidation
    gripper: ChannelValidation
