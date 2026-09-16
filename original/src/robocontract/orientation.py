"""Executable contracts for orientation values crossing API boundaries.

The trajectory pilot focuses on action effects.  This module covers a second
Physical-AI data path that is easy to validate without a simulator: a
quaternion produced under one component convention and consumed under another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


SUPPORTED_ORDERS = {"wxyz", "xyzw"}


@dataclass
class OrientationCheckResult:
    accepted: bool
    issues: list[str] = field(default_factory=list)
    angular_error_rad: float = 0.0
    max_basis_displacement: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "issues": self.issues,
            "angular_error_rad": self.angular_error_rad,
            "angular_error_deg": math.degrees(self.angular_error_rad),
            "max_basis_displacement": self.max_basis_displacement,
        }


def _values(quaternion: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4:
        raise ValueError(f"quaternion must have four components, found {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion contains a non-finite component")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("quaternion has zero norm")
    return tuple(value / norm for value in values)


def canonical_xyzw(
    quaternion: Iterable[float], order: str
) -> tuple[float, float, float, float]:
    """Normalize a quaternion and return it in canonical XYZW order."""

    if order not in SUPPORTED_ORDERS:
        raise ValueError(f"unsupported quaternion order: {order}")
    first, second, third, fourth = _values(quaternion)
    if order == "xyzw":
        return first, second, third, fourth
    return second, third, fourth, first


def quaternion_shape_range_accepts(
    quaternion: Iterable[float], *, norm_tolerance: float = 1e-6
) -> bool:
    """Non-semantic baseline: finite length-4 values in range with unit norm."""

    try:
        raw = tuple(float(value) for value in quaternion)
    except (TypeError, ValueError):
        return False
    if len(raw) != 4 or not all(math.isfinite(value) for value in raw):
        return False
    if any(value < -1.0 or value > 1.0 for value in raw):
        return False
    return abs(math.sqrt(sum(value * value for value in raw)) - 1.0) <= norm_tolerance


def quaternion_geodesic_rad(
    left_xyzw: Iterable[float], right_xyzw: Iterable[float]
) -> float:
    """Shortest SO(3) angular distance, treating q and -q as identical."""

    left = _values(left_xyzw)
    right = _values(right_xyzw)
    dot = abs(sum(a * b for a, b in zip(left, right)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def _rotation_matrix_xyzw(
    quaternion: Iterable[float],
) -> tuple[tuple[float, float, float], ...]:
    x, y, z, w = _values(quaternion)
    return (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )


def _max_basis_displacement(
    left_xyzw: Iterable[float], right_xyzw: Iterable[float]
) -> float:
    left = _rotation_matrix_xyzw(left_xyzw)
    right = _rotation_matrix_xyzw(right_xyzw)
    # A rotation matrix column is the corresponding transformed basis vector.
    return max(
        math.sqrt(sum((left[row][column] - right[row][column]) ** 2 for row in range(3)))
        for column in range(3)
    )


def check_orientation_transfer(
    source: Iterable[float],
    *,
    source_order: str,
    target: Iterable[float],
    target_order: str,
    operations: tuple[str, ...] = (),
    tolerance_rad: float = 1e-6,
) -> OrientationCheckResult:
    """Check convention declaration and physical orientation preservation."""

    issues: list[str] = []
    if source_order != target_order and "reorder_quaternion" not in operations:
        issues.append(
            "undeclared quaternion-order change: reorder_quaternion is required"
        )
    try:
        source_xyzw = canonical_xyzw(source, source_order)
        target_xyzw = canonical_xyzw(target, target_order)
        angular_error = quaternion_geodesic_rad(source_xyzw, target_xyzw)
        basis_displacement = _max_basis_displacement(source_xyzw, target_xyzw)
        if angular_error > tolerance_rad:
            issues.append(
                f"physical orientations diverge by {math.degrees(angular_error):.6g} degrees"
            )
    except (TypeError, ValueError) as error:
        issues.append(f"orientation decoding failed: {error}")
        angular_error = math.inf
        basis_displacement = math.inf

    return OrientationCheckResult(
        accepted=not issues,
        issues=issues,
        angular_error_rad=angular_error,
        max_basis_displacement=basis_displacement,
    )
