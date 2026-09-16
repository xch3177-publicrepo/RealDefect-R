"""Physical contracts for transforming absolute poses and delta vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class DeltaTransformCheck:
    accepted: bool
    issues: list[str] = field(default_factory=list)
    expected_delta: list[float] = field(default_factory=list)
    observed_delta: list[float] = field(default_factory=list)
    translation_leak: list[float] = field(default_factory=list)
    error_norm_m: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "issues": self.issues,
            "expected_delta": self.expected_delta,
            "observed_delta": self.observed_delta,
            "translation_leak": self.translation_leak,
            "error_norm_m": self.error_norm_m,
        }


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("rigid-transform checks require numpy") from error
    return np


def transform_delta_correct(delta: Any, base_rotation: Any) -> Any:
    """Transform a free vector: rotate it, never add the frame origin."""

    np = _numpy()
    return np.asarray(base_rotation, dtype="f8") @ np.asarray(delta, dtype="f8")


def transform_delta_as_absolute_pose_bug(
    delta: Any, base_translation: Any, base_rotation: Any
) -> Any:
    """Historical PR #626 bug: use the point/pose transform for a delta."""

    np = _numpy()
    return np.asarray(base_translation, dtype="f8") + transform_delta_correct(
        delta, base_rotation
    )


def check_delta_transform(
    delta: Any,
    *,
    base_translation: Any,
    base_rotation: Any,
    observed_world_delta: Any,
    tolerance_m: float = 1e-9,
) -> DeltaTransformCheck:
    """Require translation invariance and rotation-only vector semantics."""

    np = _numpy()
    expected = transform_delta_correct(delta, base_rotation)
    observed = np.asarray(observed_world_delta, dtype="f8")
    translation = np.asarray(base_translation, dtype="f8")
    issues: list[str] = []
    if expected.shape != (3,) or observed.shape != (3,) or translation.shape != (3,):
        return DeltaTransformCheck(False, ["delta/base vectors must be three-dimensional"])
    error = observed - expected
    error_norm = float(np.linalg.norm(error))
    if error_norm > tolerance_m:
        issues.append(f"delta frame transform leaks {error_norm:.6g} m of translation")
    if abs(float(np.linalg.norm(expected)) - float(np.linalg.norm(delta))) > tolerance_m:
        issues.append("base rotation does not preserve delta-vector norm")
    if np.allclose(error, translation, atol=tolerance_m) and np.linalg.norm(translation) > tolerance_m:
        issues.append("observed error equals the base-frame origin")
    return DeltaTransformCheck(
        accepted=not issues,
        issues=issues,
        expected_delta=expected.tolist(),
        observed_delta=observed.tolist(),
        translation_leak=error.tolist(),
        error_norm_m=error_norm,
    )


def rotation_matrix_from_wxyz(quaternion: Any) -> Any:
    np = _numpy()
    q = np.asarray(quaternion, dtype="f8")
    if q.shape != (4,):
        raise ValueError("quaternion must contain four WXYZ components")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid quaternion norm")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype="f8",
    )
