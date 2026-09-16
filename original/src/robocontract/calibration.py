"""Executable physical-anchor contracts for scalar actuator calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


@dataclass
class CalibrationCheck:
    accepted: bool
    issues: list[str] = field(default_factory=list)
    observed_outputs: list[float] = field(default_factory=list)
    expected_outputs: list[float] = field(default_factory=list)
    max_abs_error: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "issues": self.issues,
            "observed_outputs": self.observed_outputs,
            "expected_outputs": self.expected_outputs,
            "max_abs_error": self.max_abs_error,
        }


def encoder_count_to_radian(
    count: float, *, zero_count: float, counts_per_revolution: float
) -> float:
    return (float(count) - zero_count) * 2.0 * math.pi / counts_per_revolution


def normalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("calibration upper bound must exceed lower bound")
    return (float(value) - lower) / (upper - lower)


def unnormalize(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("calibration upper bound must exceed lower bound")
    return float(value) * (upper - lower) + lower


def check_encoder_anchor_mapping(
    encoder_counts: Iterable[float],
    expected_outputs: Iterable[float],
    *,
    zero_count: float,
    counts_per_revolution: float,
    calibration_lower_rad: float,
    calibration_upper_rad: float,
    tolerance: float = 1e-3,
) -> CalibrationCheck:
    counts = list(encoder_counts)
    expected = [float(value) for value in expected_outputs]
    if len(counts) != len(expected) or not counts:
        return CalibrationCheck(False, ["anchor counts and expected values differ"])
    observed = [
        normalize(
            encoder_count_to_radian(
                count,
                zero_count=zero_count,
                counts_per_revolution=counts_per_revolution,
            ),
            calibration_lower_rad,
            calibration_upper_rad,
        )
        for count in counts
    ]
    errors = [abs(left - right) for left, right in zip(observed, expected)]
    issues = [
        f"physical calibration anchor {index} differs by {error:.6g}"
        for index, error in enumerate(errors)
        if error > tolerance
    ]
    return CalibrationCheck(
        accepted=not issues,
        issues=issues,
        observed_outputs=observed,
        expected_outputs=expected,
        max_abs_error=max(errors, default=0.0),
    )
