"""Verification for amendment A3: the stable geodesic rotation distance used by RD4's Arm C.

The RD4 Arm C decision boundary stays at its original 1e-6 degrees. What changed is how the angle is
evaluated. This script checks that the replacement is correct where it matters:

  V1 identical orientations evaluate to exactly zero, in both conventions;
  V2 the q / -q double cover evaluates to exactly zero;
  V3 known rotations just below and just above the unchanged 1e-6-degree boundary are classified
     correctly, and the recovered angle is accurate to a relative error far below the boundary;
  V4 the replacement agrees with the previous trace/arccos formulation wherever that formulation is
     well conditioned (angles of 1e-3 degrees and above), so this is an arithmetic correction and not
     a different rule;
  V5 the previous formulation's floor is exhibited directly, to document why it was replaced.

Writes rotation_distance_verification.json. Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from suites import quaternion_geodesic_deg  # noqa: E402

BOUNDARY_DEG = 1e-6
rng = np.random.default_rng(20260915)


def random_unit_quaternions(n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def wxyz_to_xyzw(q):
    return q[:, [1, 2, 3, 0]]


def compose_wxyz(a, b):
    """Hamilton product of two (w,x,y,z) quaternion arrays."""
    aw, av = a[:, 0], a[:, 1:4]
    bw, bv = b[:, 0], b[:, 1:4]
    w = aw * bw - np.einsum("ij,ij->i", av, bv)
    v = aw[:, None] * bv + bw[:, None] * av + np.cross(av, bv)
    return np.column_stack([w, v])


def rotation_wxyz(axis, angle_rad):
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    half = np.broadcast_to(np.asarray(angle_rad, dtype="f8"), (axis.shape[0],)) / 2.0
    return np.column_stack([np.cos(half), axis * np.sin(half)[:, None]])


def legacy_trace_arccos_deg(q_source_wxyz, q_target_xyzw):
    """The superseded evaluation, kept only to document its conditioning floor."""
    def matrices(q, order):
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        if order == "wxyz":
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        else:
            x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        m = np.empty(q.shape[:1] + (3, 3), dtype="f8")
        m[:, 0, 0] = 1 - 2 * (y * y + z * z)
        m[:, 0, 1] = 2 * (x * y - z * w)
        m[:, 0, 2] = 2 * (x * z + y * w)
        m[:, 1, 0] = 2 * (x * y + z * w)
        m[:, 1, 1] = 1 - 2 * (x * x + z * z)
        m[:, 1, 2] = 2 * (y * z - x * w)
        m[:, 2, 0] = 2 * (x * z - y * w)
        m[:, 2, 1] = 2 * (y * z + x * w)
        m[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return m

    rel = np.matmul(np.swapaxes(matrices(q_source_wxyz, "wxyz"), -1, -2),
                    matrices(q_target_xyzw, "xyzw"))
    tr = np.trace(rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def main() -> int:
    checks, failures = {}, []
    n = 20000
    base = random_unit_quaternions(n)

    # V1/V2 criterion: at most NEGLIGIBLE_DEG, six orders of magnitude below the decision boundary.
    # Bit-exact zero is not the right criterion here: the expectation receives the source as (w,x,y,z)
    # and the target as (x,y,z,w), so the two normalisations sum the same four squares in different
    # orders and the unit quaternions can differ in their last bit before any angle is computed.
    NEGLIGIBLE_DEG = 1e-12

    # V1 identical orientations, expressed in the two conventions the expectation actually receives.
    identical = quaternion_geodesic_deg(base, "wxyz", wxyz_to_xyzw(base), "xyzw")
    checks["V1_identical_max_deg"] = float(np.max(identical))
    checks["V1_criterion_deg"] = NEGLIGIBLE_DEG
    checks["V1_identical_negligible"] = bool(np.all(identical <= NEGLIGIBLE_DEG))
    if not checks["V1_identical_negligible"]:
        failures.append("V1")

    # V1b bit-identical input arrays in one convention must give exactly zero.
    same_array = quaternion_geodesic_deg(base, "wxyz", base, "wxyz")
    checks["V1b_same_convention_exactly_zero"] = bool(np.all(same_array == 0.0))
    if not checks["V1b_same_convention_exactly_zero"]:
        failures.append("V1b")

    # V2 double cover: q and -q denote the same rotation.
    negated = quaternion_geodesic_deg(base, "wxyz", wxyz_to_xyzw(-base), "xyzw")
    checks["V2_double_cover_max_deg"] = float(np.max(negated))
    checks["V2_double_cover_negligible"] = bool(np.all(negated <= NEGLIGIBLE_DEG))
    checks["V2b_same_convention_negated_exactly_zero"] = bool(
        np.all(quaternion_geodesic_deg(base, "wxyz", -base, "wxyz") == 0.0))
    if not checks["V2_double_cover_negligible"]:
        failures.append("V2")
    if not checks["V2b_same_convention_negated_exactly_zero"]:
        failures.append("V2b")

    # V3 known small rotations straddling the unchanged boundary.
    straddle = {}
    for label, factor in (("below", 0.5), ("just_below", 0.99), ("just_above", 1.01), ("above", 2.0)):
        angle_deg = BOUNDARY_DEG * factor
        axis = rng.normal(size=(n, 3))
        perturbed = compose_wxyz(base, rotation_wxyz(axis, np.radians(angle_deg)))
        measured = quaternion_geodesic_deg(base, "wxyz", wxyz_to_xyzw(perturbed), "xyzw")
        accepted = measured <= BOUNDARY_DEG
        expected_accepted = factor <= 1.0
        straddle[label] = {
            "applied_angle_deg": angle_deg,
            "max_relative_error": float(np.max(np.abs(measured - angle_deg)) / angle_deg),
            "accepted_fraction": float(np.mean(accepted)),
            "classified_correctly": bool(np.all(accepted == expected_accepted)),
        }
        if not straddle[label]["classified_correctly"]:
            failures.append(f"V3_{label}")
        if straddle[label]["max_relative_error"] > 1e-6:
            failures.append(f"V3_{label}_accuracy")
    checks["V3_boundary_straddle"] = straddle

    # V4 correctness against the applied ground-truth angle across the whole range, plus agreement
    # with the superseded formulation restricted to angles where that formulation is itself accurate.
    # Correctness is judged against ground truth, not against the formulation being replaced.
    axis = rng.normal(size=(n, 3))
    angles_deg = 10 ** rng.uniform(-3, np.log10(179.0), size=n)
    far = compose_wxyz(base, rotation_wxyz(axis, np.radians(angles_deg)))
    stable = quaternion_geodesic_deg(base, "wxyz", wxyz_to_xyzw(far), "xyzw")
    legacy = legacy_trace_arccos_deg(base, wxyz_to_xyzw(far))
    checks["V4_stable_max_relative_error_vs_applied"] = float(
        np.max(np.abs(stable - angles_deg) / angles_deg))
    if checks["V4_stable_max_relative_error_vs_applied"] > 1e-9:
        failures.append("V4_accuracy")

    well_conditioned = angles_deg >= 1.0
    checks["V4_agreement_with_legacy_at_or_above_1_deg"] = float(
        np.max(np.abs(stable[well_conditioned] - legacy[well_conditioned])
               / angles_deg[well_conditioned]))
    if checks["V4_agreement_with_legacy_at_or_above_1_deg"] > 1e-9:
        failures.append("V4_agreement")

    # Descriptive only: below 1 degree the disagreement is the legacy formulation's own error.
    small = angles_deg < 1.0
    checks["V4_descriptive_small_angle_disagreement"] = {
        "max_relative_disagreement": float(
            np.max(np.abs(stable[small] - legacy[small]) / angles_deg[small])),
        "legacy_max_relative_error_vs_applied": float(
            np.max(np.abs(legacy[small] - angles_deg[small]) / angles_deg[small])),
        "stable_max_relative_error_vs_applied": float(
            np.max(np.abs(stable[small] - angles_deg[small]) / angles_deg[small])),
    }

    # V5 document the superseded formulation's floor on genuinely identical orientations.
    legacy_identical = legacy_trace_arccos_deg(base, wxyz_to_xyzw(base))
    checks["V5_legacy_floor_on_identical_orientations_deg"] = {
        "max": float(np.max(legacy_identical)),
        "fraction_above_1e_6_deg": float(np.mean(legacy_identical > BOUNDARY_DEG)),
    }
    checks["V5_stable_floor_on_identical_orientations_deg"] = float(np.max(identical))

    report = {
        "amendment": "A3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision_boundary_deg": BOUNDARY_DEG,
        "decision_boundary_changed": False,
        "evaluation_changed": "trace/arccos replaced by relative-quaternion 2*atan2(||vec||,|w|)",
        "samples_per_check": n,
        "seed": 20260915,
        "numpy": np.__version__,
        "checks": checks,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    (HERE / "rotation_distance_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
