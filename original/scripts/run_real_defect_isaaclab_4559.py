#!/usr/bin/env python3
"""Reproduce the quaternion-order defect fixed by Isaac Lab PR #4559."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocontract.orientation import (  # noqa: E402
    check_orientation_transfer,
    quaternion_shape_range_accepts,
)


PARENT_COMMIT = "9b36e719cc63ff3b206d7c843aca3f089e581df3"
FIX_COMMIT = "a880ba4d49aaa51262a76be06242b2eb42d20063"
CONVENTION_CHANGE_COMMIT = "9659a5cefe7195c15b9b0c0cc519d86233b2e276"
SOURCE_PATH = (
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/"
    "deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"
)


def main() -> int:
    half_sqrt_two = math.sqrt(2.0) / 2.0
    # Exact expressions in the parent and fix commits.  The parent explicitly
    # labels the tuple WXYZ; after the framework-wide migration the consumer is
    # XYZW.  PR #4559 reorders the same physical rotation.
    legacy_wxyz = [0.0, half_sqrt_two, half_sqrt_two, 0.0]
    buggy_xyzw = list(legacy_wxyz)
    fixed_xyzw = [half_sqrt_two, half_sqrt_two, 0.0, 0.0]

    buggy = check_orientation_transfer(
        legacy_wxyz,
        source_order="wxyz",
        target=buggy_xyzw,
        target_order="xyzw",
    )
    fixed = check_orientation_transfer(
        legacy_wxyz,
        source_order="wxyz",
        target=fixed_xyzw,
        target_order="xyzw",
        operations=("reorder_quaternion",),
    )
    summary = {
        "case": "Isaac Lab PR #4559: stale WXYZ grasp offset consumed as XYZW",
        "evidence_class": "merged upstream bugfix with exact pre-fix and post-fix constants",
        "native_config_accepts_buggy_tuple": True,
        "shape_range_norm_baseline_accepts_buggy_tuple": quaternion_shape_range_accepts(
            buggy_xyzw
        ),
        "shape_range_norm_baseline_accepts_fixed_tuple": quaternion_shape_range_accepts(
            fixed_xyzw
        ),
        "buggy_contract_accepted": buggy.accepted,
        "buggy_angular_error_deg": math.degrees(buggy.angular_error_rad),
        "buggy_max_basis_displacement": buggy.max_basis_displacement,
        "fixed_contract_accepted": fixed.accepted,
        "fixed_angular_error_deg": math.degrees(fixed.angular_error_rad),
        "reproduced": (
            quaternion_shape_range_accepts(buggy_xyzw)
            and not buggy.accepted
            and math.isclose(math.degrees(buggy.angular_error_rad), 120.0)
            and fixed.accepted
        ),
    }
    payload = {
        "summary": summary,
        "legacy_wxyz": legacy_wxyz,
        "buggy_xyzw": buggy_xyzw,
        "fixed_xyzw": fixed_xyzw,
        "buggy_check": buggy.as_dict(),
        "fixed_check": fixed.as_dict(),
        "frozen_inputs": {
            "repository": "isaac-sim/IsaacLab",
            "convention_change_commit": CONVENTION_CHANGE_COMMIT,
            "pre_fix_parent_commit": PARENT_COMMIT,
            "merged_fix_commit": FIX_COMMIT,
            "source_path": SOURCE_PATH,
        },
        "sources": {
            "pull_request": "https://github.com/isaac-sim/IsaacLab/pull/4559",
            "fix_commit": f"https://github.com/isaac-sim/IsaacLab/commit/{FIX_COMMIT}",
            "release": "https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta",
        },
        "limitations": [
            "This is an exact code/configuration-path reproduction, not a full Isaac Sim rollout.",
            "The physical consequence is computed analytically on SO(3); no renderer or dynamics is required.",
            "The case contains one real affected configuration, so it is defect evidence rather than a statistical benchmark.",
            "native_config_accepts_buggy_tuple reflects the absence of an order field/check at this numeric configuration boundary.",
        ],
    }
    output = ROOT / "results" / "real-defect-isaaclab-4559.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("buggy issues:")
    for issue in buggy.issues:
        print(f"- {issue}")
    print(f"wrote {output}")
    return 0 if summary["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
