#!/usr/bin/env python3
"""Reproduce OpenPI PR #557's ALOHA gripper calibration mismatch."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocontract.calibration import (  # noqa: E402
    check_encoder_anchor_mapping,
    normalize,
    unnormalize,
)


FIXTURE = ROOT / "fixtures" / "openpi-aloha-pen-uncap-row0-lowdim.json"
FIX_COMMIT = "acda45e1544031df86d5ac68e86892a510cb47af"
PARENT_COMMIT = "072217ef94d2ac8710a2113048a398bc4b69efec"
MERGE_COMMIT = "66dbde5240b8ce9f05ac039cfaf460bebdcb05cd"
ENCODER_COUNTS = [2405, 3110]
ENCODER_ZERO = 2048
COUNTS_PER_REVOLUTION = 4096
OLD_BOUNDS = (0.4, 1.5)
FIXED_BOUNDS = (0.5476, 1.6296)


def _linear_to_radian(normalized_position: float) -> float:
    linear = unnormalize(normalized_position, 0.01844, 0.05800)
    arm_length = 0.036
    horn_radius = 0.022
    ratio = (horn_radius**2 + linear**2 - arm_length**2) / (
        2 * horn_radius * linear
    )
    return math.asin(max(-1.0, min(1.0, ratio)))


def _state_transform(value: float, bounds: tuple[float, float]) -> float:
    return normalize(_linear_to_radian(value), *bounds)


def _action_inverse_old(value: float) -> float:
    joint_radian = unnormalize(value, -0.6213, 1.4910)
    return normalize(joint_radian, *OLD_BOUNDS)


def _action_inverse_fixed(value: float) -> float:
    joint_radian = unnormalize(value, -0.6213, 1.4910)
    return joint_radian - FIXED_BOUNDS[0]


def main() -> int:
    fixture = json.loads(FIXTURE.read_text())
    old = check_encoder_anchor_mapping(
        ENCODER_COUNTS,
        [0.0, 1.0],
        zero_count=ENCODER_ZERO,
        counts_per_revolution=COUNTS_PER_REVOLUTION,
        calibration_lower_rad=OLD_BOUNDS[0],
        calibration_upper_rad=OLD_BOUNDS[1],
    )
    fixed = check_encoder_anchor_mapping(
        ENCODER_COUNTS,
        [0.0, 1.0],
        zero_count=ENCODER_ZERO,
        counts_per_revolution=COUNTS_PER_REVOLUTION,
        calibration_lower_rad=FIXED_BOUNDS[0],
        calibration_upper_rad=FIXED_BOUNDS[1],
    )

    # A fieldwise self-roundtrip is a deliberately weak baseline: any
    # internally consistent but physically wrong min/max pair passes it.
    anchor_radians = [
        (count - ENCODER_ZERO) * 2 * math.pi / COUNTS_PER_REVOLUTION
        for count in ENCODER_COUNTS
    ]
    old_roundtrip_errors = [
        abs(unnormalize(normalize(value, *OLD_BOUNDS), *OLD_BOUNDS) - value)
        for value in anchor_radians
    ]
    state_grippers = [fixture["observation_state"][6], fixture["observation_state"][13]]
    action_grippers = [fixture["action"][6], fixture["action"][13]]
    public_row = {
        "raw_state_grippers": state_grippers,
        "old_state_model_values": [_state_transform(value, OLD_BOUNDS) for value in state_grippers],
        "fixed_state_model_values": [
            _state_transform(value, FIXED_BOUNDS) for value in state_grippers
        ],
        "raw_action_grippers": action_grippers,
        "old_action_model_values": [_action_inverse_old(value) for value in action_grippers],
        "fixed_action_model_values": [
            _action_inverse_fixed(value) for value in action_grippers
        ],
    }
    summary = {
        "case": "OpenPI PR #557: inaccurate ALOHA gripper calibration anchors",
        "evidence_class": "merged VLA data-transform fix plus encoder-anchor oracle and public low-dimensional row",
        "public_dataset_rows": fixture["num_rows_total"],
        "old_self_roundtrip_baseline_accepted": max(old_roundtrip_errors) < 1e-12,
        "old_anchor_contract_accepted": old.accepted,
        "old_anchor_max_error": old.max_abs_error,
        "fixed_anchor_contract_accepted": fixed.accepted,
        "fixed_anchor_max_error": fixed.max_abs_error,
        "reproduced": (
            max(old_roundtrip_errors) < 1e-12
            and not old.accepted
            and old.max_abs_error > 0.1
            and fixed.accepted
            and fixed.max_abs_error < 1e-3
        ),
    }
    payload = {
        "summary": summary,
        "old_anchor_check": old.as_dict(),
        "fixed_anchor_check": fixed.as_dict(),
        "old_self_roundtrip_errors_rad": old_roundtrip_errors,
        "public_row_transform": public_row,
        "frozen_inputs": {
            "dataset": fixture["dataset"],
            "dataset_revision": fixture["revision"],
            "dataset_row": fixture["row_index"],
            "fix_commit": FIX_COMMIT,
            "parent_commit": PARENT_COMMIT,
            "merge_commit": MERGE_COMMIT,
            "encoder_counts": ENCODER_COUNTS,
            "encoder_zero": ENCODER_ZERO,
            "counts_per_revolution": COUNTS_PER_REVOLUTION,
        },
        "sources": {
            "pull_request": "https://github.com/Physical-Intelligence/openpi/pull/557",
            "fix_commit": f"https://github.com/Physical-Intelligence/openpi/commit/{FIX_COMMIT}",
            "policy_source": f"https://github.com/Physical-Intelligence/openpi/blob/{FIX_COMMIT}/src/openpi/policies/aloha_policy.py#L137-L160",
            "dataset": "https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse",
        },
        "limitations": [
            "The physical oracle validates the two encoder endpoints stated in the merged fix; it does not rerun VLA training or reproduce the reported task-performance degradation.",
            "Only one public low-dimensional row is frozen locally; 36,900 is the server-reported dataset row count, not the number downloaded.",
            "The old normalization pair is mathematically self-invertible, demonstrating why a fieldwise roundtrip is insufficient; the physical encoder anchors provide the ground truth.",
            "OpenPI intentionally uses asymmetric state/action transforms after the fix, so this audit does not demand a single global bijection for the whole policy adapter.",
        ],
    }
    output = ROOT / "results" / "real-defect-openpi-557.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(public_row, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0 if summary["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
