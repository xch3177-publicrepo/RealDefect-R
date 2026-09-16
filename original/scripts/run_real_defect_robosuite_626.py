#!/usr/bin/env python3
"""Replay robosuite PR #626's base-frame delta bug on public actions/XML."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocontract.rigid_transform import (  # noqa: E402
    check_delta_transform,
    rotation_matrix_from_wxyz,
    transform_delta_as_absolute_pose_bug,
    transform_delta_correct,
)


DATASET = ROOT / "data" / "raw" / "robomimic" / "low_dim_v15.hdf5"
DATASET_SHA256 = "0d302b37fb08e99b00de287b833925589ad3b65d199306e55562938bcbd746f1"
BUG_PARENT = "53ed0ff33019aa29f4ca12aefaa67d2d53e65112"
FIX_COMMIT = "43a498d4e2adc445012fa9d6e00821d5c22a14e3"
MERGE_COMMIT = "1c947f05d307b2660fb7821118a316ea7447586f"
V150_COMMIT = "1a8701b90c07c6595ace4af9935d7c5ebe1baed3"
V151_COMMIT = "51cc01785bab80ffeed20da15e67d7dd4140e76a"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_transform(model_xml: str) -> tuple[list[float], list[list[float]]]:
    root = ET.fromstring(model_xml)
    body = next(
        element
        for element in root.iter("body")
        if element.attrib.get("name") == "robot0_base"
    )
    translation = [float(value) for value in body.attrib.get("pos", "0 0 0").split()]
    quaternion = [float(value) for value in body.attrib.get("quat", "1 0 0 0").split()]
    return translation, rotation_matrix_from_wxyz(quaternion).tolist()


def _decode_translation(raw_actions, controller):
    import numpy as np

    raw = np.asarray(raw_actions, dtype="f8")[:, :3]
    input_min = np.broadcast_to(np.asarray(controller["input_min"], dtype="f8"), (3,))
    input_max = np.broadcast_to(np.asarray(controller["input_max"], dtype="f8"), (3,))
    output_min = np.asarray(controller["output_min"], dtype="f8")[:3]
    output_max = np.asarray(controller["output_max"], dtype="f8")[:3]
    fraction = (np.clip(raw, input_min, input_max) - input_min) / (input_max - input_min)
    return output_min + fraction * (output_max - output_min)


def main() -> int:
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError("this reproducer requires h5py and numpy") from error

    if _sha256(DATASET) != DATASET_SHA256:
        raise ValueError("frozen public robomimic dataset SHA mismatch")

    rows: list[dict] = []
    input_range_accepts = True
    fixed_contract_violations = 0
    with h5py.File(DATASET, "r") as handle:
        env_args = json.loads(str(handle["data"].attrs["env_args"]))
        controller = env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]
        for demo_id in sorted(
            handle["data"].keys(), key=lambda value: int(value.rsplit("_", 1)[1])
        ):
            demo = handle[f"data/{demo_id}"]
            raw_actions = np.asarray(demo["actions"])
            input_range_accepts = input_range_accepts and bool(
                np.all(raw_actions >= -1.0) and np.all(raw_actions <= 1.0)
            )
            deltas = _decode_translation(raw_actions, controller)
            translation, rotation = _base_transform(str(demo.attrs["model_file"]))
            translation_array = np.asarray(translation)
            rotation_array = np.asarray(rotation)
            for step, delta in enumerate(deltas):
                correct = transform_delta_correct(delta, rotation_array)
                buggy = transform_delta_as_absolute_pose_bug(
                    delta, translation_array, rotation_array
                )
                check = check_delta_transform(
                    delta,
                    base_translation=translation_array,
                    base_rotation=rotation_array,
                    observed_world_delta=buggy,
                )
                fixed_check = check_delta_transform(
                    delta,
                    base_translation=translation_array,
                    base_rotation=rotation_array,
                    observed_world_delta=correct,
                )
                fixed_contract_violations += not fixed_check.accepted
                rows.append(
                    {
                        "demo_id": demo_id,
                        "step": step,
                        "source_delta_m": delta.tolist(),
                        "base_translation_m": translation,
                        "correct_world_delta_m": correct.tolist(),
                        "buggy_world_delta_m": buggy.tolist(),
                        "contract_accepted": check.accepted,
                        "error_norm_m": check.error_norm_m,
                    }
                )

    error_norms = [row["error_norm_m"] for row in rows]
    buggy_norms = [float(np.linalg.norm(row["buggy_world_delta_m"])) for row in rows]
    correct_norms = [float(np.linalg.norm(row["correct_world_delta_m"])) for row in rows]
    first_nonzero = next(row for row in rows if np.linalg.norm(row["source_delta_m"]) > 0)
    summary = {
        "case": "robosuite PR #626: base translation leaked into Mink IK delta actions",
        "evidence_class": "merged upstream bugfix with public-action and embedded-model exact branch replay",
        "public_transitions_replayed": len(rows),
        "input_shape_range_baseline_accepted": input_range_accepts,
        "buggy_contract_violations": sum(not row["contract_accepted"] for row in rows),
        "fixed_contract_violations": fixed_contract_violations,
        "fixed_contract_accepted": fixed_contract_violations == 0,
        "median_endpoint_error_m": statistics.median(error_norms),
        "min_endpoint_error_m": min(error_norms),
        "max_endpoint_error_m": max(error_norms),
        "median_correct_delta_norm_m": statistics.median(correct_norms),
        "median_buggy_delta_norm_m": statistics.median(buggy_norms),
        "reproduced": (
            input_range_accepts
            and len(rows) == 9666
            and all(not row["contract_accepted"] for row in rows)
            and fixed_contract_violations == 0
            and math.isclose(min(error_norms), max(error_norms), abs_tol=1e-12)
        ),
    }
    payload = {
        "summary": summary,
        "representative_nonzero_step": first_nonzero,
        "frozen_inputs": {
            "dataset": "robomimic Lift PH low_dim v1.5",
            "dataset_sha256": DATASET_SHA256,
            "bug_parent": BUG_PARENT,
            "logic_fix_commit": FIX_COMMIT,
            "merge_commit": MERGE_COMMIT,
            "v1.5.0_commit": V150_COMMIT,
            "v1.5.1_commit": V151_COMMIT,
            "upstream_path": "robosuite/examples/third_party_controller/mink_controller.py",
        },
        "sources": {
            "pull_request": "https://github.com/ARISE-Initiative/robosuite/pull/626",
            "merge_commit": f"https://github.com/ARISE-Initiative/robosuite/commit/{MERGE_COMMIT}",
            "release_notes": "https://github.com/ARISE-Initiative/robosuite/releases",
            "controller_docs": "https://robosuite.ai/docs/source/robosuite.controllers.parts.arm.html",
        },
        "limitations": [
            "The public Lift demonstrations were collected with OSC_POSE/world, not the affected Mink/base branch.",
            "This is an exact branch replay driven by real public action values and each episode's official embedded model XML, not an end-to-end historical rollout.",
            "A range checker applied to the serialized normalized input accepts it; an instrumented checker applied after the faulty internal transform could also flag the very large delta.",
            "The physical endpoint discrepancy is algebraic and does not require MuJoCo dynamics to establish.",
        ],
    }
    output = ROOT / "results" / "real-defect-robosuite-626.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(first_nonzero, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0 if summary["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
