#!/usr/bin/env python3
"""Execute and audit Isaac Lab's public legacy-quaternion migration path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocontract.isaaclab_migration import (  # noqa: E402
    audit_migration,
    run_frozen_upstream_converter,
)


SOURCE = ROOT / "data" / "raw" / "isaaclab" / "franka_stack_v51.hdf5"
CONVERTER_SOURCE = (
    ROOT
    / "data"
    / "raw"
    / "isaaclab"
    / "hdf5_dataset_file_handler_beta2_patch1.py"
)
TARGET = ROOT / "runs" / "isaaclab-quaternion-migration" / "native-output.hdf5"
SOURCE_SHA256 = "e0954647328801b091b81604915acd4eff2093b2b26ff0b05753c8f85fbb61b3"
CONVERTER_SHA256 = "b5c4365cfd46107dfe56288b092bbdb03d993dbb83e23415742965db7df8d793"
SOURCE_SEMANTICS_COMMIT = "37ddf626871758333d6ed89cf64ad702aef127d0"
CONVERTER_COMMIT = "ffff603eafc6b74264a5261cc0183d6a65390d78"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    source_hash = _sha256(SOURCE)
    converter_hash = _sha256(CONVERTER_SOURCE)
    if source_hash != SOURCE_SHA256:
        raise ValueError(f"source SHA mismatch: expected {SOURCE_SHA256}, got {source_hash}")
    if converter_hash != CONVERTER_SHA256:
        raise ValueError(
            f"converter SHA mismatch: expected {CONVERTER_SHA256}, got {converter_hash}"
        )

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    run_frozen_upstream_converter(CONVERTER_SOURCE, SOURCE, TARGET)
    audit = audit_migration(SOURCE, TARGET)
    summary = {
        "case": "Isaac Lab legacy HDF5 WXYZ-to-XYZW partial migration",
        "evidence_class": "unmodified released converter on official public demonstration HDF5",
        "schema_baseline_accepted": audit["schema_baseline"]["accepted"],
        "target_format_version": audit["schema_baseline"]["target_format_version"],
        "declared_quaternion_instances": audit["contract"]["declared_instances"],
        "converted_quaternion_instances": audit["contract"]["converted_instances"],
        "missed_quaternion_instances": audit["contract"]["undischarged_instances"],
        "semantic_coverage": audit["contract"]["instance_weighted_coverage"],
        "robocontract_accepted": audit["contract"]["accepted"],
        "mean_missed_orientation_error_deg": audit["physical_oracle"]["mean_error_deg"],
        "median_missed_orientation_error_deg": audit["physical_oracle"]["median_error_deg"],
        "p95_missed_orientation_error_deg": audit["physical_oracle"]["p95_error_deg"],
        "missed_over_90_degrees": audit["physical_oracle"]["over_90_degree_instances"],
        "reproduced": (
            audit["schema_baseline"]["accepted"]
            and not audit["contract"]["accepted"]
            and audit["contract"]["converted_instances"] == 8692
            and audit["contract"]["undischarged_instances"] == 15141
            and audit["physical_oracle"]["over_90_degree_instances"] == 15141
        ),
    }
    payload = {
        "summary": summary,
        "audit": audit,
        "frozen_inputs": {
            "dataset": "Isaac Lab v2.3.2 Franka stack human demonstrations",
            "dataset_url": "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Mimic/franka_stack_datasets/dataset.hdf5",
            "dataset_sha256": SOURCE_SHA256,
            "source_semantics_commit": SOURCE_SEMANTICS_COMMIT,
            "converter_commit": CONVERTER_COMMIT,
            "converter_source_sha256": CONVERTER_SHA256,
        },
        "sources": {
            "official_fixture_documentation": "https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/imitation-learning/teleop_imitation.html",
            "v2_task_field_provenance": f"https://github.com/isaac-sim/IsaacLab/blob/{SOURCE_SEMANTICS_COMMIT}/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack/mdp/observations.py",
            "v3_converter": f"https://github.com/isaac-sim/IsaacLab/blob/{CONVERTER_COMMIT}/source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py#L285-L351",
        },
        "limitations": [
            "The frozen upstream class and static converter body execute unchanged; only unavailable package imports and the documented quaternion permutation are stubbed.",
            "The independent physical oracle uses explicit Hamilton rotation matrices and does not import the upstream converter helper.",
            "Repeated cube orientations stored in cube_orientations and object are separate serialized contract obligations but not independent physical samples.",
            "The public source has an unrelated stale data.total attribute; it is reported separately and excluded from the quaternion contract decision.",
            "This result applies to the inspected beta migration utility, not every Isaac Lab runtime path or dataset.",
        ],
    }
    output = ROOT / "results" / "real-defect-isaaclab-quaternion-migration.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0 if summary["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
