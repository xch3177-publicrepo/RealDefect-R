#!/usr/bin/env python3
"""Audit all public Cosmos3-DROID success/failure episode metadata on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocontract.droid_metadata_collection import (  # noqa: E402
    audit_droid_metadata_collection,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/droid")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/p8-droid-full-metadata-audit.json",
    )
    args = parser.parse_args()

    results = {
        split: audit_droid_metadata_collection(args.data_root / split, split=split)
        for split in ("success", "failure")
    }
    payload = {
        "summary": {
            "splits": 2,
            "episode_rows": sum(
                result["summary"]["episode_rows"] for result in results.values()
            ),
            "declared_frames_represented": sum(
                result["summary"]["declared_frames_represented"]
                for result in results.values()
            ),
            "episode_metadata_shards": sum(
                result["summary"]["episode_metadata_shards"]
                for result in results.values()
            ),
            "identity_inconsistent_rows": sum(
                result["summary"]["identity_inconsistent_rows"]
                for result in results.values()
            ),
            "task_companion_mismatch_rows": sum(
                result["summary"]["task_companion_mismatch_rows"]
                for result in results.values()
            ),
            "all_declared_count_checks_passed": all(
                result["summary"]["declared_count_checks_passed"]
                for result in results.values()
            ),
            "gpu_used": False,
            "scope": "complete public episode metadata for both splits; no trajectories or videos",
        },
        "splits": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0 if payload["summary"]["all_declared_count_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

