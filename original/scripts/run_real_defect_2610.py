#!/usr/bin/env python3
"""Minimal code-path reproduction of LeRobot issue #2610 on public metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocontract.episode_windows import (  # noqa: E402
    check_window_query,
    expected_window_query,
    upstream_2610_buggy_query,
)


METADATA = (
    ROOT
    / "data"
    / "raw"
    / "lerobot"
    / "pusht"
    / "meta"
    / "episodes"
    / "chunk-000"
    / "file-000.parquet"
)
EXPECTED_SHA256 = "bc1226f33d3d1635ec1954f9942073709be8e106fde5f4f16b9d52edc4e0ebc4"
DATASET_REVISION = "7628202a2180972f291ba1bc6723834921e72c19"
BUGGY_LEROBOT_COMMIT = "7f40b3bf82ef9d0fb96d3ba6dc8cc33dd6cd90fe"
PROPOSED_FIX_COMMIT = "1e6a0d09bcc5de620344525311ad86575fa8c8da"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("this reproducer requires pyarrow") from error

    actual_sha = _sha256(METADATA)
    if actual_sha != EXPECTED_SHA256:
        raise ValueError(
            f"metadata SHA mismatch: expected {EXPECTED_SHA256}, got {actual_sha}"
        )
    episodes = pq.read_table(
        METADATA,
        columns=["episode_index", "dataset_from_index", "dataset_to_index", "length"],
    ).to_pylist()
    episode = next(row for row in episodes if row["episode_index"] == 1)
    start = int(episode["dataset_from_index"])
    end = int(episode["dataset_to_index"])
    length = int(episode["length"])
    if end - start != length:
        raise ValueError("episode length conflicts with absolute index boundaries")

    rows = []
    for relative_index in range(length):
        absolute_index = start + relative_index
        buggy_query, buggy_pad = upstream_2610_buggy_query(
            relative_index=relative_index,
            delta=0,
            episode_start=start,
            episode_end=end,
        )
        fixed_query, fixed_pad = expected_window_query(
            absolute_index, 0, start, end
        )
        buggy_check = check_window_query(
            absolute_index=absolute_index,
            delta=0,
            episode_start=start,
            episode_end=end,
            observed_query_index=buggy_query,
            observed_is_pad=buggy_pad,
        )
        fixed_check = check_window_query(
            absolute_index=absolute_index,
            delta=0,
            episode_start=start,
            episode_end=end,
            observed_query_index=fixed_query,
            observed_is_pad=fixed_pad,
        )
        rows.append(
            {
                "relative_index": relative_index,
                "absolute_index": absolute_index,
                "buggy_query_index": buggy_query,
                "buggy_is_pad": buggy_pad,
                "buggy_contract_accepted": buggy_check.accepted,
                "buggy_contract_issues": buggy_check.issues,
                "fixed_query_index": fixed_query,
                "fixed_is_pad": fixed_pad,
                "fixed_contract_accepted": fixed_check.accepted,
            }
        )

    summary = {
        "case": "LeRobot issue #2610: delta timestamps plus episode filter",
        "evidence_class": "minimal upstream-code-path reproduction on public dataset metadata",
        "full_upstream_api_execution": False,
        "episode_index": 1,
        "episode_start": start,
        "episode_end": end,
        "episode_frames": length,
        "buggy_frames_marked_pad": sum(row["buggy_is_pad"] for row in rows),
        "buggy_contract_violations": sum(
            not row["buggy_contract_accepted"] for row in rows
        ),
        "fixed_frames_marked_pad": sum(row["fixed_is_pad"] for row in rows),
        "fixed_contract_violations": sum(
            not row["fixed_contract_accepted"] for row in rows
        ),
        "reproduced": all(row["buggy_is_pad"] for row in rows)
        and all(not row["fixed_is_pad"] for row in rows),
    }
    payload = {
        "summary": summary,
        "frozen_inputs": {
            "dataset": "lerobot/pusht",
            "dataset_revision": DATASET_REVISION,
            "metadata_path": "meta/episodes/chunk-000/file-000.parquet",
            "metadata_sha256": EXPECTED_SHA256,
            "buggy_lerobot_commit": BUGGY_LEROBOT_COMMIT,
            "buggy_source_file_sha256": "bef8f912c8be5503051f8368608562456b700eacfda4d91911d8bdf629fac393",
            "proposed_fix_commit": PROPOSED_FIX_COMMIT,
        },
        "sources": {
            "issue": "https://github.com/huggingface/lerobot/issues/2610",
            "proposed_fix": "https://github.com/huggingface/lerobot/pull/2612",
            "dataset": "https://huggingface.co/datasets/lerobot/pusht",
        },
        "rows": rows,
        "limitations": [
            "This runs the exact faulty index arithmetic as a minimal transcription, not the complete LeRobot Python API.",
            "Only episode metadata is required; image/video payloads are not downloaded.",
            "The proposed fix PR was open in the inspected record, so this result does not claim a released fix version.",
        ],
    }
    output = ROOT / "results" / "real-defect-lerobot-2610.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {output}")
    return 0 if summary["reproduced"] and summary["fixed_contract_violations"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

