#!/usr/bin/env python3
"""Reproduce LeRobot PR #2057's v2.1→v3 offset-reset defect."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robocontract.episode_metadata import (  # noqa: E402
    check_episode_index_chain,
    simulate_v21_to_v30_offsets,
)


EPISODES = ROOT / "data" / "raw" / "lerobot" / "libero-v21" / "meta" / "episodes.jsonl"
EPISODES_SHA256 = "0479a2ced6117700e4638a1da54dccd9f60ae5f47485ea6d91390c869befcf08"
DATASET_REVISION = "affa19c0de0f6bce2a7edd26dddef8a532e7e6f6"
MERGED_FIX_COMMIT = "ec40ccde0d892d6ada167f371e93eef997b2e669"
TARGET_FILE_SIZE_MB = 100.0

# Actual Git-LFS pointer metadata for episodes 0..4 in the frozen dataset.
FILE_EVIDENCE = [
    {
        "sha256": "ef7e4c224eccd84b6467987d8b65777b38c4d533c1e4bc3084234eb2edf5f040",
        "size": 28005402,
    },
    {
        "sha256": "ee424759f47e856d7b2f93706d70cb0cefe4366d477e54daff9bb6dc65b905df",
        "size": 37410833,
    },
    {
        "sha256": "27625daed9225e5136088b4adc06e0af594da8024a3fd64a911543c4a510266b",
        "size": 35344252,
    },
    {
        "sha256": "e9308a33462de5c06b7594bdfe96def4b6b17cf84cb958cd6f81e3170b15f4dc",
        "size": 28705859,
    },
    {
        "sha256": "68d449be4c75832e9b72cbb4127e048e28a45ff7b1023af6ca5cb5a4ac17e0f7",
        "size": 36225726,
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    actual_sha = _sha256(EPISODES)
    if actual_sha != EPISODES_SHA256:
        raise ValueError(
            f"episodes metadata SHA mismatch: expected {EPISODES_SHA256}, got {actual_sha}"
        )
    with EPISODES.open() as handle:
        public_rows = [json.loads(line) for line in handle if line.strip()]
    if len(public_rows) != 1693:
        raise ValueError(f"expected 1693 public episodes, found {len(public_rows)}")
    selected = public_rows[: len(FILE_EVIDENCE)]
    lengths = [int(row["length"]) for row in selected]
    sizes = [int(row["size"]) for row in FILE_EVIDENCE]

    buggy = simulate_v21_to_v30_offsets(
        episode_lengths=lengths,
        episode_sizes_bytes=sizes,
        target_file_size_mb=TARGET_FILE_SIZE_MB,
        reset_global_frame_count=True,
    )
    fixed = simulate_v21_to_v30_offsets(
        episode_lengths=lengths,
        episode_sizes_bytes=sizes,
        target_file_size_mb=TARGET_FILE_SIZE_MB,
        reset_global_frame_count=False,
    )
    buggy_check = check_episode_index_chain(buggy)
    fixed_check = check_episode_index_chain(fixed)
    summary = {
        "case": "LeRobot PR #2057: v2.1-to-v3 global frame offset reset",
        "evidence_class": "merged upstream bugfix reproduced with public v2.1 metadata and file sizes",
        "full_upstream_converter_execution": False,
        "episodes_evaluated": len(selected),
        "target_file_size_mb": TARGET_FILE_SIZE_MB,
        "first_output_file_closes_after_episode": 2,
        "buggy_contract_accepted": buggy_check.accepted,
        "buggy_contract_violations": len(buggy_check.issues),
        "fixed_contract_accepted": fixed_check.accepted,
        "fixed_contract_violations": len(fixed_check.issues),
        "reproduced": not buggy_check.accepted and fixed_check.accepted,
    }
    payload = {
        "summary": summary,
        "buggy_rows": buggy,
        "buggy_issues": buggy_check.issues,
        "fixed_rows": fixed,
        "fixed_issues": fixed_check.issues,
        "public_episode_metadata": selected,
        "file_lfs_evidence": FILE_EVIDENCE,
        "frozen_inputs": {
            "dataset": "HuggingFaceVLA/libero",
            "dataset_revision": DATASET_REVISION,
            "episodes_sha256": EPISODES_SHA256,
            "merged_fix_commit": MERGED_FIX_COMMIT,
        },
        "sources": {
            "pull_request": "https://github.com/huggingface/lerobot/pull/2057",
            "release_notes": "https://github.com/huggingface/lerobot/releases",
            "dataset": "https://huggingface.co/datasets/HuggingFaceVLA/libero",
        },
        "limitations": [
            "This executes the historical metadata-index loop, not the full converter dependency stack.",
            "The first five public episode lengths and actual LFS sizes are sufficient to cross the default 100 MiB file boundary.",
            "The contract checks cross-row global continuity; each faulty row remains individually well typed and length-consistent.",
        ],
    }
    output = ROOT / "results" / "real-defect-lerobot-2057.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("buggy issues:")
    for issue in buggy_check.issues:
        print(f"- {issue}")
    print(f"wrote {output}")
    return 0 if summary["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

