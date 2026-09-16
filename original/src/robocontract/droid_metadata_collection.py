"""Collection-level CPU audit for sharded Cosmos3-DROID metadata.

The single-shard audit checks each Parquet artifact deeply.  This module adds
the obligations that only become visible after all episode-metadata shards are
viewed together: global episode/frame continuity and the relationship between
per-episode task-index summaries and the companion task vocabulary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq

from .droid_metadata_audit import audit_droid_metadata, sha256_file


COSMOS3_DROID_REVISION = "dabaaffe428d67cf93fd355b82658934ee59bfec"
KNOWN_EPISODE_METADATA: dict[str, dict[str, tuple[int, str]]] = {
    "success": {
        "file-000.parquet": (
            80_080_877,
            "34c4593b97aa4696580ef974b77c711395cec22ed22ae99ef70b2bf4101b9917",
        ),
        "file-001.parquet": (
            61_590_380,
            "ee66a83456b1b92667955b893bfec38204459c82577563cd34080ced888d234b",
        ),
        "file-002.parquet": (
            58_286_935,
            "38a9226fb6fc8c2d2b6aaa7e4238bc8cc6321131fd2f06b951bec0f232f3e042",
        ),
        "file-003.parquet": (
            71_074_101,
            "b5215e4461bc6a3b7e6586f0c2773d8fb55eb3aa85004f35affcc565d792b508",
        ),
        "file-004.parquet": (
            39_789_260,
            "c60734b53bef8c913665802ff18a11f1c4fe64b09ee4e8418eaaa148d9db9a7d",
        ),
    },
    "failure": {
        "file-000.parquet": (
            75_911_196,
            "45243e378c44b8fad4a6d52e1bf7e5dec4f645390ab66fd6593fc2e44cd2be73",
        ),
    },
}


def _scalar(value: Any) -> float:
    values = np.asarray(value).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"expected scalar statistic, got {values.shape}")
    return float(values[0])


def _close(left: float, right: float, *, atol: float = 1e-5) -> bool:
    return math.isclose(left, right, rel_tol=1e-7, abs_tol=atol)


def _lab(episode_id: str) -> str:
    return episode_id.split("/", 1)[0] if episode_id else "<empty>"


def audit_droid_cross_shard_identity(
    parquet_paths: Iterable[Path], tasks_path: Path
) -> dict[str, Any]:
    """Audit global identity and task references across ordered shards."""

    paths = list(parquet_paths)
    task_table = pq.read_table(tasks_path, columns=["task_index", "task"]).to_pydict()
    global_tasks = {
        int(index): str(task)
        for index, task in zip(task_table["task_index"], task_table["task"])
    }
    columns = [
        "episode_index",
        "episode_id",
        "tasks",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "stats/episode_index/min",
        "stats/episode_index/max",
        "stats/episode_index/mean",
        "stats/episode_index/std",
        "stats/index/min",
        "stats/index/max",
        "stats/index/mean",
        "stats/task_index/min",
        "stats/task_index/max",
        "stats/task_index/std",
    ]

    expected_episode = 0
    expected_frame = 0
    seen_episode_ids: set[str] = set()
    lab_episode_counter: Counter[str] = Counter()
    lab_frame_counter: Counter[str] = Counter()
    lab_task_vocabularies: dict[str, dict[str, int]] = defaultdict(dict)
    lab_rows: Counter[str] = Counter()
    identity_mismatch_by_lab: Counter[str] = Counter()
    task_mismatch_by_lab: Counter[str] = Counter()
    violations: Counter[str] = Counter()
    identity_mismatches = 0
    identity_local_matches = 0
    task_companion_mismatches = 0
    task_local_matches = 0
    total_rows = 0

    for path in paths:
        data = pq.read_table(path, columns=columns).to_pydict()
        order = sorted(
            range(len(data["episode_index"])),
            key=lambda index: int(data["episode_index"][index]),
        )
        for index in order:
            total_rows += 1
            episode_index = int(data["episode_index"][index])
            episode_id = str(data["episode_id"][index])
            length = int(data["length"][index])
            start = int(data["dataset_from_index"][index])
            stop = int(data["dataset_to_index"][index])
            lab = _lab(episode_id)
            lab_rows[lab] += 1

            violations["global_episode_sequence_breaks"] += int(
                episode_index != expected_episode
            )
            violations["global_dataset_chain_breaks"] += int(start != expected_frame)
            violations["dataset_span_length_mismatches"] += int(
                stop - start != length
            )
            violations["duplicate_episode_ids"] += int(episode_id in seen_episode_ids)
            seen_episode_ids.add(episode_id)
            expected_episode = episode_index + 1
            expected_frame = stop

            episode_stats_global = (
                _close(_scalar(data["stats/episode_index/min"][index]), episode_index)
                and _close(
                    _scalar(data["stats/episode_index/max"][index]), episode_index
                )
                and _close(
                    _scalar(data["stats/episode_index/mean"][index]), episode_index
                )
                and _close(_scalar(data["stats/episode_index/std"][index]), 0.0)
            )
            index_stats_global = (
                _close(_scalar(data["stats/index/min"][index]), start)
                and _close(_scalar(data["stats/index/max"][index]), stop - 1)
                and _close(
                    _scalar(data["stats/index/mean"][index]),
                    (start + stop - 1) / 2,
                )
            )
            identity_mismatch = not (episode_stats_global and index_stats_global)
            if identity_mismatch:
                identity_mismatches += 1
                identity_mismatch_by_lab[lab] += 1
                local_identity = (
                    _close(
                        _scalar(data["stats/episode_index/min"][index]),
                        lab_episode_counter[lab],
                    )
                    and _close(
                        _scalar(data["stats/index/min"][index]),
                        lab_frame_counter[lab],
                    )
                )
                identity_local_matches += int(local_identity)

            task_values = data["tasks"][index]
            task_text = str(task_values[0]) if len(task_values) == 1 else ""
            task_min = _scalar(data["stats/task_index/min"][index])
            task_max = _scalar(data["stats/task_index/max"][index])
            task_std = _scalar(data["stats/task_index/std"][index])
            task_constant = _close(task_min, task_max) and _close(task_std, 0.0)
            task_integral = _close(task_min, round(task_min))
            violations["nonconstant_task_index_summaries"] += int(not task_constant)
            violations["nonintegral_task_index_summaries"] += int(not task_integral)
            observed_task = int(round(task_min))
            companion_matches = global_tasks.get(observed_task) == task_text
            if not companion_matches:
                task_companion_mismatches += 1
                task_mismatch_by_lab[lab] += 1

            local_vocabulary = lab_task_vocabularies[lab]
            if task_text not in local_vocabulary:
                local_vocabulary[task_text] = len(local_vocabulary)
            task_local_matches += int(observed_task == local_vocabulary[task_text])

            lab_episode_counter[lab] += 1
            lab_frame_counter[lab] += length

    return {
        "rows_checked": total_rows,
        "final_episode_stop": expected_episode,
        "final_frame_stop": expected_frame,
        "global_continuity_violations": dict(violations),
        "lab_rows": dict(sorted(lab_rows.items())),
        "identity_summary": {
            "global_identity_inconsistent_rows": identity_mismatches,
            "inconsistent_rows_matching_lab_local_pattern": identity_local_matches,
            "mismatch_by_lab": dict(sorted(identity_mismatch_by_lab.items())),
        },
        "task_referential_summary": {
            "companion_task_text_mismatch_rows": task_companion_mismatches,
            "rows_matching_lab_local_task_vocabulary": task_local_matches,
            "mismatch_by_lab": dict(sorted(task_mismatch_by_lab.items())),
            "global_task_rows": len(global_tasks),
            "lab_local_vocabulary_sizes": {
                lab: len(vocabulary)
                for lab, vocabulary in sorted(lab_task_vocabularies.items())
            },
        },
    }


def audit_droid_cross_shard_video_lineage(
    parquet_paths: Iterable[Path], *, fps: float
) -> dict[str, Any]:
    """Audit video-file interval chains after joining metadata shards.

    A video file may straddle two episode-metadata Parquet shards.  Checking
    each metadata shard alone therefore produces false "non-zero start"
    findings; the file identity has to be grouped across the collection.
    """

    paths = list(parquet_paths)
    names = pq.ParquetFile(paths[0]).schema_arrow.names
    suffix = "/from_timestamp"
    prefixes = sorted(
        name[: -len(suffix)]
        for name in names
        if name.startswith("videos/") and name.endswith(suffix)
    )
    per_camera: dict[str, Any] = {}
    total_violations: Counter[str] = Counter()
    for prefix in prefixes:
        columns = [
            "length",
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
            f"{prefix}/from_timestamp",
            f"{prefix}/to_timestamp",
        ]
        groups: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
        violations: Counter[str] = Counter()
        rows = 0
        for path in paths:
            data = pq.read_table(path, columns=columns).to_pydict()
            for index, length in enumerate(data["length"]):
                rows += 1
                chunk = int(data[f"{prefix}/chunk_index"][index])
                file_index = int(data[f"{prefix}/file_index"][index])
                start = float(data[f"{prefix}/from_timestamp"][index])
                stop = float(data[f"{prefix}/to_timestamp"][index])
                violations["invalid_or_reversed_intervals"] += int(
                    not (math.isfinite(start) and math.isfinite(stop)) or stop < start
                )
                violations["negative_chunk_or_file_indices"] += int(
                    chunk < 0 or file_index < 0
                )
                violations["duration_vs_length_fps_mismatches"] += int(
                    not _close(stop - start, int(length) / fps, atol=1e-6)
                )
                groups[(chunk, file_index)].append((start, stop))
        for intervals in groups.values():
            intervals.sort()
            violations["files_not_starting_at_zero"] += int(
                not _close(intervals[0][0], 0.0, atol=1e-6)
            )
            for previous, current in zip(intervals, intervals[1:]):
                delta = current[0] - previous[1]
                violations["within_file_chain_breaks"] += int(abs(delta) > 1e-6)
                violations["within_file_overlaps"] += int(delta < -1e-6)
        total_violations.update(violations)
        per_camera[prefix.removeprefix("videos/")] = {
            "rows_checked": rows,
            "file_count": len(groups),
            "violations": dict(violations),
        }
    return {
        "camera_count": len(prefixes),
        "violations": dict(total_violations),
        "per_camera": per_camera,
        "status": "pass" if not any(total_violations.values()) else "violations",
    }


def audit_droid_metadata_collection(split_root: Path, *, split: str) -> dict[str, Any]:
    """Run all single-shard checks plus collection-level contracts."""

    started = time.perf_counter()
    split_root = split_root.resolve()
    info_path = split_root / "info.json"
    stats_path = split_root / "stats.json"
    tasks_path = split_root / "tasks.parquet"
    parquet_paths = sorted((split_root / "episodes").glob("file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no episode metadata below {split_root}")
    info = json.loads(info_path.read_text())

    per_shard = []
    manifests = []
    known = KNOWN_EPISODE_METADATA.get(split, {})
    for path in parquet_paths:
        result = audit_droid_metadata(
            path,
            info_path=info_path,
            tasks_path=tasks_path,
            stats_path=stats_path,
        )
        per_shard.append(result)
        expected = known.get(path.name)
        digest = result["contracts"]["D0"]["provenance"]["sha256"]
        size = path.stat().st_size
        manifests.append(
            {
                "local_path": str(path),
                "remote_path": (
                    f"{split}/meta/episodes/chunk-000/{path.name}"
                ),
                "size_bytes": size,
                "sha256": digest,
                "expected_size_bytes": expected[0] if expected else None,
                "expected_sha256": expected[1] if expected else None,
                "known_revision_match": bool(
                    expected and size == expected[0] and digest == expected[1]
                ),
            }
        )

    cross_shard = audit_droid_cross_shard_identity(parquet_paths, tasks_path)
    cross_shard_video = audit_droid_cross_shard_video_lineage(
        parquet_paths, fps=float(info["fps"])
    )
    rows = sum(result["summary"]["rows"] for result in per_shard)
    expected_rows = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    expected_tasks = int(info["total_tasks"])
    task_rows = pq.ParquetFile(tasks_path).metadata.num_rows
    aggregate = {
        contract: dict(
            Counter(
                {
                    key: sum(
                        int(
                            result["contracts"][contract]
                            .get("violations", {})
                            .get(key, 0)
                        )
                        for result in per_shard
                    )
                    for key in {
                        name
                        for result in per_shard
                        for name in result["contracts"][contract]
                        .get("violations", {})
                    }
                }
            )
        )
        for contract in ("D1", "D2", "D3")
    }
    declared_count_checks = {
        "episode_rows_match_info": rows == expected_rows,
        "final_frame_stop_matches_info": (
            cross_shard["final_frame_stop"] == expected_frames
        ),
        "task_rows_match_info": task_rows == expected_tasks,
        "all_known_shards_present": set(known) == {path.name for path in parquet_paths},
        "all_known_hashes_match": all(
            entry["known_revision_match"] for entry in manifests
        ),
    }
    return {
        "audit_version": "1",
        "dataset": {
            "repo_id": "nvidia/Cosmos3-DROID",
            "revision": COSMOS3_DROID_REVISION,
            "split": split,
            "scope": "complete episode metadata; trajectories and videos excluded",
        },
        "summary": {
            "episode_rows": rows,
            "declared_frames_represented": expected_frames,
            "task_rows": task_rows,
            "episode_metadata_shards": len(parquet_paths),
            "identity_inconsistent_rows": cross_shard["identity_summary"][
                "global_identity_inconsistent_rows"
            ],
            "task_companion_mismatch_rows": cross_shard[
                "task_referential_summary"
            ]["companion_task_text_mismatch_rows"],
            "declared_count_checks_passed": all(declared_count_checks.values()),
            "gpu_used": False,
            "elapsed_s": time.perf_counter() - started,
        },
        "declared_count_checks": declared_count_checks,
        "cross_shard": cross_shard,
        "cross_shard_video_lineage": cross_shard_video,
        "aggregate_single_shard_violations": aggregate,
        "manifests": {
            "episode_metadata": manifests,
            "companions": {
                name: {
                    "local_path": str(path),
                    "remote_path": f"{split}/meta/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for name, path in (
                    ("info", info_path),
                    ("stats", stats_path),
                    ("tasks", tasks_path),
                )
            },
        },
        "per_shard": per_shard,
        "limitations": [
            "No trajectory, image, video, calibration, or task-success value is decoded.",
            "A mismatch is a metadata referential inconsistency, not proof of downstream policy harm.",
            "Rows sharing one conversion pattern are affected records, not independent root-cause bugs.",
            "Per-shard video file-start counts are partial views; collection-level video lineage is authoritative.",
        ],
    }
