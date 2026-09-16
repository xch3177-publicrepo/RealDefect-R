"""CPU-only integrity audit for LeRobot v3 DROID episode metadata.

The audit deliberately operates on episode metadata and summary statistics.  It
does not claim to validate the underlying trajectories, videos, calibration, or
task success without those artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq


KNOWN_COSMOS3_DROID = {
    "repo_id": "nvidia/Cosmos3-DROID",
    "revision": "dabaaffe428d67cf93fd355b82658934ee59bfec",
    "path": "success/meta/episodes/chunk-000/file-000.parquet",
    "size_bytes": 80_080_877,
    "sha256": "34c4593b97aa4696580ef974b77c711395cec22ed22ae99ef70b2bf4101b9917",
    "rows": 14_904,
    "columns": 222,
    "fps": 15.0,
}

STAT_METRICS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
ORDERED_STAT_METRICS = ("min", "q01", "q10", "q50", "q90", "q99", "max")
REQUIRED_EPISODE_COLUMNS = (
    "episode_index",
    "episode_id",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
    "meta/episodes/chunk_index",
    "meta/episodes/file_index",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def estimate_image_sample_count(length: int) -> int:
    """Mirror LeRobot's episode-image statistics sampling heuristic."""

    minimum = min(length, 100)
    return max(minimum, min(int(length**0.75), 10_000))


def _scalar(value: Any) -> float:
    array = np.asarray(value).reshape(-1)
    if len(array) != 1:
        raise ValueError(f"expected scalar statistic, found shape {np.asarray(value).shape}")
    return float(array[0])


def _close(actual: float, expected: float, *, atol: float = 1e-5) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-7, abs_tol=atol)


def _lab_from_episode_id(episode_id: str) -> str:
    return episode_id.split("/", 1)[0] if episode_id else "<empty>"


def _contract_status(violations: dict[str, int]) -> str:
    return "pass" if not any(violations.values()) else "violations"


def _stats_feature_map(column_names: list[str]) -> dict[str, set[str]]:
    features: dict[str, set[str]] = defaultdict(set)
    for column in column_names:
        if not column.startswith("stats/"):
            continue
        feature_metric = column[len("stats/") :]
        if "/" not in feature_metric:
            continue
        feature, metric = feature_metric.rsplit("/", 1)
        features[feature].add(metric)
    return dict(features)


def _video_prefixes(column_names: list[str]) -> list[str]:
    suffix = "/from_timestamp"
    return sorted(
        column[: -len(suffix)]
        for column in column_names
        if column.startswith("videos/") and column.endswith(suffix)
    )


def _audit_d0(path: Path, parquet: pq.ParquetFile, digest: str) -> dict[str, Any]:
    metadata = parquet.metadata
    names = parquet.schema_arrow.names
    missing = sorted(set(REQUIRED_EPISODE_COLUMNS) - set(names))
    known_hash = digest == KNOWN_COSMOS3_DROID["sha256"]
    violations = {
        "missing_required_columns": len(missing),
        "known_profile_size_mismatch": int(
            known_hash and path.stat().st_size != KNOWN_COSMOS3_DROID["size_bytes"]
        ),
        "known_profile_row_count_mismatch": int(
            known_hash and metadata.num_rows != KNOWN_COSMOS3_DROID["rows"]
        ),
        "known_profile_column_count_mismatch": int(
            known_hash and metadata.num_columns != KNOWN_COSMOS3_DROID["columns"]
        ),
    }
    return {
        "status": _contract_status(violations),
        "violations": violations,
        "missing_required_columns": missing,
        "parquet": {
            "rows": metadata.num_rows,
            "columns": metadata.num_columns,
            "row_groups": metadata.num_row_groups,
            "created_by": metadata.created_by,
        },
        "provenance": {
            "sha256": digest,
            "size_bytes": path.stat().st_size,
            "known_profile_match": known_hash,
            "known_profile": KNOWN_COSMOS3_DROID if known_hash else None,
        },
    }


def _audit_d1(parquet: pq.ParquetFile) -> dict[str, Any]:
    names = set(parquet.schema_arrow.names)
    stat_columns = [
        f"stats/{feature}/{metric}"
        for feature in ("episode_index", "index")
        for metric in ("min", "max", "mean", "std")
    ]
    required = list(REQUIRED_EPISODE_COLUMNS) + stat_columns
    missing = sorted(set(required) - names)
    if missing:
        return {
            "status": "skipped",
            "reason": "required lineage columns are absent",
            "missing_columns": missing,
        }

    data = parquet.read(columns=required).to_pydict()
    row_count = len(data["episode_index"])
    ordered = sorted(range(row_count), key=lambda i: int(data["episode_index"][i]))

    episode_values = [int(data["episode_index"][i]) for i in range(row_count)]
    episode_ids = [str(data["episode_id"][i]) for i in range(row_count)]
    violations = {
        "duplicate_episode_indices": row_count - len(set(episode_values)),
        "duplicate_episode_ids": row_count - len(set(episode_ids)),
        "empty_episode_ids": sum(not value for value in episode_ids),
        "empty_tasks": sum(not data["tasks"][i] for i in range(row_count)),
        "nonpositive_lengths": sum(int(data["length"][i]) <= 0 for i in range(row_count)),
        "episode_index_gaps": sum(
            episode_values[ordered[j]] != episode_values[ordered[j - 1]] + 1
            for j in range(1, row_count)
        ),
        "dataset_span_length_mismatches": sum(
            int(data["dataset_to_index"][i]) - int(data["dataset_from_index"][i])
            != int(data["length"][i])
            for i in range(row_count)
        ),
        "dataset_chain_breaks": sum(
            int(data["dataset_from_index"][ordered[j]])
            != int(data["dataset_to_index"][ordered[j - 1]])
            for j in range(1, row_count)
        ),
        "negative_chunk_or_file_indices": sum(
            any(
                int(data[column][i]) < 0
                for column in (
                    "data/chunk_index",
                    "data/file_index",
                    "meta/episodes/chunk_index",
                    "meta/episodes/file_index",
                )
            )
            for i in range(row_count)
        ),
    }

    local_episode: dict[str, int] = defaultdict(int)
    local_frame: dict[str, int] = defaultdict(int)
    episode_stat_mismatch: set[int] = set()
    index_location_mismatch: set[int] = set()
    index_std_mismatch: set[int] = set()
    local_pattern: set[int] = set()
    mismatch_by_lab: Counter[str] = Counter()

    for i in ordered:
        episode_index = int(data["episode_index"][i])
        length = int(data["length"][i])
        start = int(data["dataset_from_index"][i])
        stop = int(data["dataset_to_index"][i])
        lab = _lab_from_episode_id(str(data["episode_id"][i]))
        expected_index_std = math.sqrt((length * length - 1) / 12) if length > 1 else 0.0

        episode_consistent = (
            _close(_scalar(data["stats/episode_index/min"][i]), episode_index)
            and _close(_scalar(data["stats/episode_index/max"][i]), episode_index)
            and _close(_scalar(data["stats/episode_index/mean"][i]), episode_index)
            and _close(_scalar(data["stats/episode_index/std"][i]), 0.0)
        )
        index_location_consistent = (
            _close(_scalar(data["stats/index/min"][i]), start)
            and _close(_scalar(data["stats/index/max"][i]), stop - 1)
            and _close(_scalar(data["stats/index/mean"][i]), (start + stop - 1) / 2)
        )
        index_std_consistent = _close(
            _scalar(data["stats/index/std"][i]), expected_index_std
        )
        if not episode_consistent:
            episode_stat_mismatch.add(i)
        if not index_location_consistent:
            index_location_mismatch.add(i)
        if not index_std_consistent:
            index_std_mismatch.add(i)

        local_matches = (
            _close(_scalar(data["stats/episode_index/min"][i]), local_episode[lab])
            and _close(_scalar(data["stats/index/min"][i]), local_frame[lab])
        )
        if local_matches:
            local_pattern.add(i)
        local_episode[lab] += 1
        local_frame[lab] += length

    # A source-rebasing inconsistency changes the identity/location statistics.
    # Keep a pure std deviation separate: the public file contains two rows with
    # small floating-point aggregation error but correct global index locations.
    inconsistent = episode_stat_mismatch | index_location_mismatch
    any_index_stat_mismatch = index_location_mismatch | index_std_mismatch
    for i in inconsistent:
        mismatch_by_lab[_lab_from_episode_id(str(data["episode_id"][i]))] += 1

    violations.update(
        {
            "episode_index_stat_inconsistent_rows": len(episode_stat_mismatch),
            "global_index_location_inconsistent_rows": len(index_location_mismatch),
            "global_index_std_analytic_mismatch_rows": len(index_std_mismatch),
            "global_index_any_stat_inconsistent_rows": len(any_index_stat_mismatch),
            "local_vs_global_index_stat_inconsistent_rows": len(inconsistent),
        }
    )
    return {
        "status": _contract_status(violations),
        "violations": violations,
        "row_count": row_count,
        "index_stat_diagnosis": {
            "inconsistent_rows": len(inconsistent),
            "inconsistent_rows_matching_lab_local_pattern": len(inconsistent & local_pattern),
            "mismatch_by_lab": dict(sorted(mismatch_by_lab.items())),
        },
    }


def _audit_d2(parquet: pq.ParquetFile, *, fps: float | None) -> dict[str, Any]:
    names = parquet.schema_arrow.names
    prefixes = _video_prefixes(names)
    if not prefixes:
        return {"status": "skipped", "reason": "no video lineage columns found"}

    required = ["episode_index", "length"]
    for prefix in prefixes:
        required.extend(
            [
                f"{prefix}/chunk_index",
                f"{prefix}/file_index",
                f"{prefix}/from_timestamp",
                f"{prefix}/to_timestamp",
            ]
        )
    missing = sorted(set(required) - set(names))
    if missing:
        return {
            "status": "skipped",
            "reason": "video lineage columns are incomplete",
            "missing_columns": missing,
        }

    data = parquet.read(columns=required).to_pydict()
    row_count = len(data["episode_index"])
    per_camera: dict[str, Any] = {}
    totals = Counter()
    for prefix in prefixes:
        invalid_time = 0
        duration_mismatch = 0
        negative_mapping = 0
        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i in range(row_count):
            chunk = int(data[f"{prefix}/chunk_index"][i])
            file_index = int(data[f"{prefix}/file_index"][i])
            start = float(data[f"{prefix}/from_timestamp"][i])
            stop = float(data[f"{prefix}/to_timestamp"][i])
            invalid_time += int(not (math.isfinite(start) and math.isfinite(stop)) or stop < start)
            negative_mapping += int(chunk < 0 or file_index < 0)
            if fps is not None:
                duration_mismatch += int(not _close(stop - start, int(data["length"][i]) / fps, atol=1e-6))
            groups[(chunk, file_index)].append(i)

        nonzero_starts = 0
        chain_breaks = 0
        overlaps = 0
        for indices in groups.values():
            indices.sort(key=lambda i: float(data[f"{prefix}/from_timestamp"][i]))
            nonzero_starts += int(not _close(float(data[f"{prefix}/from_timestamp"][indices[0]]), 0.0, atol=1e-6))
            for previous, current in zip(indices, indices[1:]):
                delta = float(data[f"{prefix}/from_timestamp"][current]) - float(
                    data[f"{prefix}/to_timestamp"][previous]
                )
                chain_breaks += int(abs(delta) > 1e-6)
                overlaps += int(delta < -1e-6)

        camera_violations = {
            "invalid_or_reversed_intervals": invalid_time,
            "duration_vs_length_fps_mismatches": duration_mismatch,
            "negative_chunk_or_file_indices": negative_mapping,
            "files_not_starting_at_zero": nonzero_starts,
            "within_file_chain_breaks": chain_breaks,
            "within_file_overlaps": overlaps,
        }
        totals.update(camera_violations)
        per_camera[prefix.removeprefix("videos/")] = {
            "file_count": len(groups),
            "violations": camera_violations,
        }

    violations = dict(totals)
    return {
        "status": _contract_status(violations),
        "fps": fps,
        "violations": violations,
        "per_camera": per_camera,
        "note": "Camera file identities and timestamp offsets are intentionally audited independently.",
    }


def _audit_d3(parquet: pq.ParquetFile, *, fps: float | None, batch_size: int = 128) -> dict[str, Any]:
    names = parquet.schema_arrow.names
    feature_map = _stats_feature_map(names)
    if not feature_map:
        return {"status": "skipped", "reason": "no episode statistics columns found"}

    incomplete_features = {
        feature: sorted(set(STAT_METRICS) - metrics)
        for feature, metrics in feature_map.items()
        if set(STAT_METRICS) - metrics
    }
    complete_features = sorted(set(feature_map) - set(incomplete_features))
    columns = ["length"] + [
        f"stats/{feature}/{metric}" for feature in complete_features for metric in STAT_METRICS
    ]
    violations = Counter(
        {
            "features_missing_metrics": len(incomplete_features),
            "stat_shape_mismatches": 0,
            "nonfinite_stat_values": 0,
            "nonmonotone_quantiles": 0,
            "mean_outside_min_max": 0,
            "negative_standard_deviation": 0,
            "sample_count_mismatches": 0,
            "frame_index_analytic_mismatches": 0,
            "timestamp_analytic_mismatches": 0,
            "nonconstant_task_index_stats": 0,
        }
    )
    checked_rows = 0
    checked_feature_rows = 0

    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            checked_rows += 1
            length = int(data["length"][i])
            for feature in complete_features:
                checked_feature_rows += 1
                values = {
                    metric: np.asarray(data[f"stats/{feature}/{metric}"][i], dtype=float).reshape(-1)
                    for metric in STAT_METRICS
                }
                canonical_dim = len(values["min"])
                if canonical_dim == 0 or len(values["count"]) != 1 or any(
                    len(values[metric]) != canonical_dim for metric in STAT_METRICS if metric != "count"
                ):
                    violations["stat_shape_mismatches"] += 1
                if any(not np.isfinite(value).all() for value in values.values()):
                    violations["nonfinite_stat_values"] += 1
                ordered = [values[metric] for metric in ORDERED_STAT_METRICS]
                if any(np.any(left > right + 1e-9) for left, right in zip(ordered, ordered[1:])):
                    violations["nonmonotone_quantiles"] += 1
                if np.any(values["mean"] < values["min"] - 1e-5) or np.any(
                    values["mean"] > values["max"] + 1e-5
                ):
                    violations["mean_outside_min_max"] += 1
                if np.any(values["std"] < -1e-12):
                    violations["negative_standard_deviation"] += 1

                expected_count = (
                    estimate_image_sample_count(length)
                    if ".image." in feature or feature.startswith("observation.image")
                    else length
                )
                if len(values["count"]) != 1 or int(values["count"][0]) != expected_count:
                    violations["sample_count_mismatches"] += 1

            if "frame_index" in complete_features:
                expected_std = math.sqrt((length * length - 1) / 12) if length > 1 else 0.0
                frame_ok = (
                    _close(_scalar(data["stats/frame_index/min"][i]), 0.0)
                    and _close(_scalar(data["stats/frame_index/max"][i]), length - 1)
                    and _close(_scalar(data["stats/frame_index/mean"][i]), (length - 1) / 2)
                    and _close(_scalar(data["stats/frame_index/std"][i]), expected_std)
                )
                violations["frame_index_analytic_mismatches"] += int(not frame_ok)
            if fps is not None and "timestamp" in complete_features:
                expected_std = (
                    math.sqrt((length * length - 1) / 12) / fps if length > 1 else 0.0
                )
                timestamp_ok = (
                    _close(_scalar(data["stats/timestamp/min"][i]), 0.0)
                    and _close(_scalar(data["stats/timestamp/max"][i]), (length - 1) / fps)
                    and _close(_scalar(data["stats/timestamp/mean"][i]), (length - 1) / (2 * fps))
                    and _close(_scalar(data["stats/timestamp/std"][i]), expected_std)
                )
                violations["timestamp_analytic_mismatches"] += int(not timestamp_ok)
            if "task_index" in complete_features:
                task_ok = (
                    _close(
                        _scalar(data["stats/task_index/min"][i]),
                        _scalar(data["stats/task_index/max"][i]),
                    )
                    and _close(_scalar(data["stats/task_index/std"][i]), 0.0)
                )
                violations["nonconstant_task_index_stats"] += int(not task_ok)

    violation_dict = dict(violations)
    return {
        "status": _contract_status(violation_dict),
        "violations": violation_dict,
        "feature_count": len(feature_map),
        "complete_feature_count": len(complete_features),
        "incomplete_features": incomplete_features,
        "rows_checked": checked_rows,
        "feature_rows_checked": checked_feature_rows,
        "image_count_rule": "LeRobot estimate_num_samples(length), not raw episode length",
    }


def _audit_d4(
    *,
    parquet: pq.ParquetFile,
    info_path: Path | None,
    tasks_path: Path | None,
    stats_path: Path | None,
) -> dict[str, Any]:
    companions = {"info": info_path, "tasks": tasks_path, "stats": stats_path}
    missing = [name for name, path in companions.items() if path is None or not path.exists()]
    if missing:
        return {
            "status": "skipped",
            "reason": "companion metadata not supplied or not present",
            "missing_companions": missing,
            "provided_paths": {name: str(path) if path is not None else None for name, path in companions.items()},
        }

    assert info_path is not None and tasks_path is not None and stats_path is not None
    info = json.loads(info_path.read_text())
    global_stats = json.loads(stats_path.read_text())
    tasks = pq.read_table(tasks_path)
    declared_features = set(info.get("features", {}))
    episode_stat_features = set(_stats_feature_map(parquet.schema_arrow.names))
    missing_from_info = sorted(episode_stat_features - declared_features)
    missing_from_global_stats = sorted(episode_stat_features - set(global_stats))
    violations = {
        "episode_stat_features_missing_from_info": len(missing_from_info),
        "episode_stat_features_missing_from_global_stats": len(missing_from_global_stats),
        "empty_tasks_table": int(tasks.num_rows == 0),
    }
    return {
        "status": _contract_status(violations),
        "violations": violations,
        "info_codebase_version": info.get("codebase_version"),
        "declared_fps": info.get("fps"),
        "tasks_rows": tasks.num_rows,
        "missing_from_info": missing_from_info,
        "missing_from_global_stats": missing_from_global_stats,
        "scope_note": "D4 checks companion referential coverage; raw trajectory recomputation is out of scope.",
    }


def audit_droid_metadata(
    parquet_path: Path,
    *,
    info_path: Path | None = None,
    tasks_path: Path | None = None,
    stats_path: Path | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    """Audit a DROID LeRobot v3 episode-metadata Parquet file."""

    started = time.perf_counter()
    parquet_path = parquet_path.resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    stage_timings: dict[str, float] = {}
    stage_started = time.perf_counter()
    digest = sha256_file(parquet_path)
    parquet = pq.ParquetFile(parquet_path)
    d0 = _audit_d0(parquet_path, parquet, digest)
    stage_timings["D0"] = (time.perf_counter() - stage_started) * 1000

    effective_fps = fps
    if effective_fps is None and digest == KNOWN_COSMOS3_DROID["sha256"]:
        effective_fps = float(KNOWN_COSMOS3_DROID["fps"])
    if effective_fps is None and info_path is not None and info_path.exists():
        effective_fps = float(json.loads(info_path.read_text())["fps"])

    stage_started = time.perf_counter()
    d1 = _audit_d1(parquet)
    stage_timings["D1"] = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    d2 = _audit_d2(parquet, fps=effective_fps)
    stage_timings["D2"] = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    d3 = _audit_d3(parquet, fps=effective_fps)
    stage_timings["D3"] = (time.perf_counter() - stage_started) * 1000
    stage_started = time.perf_counter()
    d4 = _audit_d4(
        parquet=parquet,
        info_path=info_path,
        tasks_path=tasks_path,
        stats_path=stats_path,
    )
    stage_timings["D4"] = (time.perf_counter() - stage_started) * 1000

    total_ms = (time.perf_counter() - started) * 1000
    index_inconsistencies = int(
        d1.get("violations", {}).get("local_vs_global_index_stat_inconsistent_rows", 0)
    )
    companion_manifest = {}
    for name, path in (
        ("info", info_path),
        ("tasks", tasks_path),
        ("stats", stats_path),
    ):
        if path is not None and path.exists():
            companion_manifest[name] = {
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            companion_manifest[name] = None
    return {
        "audit_version": "1",
        "scope": "episode metadata and summary statistics only; no trajectory or video validation",
        "input": {
            "parquet": str(parquet_path),
            "info": str(info_path.resolve()) if info_path is not None and info_path.exists() else None,
            "tasks": str(tasks_path.resolve()) if tasks_path is not None and tasks_path.exists() else None,
            "stats": str(stats_path.resolve()) if stats_path is not None and stats_path.exists() else None,
            "fps_used": effective_fps,
            "companion_manifest": companion_manifest,
        },
        "summary": {
            "rows": parquet.metadata.num_rows,
            "columns": parquet.metadata.num_columns,
            "local_vs_global_index_stat_inconsistent_rows": index_inconsistencies,
            "natural_metadata_inconsistency_detected": index_inconsistencies > 0,
            "audit_execution_succeeded": True,
            "gpu_used": False,
        },
        "contracts": {"D0": d0, "D1": d1, "D2": d2, "D3": d3, "D4": d4},
        "timing_ms": {**stage_timings, "total": total_ms},
    }
