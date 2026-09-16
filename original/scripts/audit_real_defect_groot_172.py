#!/usr/bin/env python3
"""Post-run integrity audit for the completed P13 RD7 GR00T replay.

The auditor reads only the compact formal result, immutable per-video JSON
checkpoints, the event ledger, and the frozen repository inputs.  It never
opens a video, imports a decoder, contacts Hugging Face, or mutates the formal
run directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "real-defect-groot-172-audit.json"
DEFAULT_MANIFEST = ROOT / "contracts" / "real-defect-corpus-manifest.json"
DEFAULT_AMENDMENT = (
    ROOT / "contracts" / "real-defect-corpus-freeze-amendment-v1.json"
)
DEFAULT_SIDECAR = ROOT / "contracts" / "p13-groot-mp4-candidates.txt.gz"
DEFAULT_FORMAL_SCRIPT = ROOT / "scripts" / "run_real_defect_groot_172.py"

AUDIT_ID = "P13-RD7-GROOT-172-POSTRUN-AUDIT-V1"
CASE_ID = "RD7-GR00T-172-373"
EXPECTED_MANIFEST_SHA256 = (
    "7255524dc1776ef3e96e12fceea138ac76ae3bb15c55df42dc7e5a9e402e0d17"
)
EXPECTED_AMENDMENT_SHA256 = (
    "58ad7063ddd44872d8fa1535e585f39079537d0a0715967f4784adf6d2adbd0c"
)
EXPECTED_SIDECAR_SHA256 = (
    "f7bbec200ec62622cf8355f2b4f1bf151f710c14425dc6ade466fee7e75891ba"
)
EXPECTED_CANDIDATE_PATHS_SHA256 = (
    "48d1e500f907ce7ec70ccb71f270c9db8dfd06f757e19d4c9a4de6eafc58fe4a"
)
ZERO_HASH = "0" * 64
GROUPS = (
    "regular",
    "sparse",
    "late",
    "continuous_pts_negative_control",
)
OUTSIDE_GROUPS = ("regular", "sparse", "late")


class AuditFailure(RuntimeError):
    """A formal evidence invariant did not verify."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditFailure(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "fraction": f"{value.numerator}/{value.denominator}",
        "seconds": float(value),
    }


def _parse_fraction_record(value: Any, label: str) -> Fraction:
    _require(isinstance(value, dict), f"{label} is not a fraction record")
    _require(set(value) == {"fraction", "seconds"}, f"{label} fields drifted")
    try:
        exact = Fraction(value["fraction"])
        seconds = float(value["seconds"])
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise AuditFailure(f"invalid {label}: {error}") from error
    _require(math.isfinite(seconds), f"{label} seconds is non-finite")
    _require(seconds == float(exact), f"{label} exact/float values disagree")
    return exact


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    errors = [
        float(row["absolute_timestamp_error_seconds"])
        for row in rows
        if row["absolute_timestamp_error_seconds"] is not None
    ]
    correct_count = sum(bool(row["correct"]) for row in rows)
    total = len(rows)
    return {
        "query_count": total,
        "correct_count": correct_count,
        "wrong_frame_count": total - correct_count,
        "exact_selected_frame_accuracy": correct_count / total if total else None,
        "wrong_frame_rate": (total - correct_count) / total if total else None,
        "missing_return_count": sum(row["actual"] is None for row in rows),
        "median_absolute_timestamp_error_seconds": statistics.median(errors)
        if errors
        else None,
        "p95_absolute_timestamp_error_seconds": _percentile(errors, 0.95),
        "max_absolute_timestamp_error_seconds": max(errors) if errors else None,
        "errors_exceeding_one_local_frame_interval": sum(
            bool(row["exceeds_one_local_frame_interval"]) for row in rows
        ),
    }


def _verify_frozen_inputs(
    result: dict[str, Any],
    manifest_path: Path,
    amendment_path: Path,
    sidecar_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    manifest, manifest_raw = _load_json(manifest_path, "canonical manifest")
    amendment, amendment_raw = _load_json(amendment_path, "freeze amendment")
    manifest_sha256 = _bytes_sha256(manifest_raw)
    amendment_sha256 = _bytes_sha256(amendment_raw)
    _require(
        manifest_sha256 == EXPECTED_MANIFEST_SHA256,
        "canonical manifest complete-file SHA-256 mismatch",
    )
    _require(
        amendment_sha256 == EXPECTED_AMENDMENT_SHA256,
        "freeze amendment complete-file SHA-256 mismatch",
    )
    try:
        compressed_sidecar = sidecar_path.read_bytes()
        uncompressed_sidecar = gzip.decompress(compressed_sidecar)
        candidate_paths = uncompressed_sidecar.decode("utf-8").splitlines()
    except (OSError, gzip.BadGzipFile, UnicodeDecodeError) as error:
        raise AuditFailure(f"cannot verify candidate sidecar: {error}") from error
    compressed_sha256 = _bytes_sha256(compressed_sidecar)
    uncompressed_sha256 = _bytes_sha256(uncompressed_sidecar)
    _require(
        compressed_sha256 == EXPECTED_SIDECAR_SHA256,
        "candidate sidecar compressed SHA-256 mismatch",
    )
    _require(
        uncompressed_sha256 == EXPECTED_CANDIDATE_PATHS_SHA256,
        "candidate sidecar uncompressed SHA-256 mismatch",
    )
    _require(len(candidate_paths) == 50656, "candidate sidecar count is not 50,656")
    _require(candidate_paths == sorted(set(candidate_paths)), "candidate paths drifted")

    _require(manifest.get("corpus_id") == "P13-REALDEFECT-8", "corpus ID drifted")
    try:
        section = manifest["rd7_groot_172"]
        source = section["source"]
        dataset = section["dataset"]
        selected = dataset["selected"]
        queries = section["queries"]
        amendment_rd7 = amendment["rd7"]
        amendment_sidecar = amendment_rd7["candidate_sidecar"]
    except (KeyError, TypeError) as error:
        raise AuditFailure(f"frozen RD7 schema is incomplete: {error}") from error
    _require(len(selected) == 10, "frozen selected-video count is not ten")
    _require(
        [record["path"] for record in selected] == candidate_paths[:10],
        "selected videos are not the candidate sidecar's lexical first ten",
    )
    _require(dataset["candidate_count"] == 50656, "manifest candidate count drifted")
    _require(
        dataset["candidate_paths_sha256"] == uncompressed_sha256,
        "manifest candidate-path root drifted",
    )
    _require(
        amendment["canonical_manifest"]["sha256"] == manifest_sha256,
        "amendment canonical-manifest root drifted",
    )
    _require(
        amendment_sidecar["compressed_sha256"] == compressed_sha256
        and amendment_sidecar["uncompressed_sha256"] == uncompressed_sha256,
        "amendment candidate-sidecar roots drifted",
    )

    binding = {
        "case_id": CASE_ID,
        "manifest_sha256": manifest_sha256,
        "freeze_amendment_sha256": amendment_sha256,
        "candidate_sidecar_compressed_sha256": compressed_sha256,
        "candidate_sidecar_uncompressed_sha256": uncompressed_sha256,
        "source": source,
        "dataset_repo_id": dataset["repo_id"],
        "dataset_revision": dataset["revision"],
        "candidate_paths_sha256": dataset["candidate_paths_sha256"],
        "selected_sha256": _json_sha256(selected),
        "queries_sha256": _json_sha256(queries),
    }
    frozen = result.get("frozen_inputs")
    _require(isinstance(frozen, dict), "result lacks frozen_inputs")
    _require(frozen.get("manifest_sha256") == manifest_sha256, "result manifest root drifted")
    _require(
        frozen.get("freeze_amendment_sha256") == amendment_sha256,
        "result amendment root drifted",
    )
    _require(frozen.get("candidate_sidecar") == amendment_sidecar, "result sidecar binding drifted")
    _require(
        frozen.get("time_semantics") == amendment_rd7["time_semantics"],
        "result time semantics drifted",
    )
    _require(frozen.get("source") == source, "result source binding drifted")
    expected_dataset = {
        key: dataset[key]
        for key in (
            "repo_id",
            "revision",
            "eligibility",
            "candidate_count",
            "candidate_paths_sha256",
            "selection",
            "selected",
        )
    }
    _require(frozen.get("dataset") == expected_dataset, "result dataset binding drifted")
    _require(frozen.get("queries") == queries, "result query freeze drifted")
    _require(frozen.get("binding") == binding, "result frozen binding content drifted")
    binding_sha256 = _json_sha256(binding)
    _require(
        frozen.get("binding_sha256") == binding_sha256,
        "result frozen binding SHA-256 mismatch",
    )
    _require(
        frozen.get("no_replacement_after_observed_outcome") is True,
        "result does not preserve the no-replacement rule",
    )
    roots = {
        "manifest_sha256": manifest_sha256,
        "amendment_sha256": amendment_sha256,
        "sidecar_compressed_sha256": compressed_sha256,
        "sidecar_uncompressed_sha256": uncompressed_sha256,
        "binding_sha256": binding_sha256,
    }
    return manifest, selected, binding, roots


def _verify_formal_script_identity(
    result: dict[str, Any], formal_script: Path
) -> tuple[str, str]:
    _require(formal_script.is_file(), f"formal script is missing: {formal_script}")
    actual_sha256 = _file_sha256(formal_script)
    execution = result.get("execution")
    _require(isinstance(execution, dict), "result lacks execution identity")
    execution_binding = execution.get("execution_binding")
    _require(isinstance(execution_binding, dict), "result lacks execution binding")
    _require(
        execution.get("script_sha256") == actual_sha256,
        "result formal-script SHA-256 differs from the supplied script",
    )
    _require(
        execution_binding.get("script_sha256") == actual_sha256,
        "execution binding formal-script SHA-256 mismatch",
    )
    _require(
        execution_binding.get("replay_backend") == execution.get("replay_backend"),
        "execution replay backend is internally inconsistent",
    )
    _require(
        execution_binding.get("platform") == execution.get("platform"),
        "execution platform is internally inconsistent",
    )
    _require(
        execution_binding.get("packages") == execution.get("packages"),
        "execution package inventory is internally inconsistent",
    )
    python_version = execution_binding.get("python_version")
    _require(
        isinstance(python_version, str)
        and isinstance(execution.get("python"), str)
        and execution["python"].startswith(python_version),
        "execution Python identity is internally inconsistent",
    )
    _require(
        execution_binding.get("python_implementation") in {"CPython", "PyPy"},
        "execution Python implementation is invalid",
    )
    backend = execution["replay_backend"]
    _require(
        backend in {"pyav_historical_compat", "torchvision_videoreader"},
        "unexpected formal replay backend",
    )
    return actual_sha256, backend


def _checkpoint_rebase(
    remote_value: Any,
    run_dir: Path,
    expected_filename: str,
) -> tuple[PurePosixPath, Path]:
    _require(isinstance(remote_value, str), "checkpoint path is not a string")
    remote = PurePosixPath(remote_value)
    _require(remote.is_absolute(), "formal checkpoint path is not absolute")
    parts = remote.parts
    _require(parts.count("video-checkpoints") == 1, "checkpoint path has invalid layout")
    marker = parts.index("video-checkpoints")
    _require(
        parts[marker:] == ("video-checkpoints", expected_filename),
        "checkpoint relative path or filename drifted",
    )
    remote_root = PurePosixPath(*parts[:marker])
    local = run_dir / "video-checkpoints" / expected_filename
    return remote_root, local


def _verify_video_inputs_and_checkpoints(
    result: dict[str, Any],
    run_dir: Path,
    selected: Sequence[dict[str, Any]],
    binding_sha256: str,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    videos = result.get("videos")
    inventory = result.get("checkpoint_inventory")
    _require(isinstance(videos, list) and len(videos) == 10, "result must contain ten videos")
    _require(
        isinstance(inventory, list) and len(inventory) == 10,
        "result must contain ten checkpoint inventory entries",
    )
    remote_roots: set[str] = set()
    verified_inventory: list[dict[str, Any]] = []
    for index, (record, video, item) in enumerate(zip(selected, videos, inventory)):
        _require(video.get("video_index") == index, f"video {index} index drifted")
        _require(video.get("selected_path") == record["path"], f"video {index} membership drifted")
        _require(item.get("video_index") == index, f"checkpoint {index} index drifted")
        _require(item.get("selected_path") == record["path"], f"checkpoint {index} membership drifted")
        expected_input = {
            "path": record["path"],
            "expected_bytes": record["bytes"],
            "observed_bytes": record["bytes"],
            "git_blob_oid": record["git_blob_oid"],
            "expected_lfs_sha256": record["lfs_sha256"],
            "observed_sha256": record["lfs_sha256"],
            "exact_match": True,
        }
        _require(video.get("input") == expected_input, f"video {index} LFS identity drifted")
        expected_filename = (
            f"video-{index:02d}-"
            f"{hashlib.sha256(record['path'].encode('utf-8')).hexdigest()[:12]}.json"
        )
        remote_root, checkpoint_path = _checkpoint_rebase(
            item.get("checkpoint"), run_dir, expected_filename
        )
        remote_roots.add(remote_root.as_posix())
        _require(checkpoint_path.is_file(), f"rebased checkpoint is missing: {checkpoint_path}")
        checkpoint, checkpoint_raw = _load_json(checkpoint_path, f"checkpoint {index}")
        checkpoint_sha256 = _bytes_sha256(checkpoint_raw)
        _require(
            checkpoint_sha256 == item.get("checkpoint_sha256"),
            f"checkpoint {index} byte SHA-256 mismatch",
        )
        _require(
            checkpoint_raw == _canonical_json_bytes(checkpoint) + b"\n",
            f"checkpoint {index} is not canonical immutable JSON",
        )
        _require(checkpoint.get("schema_version") == 1, f"checkpoint {index} schema drifted")
        _require(
            checkpoint.get("binding_sha256") == binding_sha256,
            f"checkpoint {index} binding drifted",
        )
        _require(checkpoint.get("video_index") == index, f"checkpoint {index} identity drifted")
        _require(checkpoint.get("selected") == record, f"checkpoint {index} selection drifted")
        _require(checkpoint.get("result") == video, f"checkpoint {index} result content drifted")
        verified_inventory.append(
            {
                "video_index": index,
                "selected_path": record["path"],
                "remote_checkpoint": item["checkpoint"],
                "local_checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
            }
        )
    _require(len(remote_roots) == 1, "formal checkpoints do not share one remote run root")
    return videos, next(iter(remote_roots)), verified_inventory


def _oracle_frames(video: dict[str, Any]) -> list[tuple[Fraction, dict[str, Any]]]:
    oracle = video.get("oracle")
    _require(isinstance(oracle, dict), "video lacks an oracle")
    frames = oracle.get("frames")
    _require(isinstance(frames, list) and len(frames) >= 2, "oracle frame list is invalid")
    _require(oracle.get("frame_count") == len(frames), "oracle frame count drifted")
    _require(
        oracle.get("pts_rgb_oracle_sha256") == _json_sha256(frames),
        "PTS+RGB oracle root mismatch",
    )
    timed: list[tuple[Fraction, dict[str, Any]]] = []
    stream_time_base: Fraction | None = None
    for index, frame in enumerate(frames):
        _require(frame.get("source_index") == index, "oracle source indices are discontinuous")
        try:
            time_base = Fraction(frame["time_base"])
            pts = int(frame["pts"])
            seconds = float(frame["seconds"])
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as error:
            raise AuditFailure(f"malformed oracle frame {index}: {error}") from error
        _require(time_base > 0, f"oracle frame {index} time base is nonpositive")
        _require(seconds == float(pts * time_base), f"oracle frame {index} PTS seconds drifted")
        if stream_time_base is None:
            stream_time_base = time_base
        _require(time_base == stream_time_base, "oracle frame/stream time bases disagree")
        _require(_is_sha256(frame.get("rgb_sha256")), f"oracle frame {index} RGB digest invalid")
        _require(frame.get("dtype") == "uint8", f"oracle frame {index} dtype drifted")
        shape = frame.get("shape")
        _require(
            isinstance(shape, list) and len(shape) == 3 and shape[-1] == 3,
            f"oracle frame {index} shape drifted",
        )
        timed.append((pts * time_base, frame))
    _require(
        all(current[0] >= previous[0] for previous, current in zip(timed, timed[1:])),
        "oracle PTS is not nondecreasing",
    )
    origin = timed[0][0]
    duration = timed[-1][0] - origin
    _require(
        _parse_fraction_record(oracle.get("duration_origin"), "duration origin") == origin,
        "oracle duration origin drifted",
    )
    _require(
        _parse_fraction_record(oracle.get("decoded_duration"), "decoded duration") == duration,
        "oracle decoded duration drifted",
    )
    _require(oracle.get("pts_nondecreasing") is True, "oracle PTS flag drifted")
    _require(
        oracle.get("all_frame_time_bases_equal_stream") is True,
        "oracle time-base equivalence flag drifted",
    )
    _require(
        oracle.get("pts_seconds_rule") == "float(frame.pts * stream.time_base)",
        "oracle PTS-seconds rule drifted",
    )
    duplicate_count = len(timed) - len({value[0] for value in timed})
    _require(oracle.get("duplicate_pts_count") == duplicate_count, "duplicate-PTS count drifted")
    return timed


def _local_interval(times: Sequence[Fraction], selected: Fraction) -> Fraction | None:
    unique = sorted(set(times))
    if len(unique) < 2:
        return None
    position = unique.index(selected)
    candidates: list[Fraction] = []
    if position > 0:
        candidates.append(selected - unique[position - 1])
    if position + 1 < len(unique):
        candidates.append(unique[position + 1] - selected)
    positive = [value for value in candidates if value > 0]
    return min(positive) if positive else None


def _verify_query_row(
    video: dict[str, Any],
    timed_frames: Sequence[tuple[Fraction, dict[str, Any]]],
    group_name: str,
    query_index: int,
    target: Fraction,
    row: dict[str, Any],
) -> None:
    _require(row.get("group") == group_name, "query group identity drifted")
    _require(row.get("query_index") == query_index, "query index drifted")
    _require(
        _parse_fraction_record(row.get("target"), "query target") == target,
        "query target drifted",
    )
    eligible = [time for time, _ in timed_frames if time <= target]
    selected_time = max(eligible) if eligible else timed_frames[0][0]
    equivalent = [frame for time, frame in timed_frames if time == selected_time]
    times = [time for time, _ in timed_frames]
    interval = _local_interval(times, selected_time)
    expected_oracle = {
        "selected_pts": _fraction_record(selected_time),
        "equivalent_source_indices": [frame["source_index"] for frame in equivalent],
        "equivalent_rgb_sha256": [frame["rgb_sha256"] for frame in equivalent],
        "local_frame_interval_seconds": float(interval) if interval is not None else None,
    }
    _require(row.get("oracle") == expected_oracle, "per-query oracle selection drifted")

    actual = row.get("actual")
    if actual is None:
        timestamp_error = None
        pts_match = False
        digest_match = False
        shape_match = False
        dtype_match = False
        exceeds = True
    else:
        _require(isinstance(actual, dict), "query actual frame is malformed")
        _require(isinstance(actual.get("loaded_index"), int), "loaded index is invalid")
        actual_pts = float(actual.get("pts_seconds"))
        _require(math.isfinite(actual_pts), "actual frame PTS is non-finite")
        _require(_is_sha256(actual.get("rgb_sha256")), "actual RGB digest is invalid")
        timestamp_error = abs(actual_pts - float(selected_time))
        ticks = [Fraction(frame["time_base"]) for frame in equivalent]
        tolerance = max(1e-9, float(min(ticks)) * 1e-6)
        pts_match = timestamp_error <= tolerance
        digest_match = actual["rgb_sha256"] in {
            frame["rgb_sha256"] for frame in equivalent
        }
        oracle_shapes = {tuple(frame["shape"]) for _, frame in timed_frames}
        shape_match = tuple(actual.get("shape", ())) in oracle_shapes
        dtype_match = actual.get("dtype") == "uint8"
        exceeds = interval is not None and timestamp_error > float(interval) + tolerance
    expected_fields = {
        "pts_match": pts_match,
        "rgb_digest_match": digest_match,
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "correct": pts_match and digest_match,
        "wrong_frame": not (pts_match and digest_match),
        "absolute_timestamp_error_seconds": timestamp_error,
        "exceeds_one_local_frame_interval": exceeds,
    }
    for key, expected in expected_fields.items():
        _require(row.get(key) == expected, f"per-query {key} was not recomputed correctly")


def _expected_query_groups(
    video: dict[str, Any],
    timed_frames: Sequence[tuple[Fraction, dict[str, Any]]],
    manifest_queries: dict[str, Any],
) -> dict[str, tuple[Fraction, ...]]:
    origin = timed_frames[0][0]
    duration = timed_frames[-1][0] - origin
    return {
        "regular": tuple(
            Fraction(str(value))
            for value in manifest_queries["reported_regular_seconds"]
        ),
        "sparse": tuple(
            origin + duration * Fraction(str(value))
            for value in manifest_queries["sparse_duration_fractions"]
        ),
        "late": tuple(
            origin + duration * Fraction(str(value))
            for value in manifest_queries["late_duration_fractions"]
        ),
        "continuous_pts_negative_control": tuple(
            value for value, _ in timed_frames
        ),
    }


def _verify_video_metrics(
    video: dict[str, Any], manifest_queries: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    timed_frames = _oracle_frames(video)
    expected_groups = _expected_query_groups(video, timed_frames, manifest_queries)
    recorded_groups = video.get("query_groups")
    _require(isinstance(recorded_groups, dict), "video query_groups is invalid")
    _require(set(recorded_groups) == set(GROUPS), "video query-group membership drifted")
    for group_name, targets in expected_groups.items():
        parsed = tuple(
            _parse_fraction_record(value, f"{group_name} target")
            for value in recorded_groups[group_name]
        )
        _require(parsed == targets, f"{group_name} query schedule drifted")

    branches = video.get("branches")
    _require(isinstance(branches, dict) and set(branches) == {"old", "fixed"}, "branch set drifted")
    verified: dict[str, dict[str, list[dict[str, Any]]]] = {}
    expected_branch_summaries: dict[str, Any] = {}
    for branch_name in ("old", "fixed"):
        branch = branches[branch_name]
        _require(set(branch) == set(GROUPS), f"{branch_name} group set drifted")
        verified[branch_name] = {}
        all_rows: list[dict[str, Any]] = []
        outside_rows: list[dict[str, Any]] = []
        all_structural = True
        for group_name in GROUPS:
            group = branch[group_name]
            rows = group.get("queries")
            targets = expected_groups[group_name]
            _require(isinstance(rows, list) and len(rows) == len(targets), "query row count drifted")
            for index, (target, row) in enumerate(zip(targets, rows)):
                _verify_query_row(video, timed_frames, group_name, index, target, row)
            recomputed_metrics = _metric_summary(rows)
            _require(group.get("metrics") == recomputed_metrics, "group metrics do not match rows")
            structure = group.get("structure_baseline")
            _require(isinstance(structure, dict), "structure baseline is missing")
            expected_count = len(targets)
            _require(
                structure
                == {
                    "readable": True,
                    "return_count": expected_count,
                    "expected_return_count": expected_count,
                    "return_count_matches": True,
                    "extra_return_count": 0,
                    "all_shapes_match_oracle": True,
                    "all_dtypes_uint8": True,
                    "accepted": True,
                },
                "formal structure baseline drifted",
            )
            _require(
                isinstance(group.get("loaded_count"), int)
                and group["loaded_count"] >= expected_count,
                "historical replay loaded_count is invalid",
            )
            verified[branch_name][group_name] = rows
            all_rows.extend(rows)
            if group_name in OUTSIDE_GROUPS:
                outside_rows.extend(rows)
            all_structural = all_structural and structure["accepted"]
        negative_rows = verified[branch_name]["continuous_pts_negative_control"]
        expected_branch_summaries[branch_name] = {
            "all_groups_structure_baseline_accepted": all_structural,
            "all_queries": _metric_summary(all_rows),
            "outside_negative_control": _metric_summary(outside_rows),
            "negative_control": _metric_summary(negative_rows),
            "temporal_contract_accepted": all_structural
            and all(row["correct"] for row in all_rows),
        }
    _require(
        video.get("branch_summaries") == expected_branch_summaries,
        "video branch summaries do not match query rows",
    )
    old_wrong = (
        expected_branch_summaries["old"]["outside_negative_control"][
            "wrong_frame_count"
        ]
        > 0
    )
    old_negative_correct = (
        expected_branch_summaries["old"]["negative_control"]["wrong_frame_count"]
        == 0
    )
    fixed_correct = (
        expected_branch_summaries["fixed"]["all_queries"]["wrong_frame_count"]
        == 0
    )
    both_structural = all(
        expected_branch_summaries[name]["all_groups_structure_baseline_accepted"]
        for name in ("old", "fixed")
    )
    expected_correctness = {
        "old_wrong_outside_negative_control": old_wrong,
        "old_negative_control_all_correct": old_negative_correct,
        "fixed_all_queries_correct": fixed_correct,
        "both_branches_structurally_readable": both_structural,
        "temporal_contract_rejects_old": not expected_branch_summaries["old"][
            "temporal_contract_accepted"
        ],
        "temporal_contract_accepts_fixed": expected_branch_summaries["fixed"][
            "temporal_contract_accepted"
        ],
    }
    expected_correctness["reproduced_on_video"] = all(
        [
            old_wrong,
            old_negative_correct,
            fixed_correct,
            both_structural,
            expected_correctness["temporal_contract_rejects_old"],
            expected_correctness["temporal_contract_accepts_fixed"],
        ]
    )
    _require(video.get("correctness") == expected_correctness, "video correctness drifted")
    return verified


def _verify_all_metrics(
    result: dict[str, Any],
    videos: Sequence[dict[str, Any]],
    manifest_queries: dict[str, Any],
) -> dict[str, Any]:
    collected: dict[str, dict[str, list[dict[str, Any]]]] = {
        branch: {group: [] for group in GROUPS} for branch in ("old", "fixed")
    }
    for video in videos:
        verified = _verify_video_metrics(video, manifest_queries)
        for branch in ("old", "fixed"):
            for group in GROUPS:
                collected[branch][group].extend(verified[branch][group])

    expected_aggregate: dict[str, Any] = {}
    for branch in ("old", "fixed"):
        all_rows = [
            row for group in GROUPS for row in collected[branch][group]
        ]
        expected_aggregate[branch] = {
            "all_queries": _metric_summary(all_rows),
            "by_group": {
                group: _metric_summary(collected[branch][group])
                for group in sorted(GROUPS)
            },
            "all_videos_structure_baseline_accepted": True,
        }
    _require(result.get("aggregate") == expected_aggregate, "result aggregate does not match rows")

    expected_counts = {
        "regular": 100,
        "sparse": 50,
        "late": 50,
        "continuous_pts_negative_control": 2038,
    }
    for branch in ("old", "fixed"):
        for group, count in expected_counts.items():
            _require(
                len(collected[branch][group]) == count,
                f"{branch}/{group} query count is not {count}",
            )
    for group in OUTSIDE_GROUPS:
        _require(
            _metric_summary(collected["old"][group])["wrong_frame_count"]
            == expected_counts[group],
            f"old/{group} does not have every query wrong",
        )
    _require(
        _metric_summary(collected["old"]["continuous_pts_negative_control"])[
            "wrong_frame_count"
        ]
        == 0,
        "old continuous-PTS negative control is not 2,038/2,038 correct",
    )
    _require(
        expected_aggregate["fixed"]["all_queries"]["wrong_frame_count"] == 0,
        "fixed branch is not correct on every query",
    )
    expected_correctness = {
        "fixed_input_count": 10,
        "completed_input_count": 10,
        "all_input_sizes_and_lfs_hashes_exact": True,
        "videos_with_old_error_outside_negative_control": 10,
        "all_old_continuous_pts_negative_controls_correct": True,
        "all_fixed_queries_correct": True,
        "both_branches_structurally_readable_on_all_videos": True,
        "temporal_contract_rejects_old": True,
        "temporal_contract_accepts_fixed": True,
        "reproduced": True,
    }
    _require(result.get("correctness") == expected_correctness, "top-level correctness drifted")
    _require(result.get("status") == "REPRODUCED", "formal status is not REPRODUCED")
    return {
        "video_count": 10,
        "queries_per_branch": sum(expected_counts.values()),
        "old_wrong_by_group": {
            group: _metric_summary(collected["old"][group])["wrong_frame_count"]
            for group in GROUPS
        },
        "fixed_wrong_by_group": {
            group: _metric_summary(collected["fixed"][group])["wrong_frame_count"]
            for group in GROUPS
        },
        "old_negative_control_correct": 2038,
        "fixed_total_correct": 2238,
    }


def _verify_event_ledger(
    events_path: Path,
    binding_sha256: str,
    result: dict[str, Any],
    result_path: Path,
    inventory: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    try:
        raw = events_path.read_bytes()
    except OSError as error:
        raise AuditFailure(f"cannot read event ledger {events_path}: {error}") from error
    lines = raw.splitlines(keepends=True)
    _require(len(lines) == 32, f"event ledger has {len(lines)} lines, expected 32")
    events: list[dict[str, Any]] = []
    previous_hash = ZERO_HASH
    for index, line in enumerate(lines, start=1):
        _require(line.endswith(b"\n"), f"event line {index} is torn")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditFailure(f"invalid event line {index}: {error}") from error
        _require(line == _canonical_json_bytes(event) + b"\n", f"event line {index} is noncanonical")
        _require(event.get("schema_version") == 1, f"event line {index} schema drifted")
        _require(event.get("sequence") == index, f"event line {index} sequence drifted")
        _require(
            event.get("binding_sha256") == binding_sha256,
            f"event line {index} binding drifted",
        )
        _require(
            event.get("previous_event_sha256") == previous_hash,
            f"event line {index} predecessor drifted",
        )
        unsigned = dict(event)
        claimed_hash = unsigned.pop("event_sha256", None)
        _require(claimed_hash == _json_sha256(unsigned), f"event line {index} hash mismatch")
        previous_hash = claimed_hash
        events.append(event)

    expected_types = ["RUN_START"]
    for _ in range(10):
        expected_types.extend(
            ["VIDEO_INPUT_BEGIN", "VIDEO_INPUT_VERIFIED", "VIDEO_COMPLETE"]
        )
    expected_types.append("RUN_COMPLETE")
    _require(
        [event.get("event") for event in events] == expected_types,
        "event lifecycle or order drifted",
    )
    start_payload = events[0].get("payload", {})
    frozen = result["frozen_inputs"]
    execution = result["execution"]
    _require(
        start_payload.get("manifest_sha256") == frozen["manifest_sha256"]
        and start_payload.get("freeze_amendment_sha256")
        == frozen["freeze_amendment_sha256"],
        "RUN_START frozen roots drifted",
    )
    _require(start_payload.get("selected_count") == 10, "RUN_START selected count drifted")
    _require(
        start_payload.get("download_endpoint") == "https://huggingface.co"
        and start_payload.get("download_authentication") == "disabled",
        "RUN_START download boundary drifted",
    )
    _require(
        start_payload.get("replay_backend") == execution["replay_backend"],
        "RUN_START replay backend drifted",
    )
    _require(isinstance(start_payload.get("download_authorized"), bool), "RUN_START download flag invalid")

    for index, (video, item) in enumerate(zip(result["videos"], inventory)):
        begin = events[1 + 3 * index]["payload"]
        verified = events[2 + 3 * index]["payload"]
        complete = events[3 + 3 * index]["payload"]
        expected_identity = {
            "video_index": index,
            "selected_path": video["selected_path"],
        }
        _require(begin == expected_identity, f"video {index} begin event drifted")
        _require(
            verified.get("video_index") == index
            and verified.get("selected_path") == video["selected_path"]
            and verified.get("observed_bytes") == video["input"]["observed_bytes"]
            and verified.get("observed_sha256") == video["input"]["observed_sha256"]
            and isinstance(verified.get("downloaded_this_invocation"), bool),
            f"video {index} verified-input event drifted",
        )
        _require(
            complete
            == {
                "video_index": index,
                "selected_path": video["selected_path"],
                "checkpoint": item["remote_checkpoint"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "oracle_sha256": video["oracle"]["pts_rgb_oracle_sha256"],
                "reproduced_on_video": video["correctness"]["reproduced_on_video"],
            },
            f"video {index} completion event drifted",
        )

    final = events[-1]
    _require(final.get("event") == "RUN_COMPLETE", "final event is not RUN_COMPLETE")
    final_payload = final.get("payload", {})
    result_sha256 = _file_sha256(result_path)
    _require(
        final_payload.get("result_sha256") == result_sha256,
        "RUN_COMPLETE result SHA-256 mismatch",
    )
    _require(final_payload.get("status") == result.get("status"), "RUN_COMPLETE status drifted")
    remote_result = final_payload.get("result_path")
    _require(
        isinstance(remote_result, str)
        and PurePosixPath(remote_result).is_absolute()
        and PurePosixPath(remote_result).name == result_path.name,
        "RUN_COMPLETE remote result path drifted",
    )
    return {
        "event_count": len(events),
        "event_log_sha256": _bytes_sha256(raw),
        "event_tail_sha256": previous_hash,
        "first_event": events[0]["event"],
        "final_event": final["event"],
        "remote_result_path": remote_result,
    }


def perform_audit(
    *,
    result_path: Path,
    run_dir: Path,
    formal_script: Path = DEFAULT_FORMAL_SCRIPT,
    manifest_path: Path = DEFAULT_MANIFEST,
    amendment_path: Path = DEFAULT_AMENDMENT,
    sidecar_path: Path = DEFAULT_SIDECAR,
    created_utc: str | None = None,
) -> dict[str, Any]:
    result, result_raw = _load_json(result_path, "formal result")
    _require(result.get("schema_version") == 1, "formal result schema drifted")
    _require(result.get("case_id") == CASE_ID, "formal case identity drifted")
    _require(result.get("evidence_class") == "E2", "formal evidence class drifted")
    manifest, selected, binding, roots = _verify_frozen_inputs(
        result, manifest_path, amendment_path, sidecar_path
    )
    formal_script_sha256, replay_backend = _verify_formal_script_identity(
        result, formal_script
    )
    videos, remote_checkpoint_root, checkpoint_inventory = (
        _verify_video_inputs_and_checkpoints(
            result,
            run_dir,
            selected,
            roots["binding_sha256"],
        )
    )
    _require(
        all(video.get("replay_backend") == replay_backend for video in videos),
        "video checkpoint replay backends differ from the execution binding",
    )
    metrics = _verify_all_metrics(
        result, videos, manifest["rd7_groot_172"]["queries"]
    )
    events_path = run_dir / "events.jsonl"
    ledger = _verify_event_ledger(
        events_path,
        roots["binding_sha256"],
        result,
        result_path,
        checkpoint_inventory,
    )
    auditor_sha256 = _file_sha256(Path(__file__).resolve())
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "created_utc": created_utc or _utc_now(),
        "status": "AUDIT_PASS",
        "scope": (
            "compact post-run integrity and deterministic metric audit; "
            "no video decode, no download, no mutation of formal evidence"
        ),
        "inputs": {
            "result": str(result_path),
            "result_sha256": _bytes_sha256(result_raw),
            "run_dir": str(run_dir),
            "event_log": str(events_path),
            "event_log_sha256": ledger["event_log_sha256"],
            "formal_script": str(formal_script),
            "formal_script_sha256": formal_script_sha256,
            "manifest": str(manifest_path),
            "amendment": str(amendment_path),
            "candidate_sidecar": str(sidecar_path),
            "remote_checkpoint_root": remote_checkpoint_root,
        },
        "verified": {
            "frozen_roots_and_binding": True,
            "fixed_ten_video_membership": True,
            "captured_input_sizes_and_lfs_hashes": True,
            "ten_checkpoint_byte_hashes_and_content_identities": True,
            "event_ledger_32_line_hash_chain": True,
            "final_event_run_complete": True,
            "per_query_oracle_fields_and_metrics_recomputed": True,
            "aggregate_metrics_recomputed": True,
            "formal_script_and_execution_binding_consistent": True,
            "fixed_all_queries_correct": True,
            "old_three_outside_groups_all_wrong": True,
            "continuous_pts_negative_control_2038_all_correct": True,
        },
        "frozen_roots": roots,
        "event_ledger": ledger,
        "checkpoint_inventory": checkpoint_inventory,
        "recomputed": metrics,
        "auditor": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": auditor_sha256,
            "python": sys.version,
        },
        "limitations": [
            "The audit verifies the captured input LFS hashes against the pre-outcome manifest; it deliberately does not reopen or rehash the large video files.",
            "Remote absolute checkpoint paths are validated for one common formal root and then rebased through their video-checkpoints-relative suffix onto --run-dir.",
            "This is a compact-evidence audit, not a second decode or an independent execution of the historical GR00T stack.",
        ],
    }


def _write_create_once(path: Path, payload: dict[str, Any]) -> str:
    serialized = _canonical_json_bytes(payload) + b"\n"
    digest = _bytes_sha256(serialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        _require(existing == serialized, f"create-once audit output already differs: {path}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with open(descriptor, "wb", closefd=True) as handle:
            handle.write(serialized)
            handle.flush()
            import os

            os.fsync(handle.fileno())
        try:
            import os

            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == serialized, "create-once audit race differs")
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _existing_created_utc(path: Path) -> str | None:
    if not path.exists():
        return None
    existing, _ = _load_json(path, "existing audit output")
    _require(existing.get("audit_id") == AUDIT_ID, "existing audit output identity drifted")
    created = existing.get("created_utc")
    _require(isinstance(created, str), "existing audit output creation time is invalid")
    return created


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal-script", type=Path, default=DEFAULT_FORMAL_SCRIPT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--candidate-sidecar", type=Path, default=DEFAULT_SIDECAR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        created_utc = _existing_created_utc(args.output)
        payload = perform_audit(
            result_path=args.result,
            run_dir=args.run_dir,
            formal_script=args.formal_script,
            manifest_path=args.manifest,
            amendment_path=args.amendment,
            sidecar_path=args.candidate_sidecar,
            created_utc=created_utc,
        )
        output_sha256 = _write_create_once(args.output, payload)
    except Exception as error:
        print(f"RD7 post-run audit failed closed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "result_sha256": payload["inputs"]["result_sha256"],
                "event_count": payload["event_ledger"]["event_count"],
                "checkpoint_count": len(payload["checkpoint_inventory"]),
                "output": str(args.output),
                "output_sha256": output_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
