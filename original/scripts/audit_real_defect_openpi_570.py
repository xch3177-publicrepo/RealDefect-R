#!/usr/bin/env python3
"""Offline post-run auditor for the frozen P13 RD8 OpenPI #570 replay.

The auditor never contacts Hugging Face and never mutates ``--run-dir``.  It
re-hashes every compact projection/checkpoint, rebuilds the logical roots, and
recomputes the non-histogram scientific claims directly from the captured
float32 arrays.  Absolute paths recorded on the formal host are accepted only
after their suffix is validated and then rebased beneath ``--run-dir``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "results" / "real-defect-openpi-570.json"
DEFAULT_OUTPUT = ROOT / "results" / "real-defect-openpi-570-audit.json"
DEFAULT_MANIFEST = ROOT / "contracts" / "real-defect-corpus-manifest.json"
DEFAULT_AMENDMENT = ROOT / "contracts" / "real-defect-corpus-freeze-amendment-v1.json"
DEFAULT_FORMAL_SCRIPT = ROOT / "scripts" / "run_real_defect_openpi_570.py"

AUDIT_ID = "P13-RD8-OPENPI-570-POSTRUN-AUDIT-V1"
CASE_ID = "RD8-OPENPI-570-619-623"
EXPECTED_MANIFEST_SHA256 = "7255524dc1776ef3e96e12fceea138ac76ae3bb15c55df42dc7e5a9e402e0d17"
EXPECTED_AMENDMENT_SHA256 = "58ad7063ddd44872d8fa1535e585f39079537d0a0715967f4784adf6d2adbd0c"
EXPECTED_ORDERS_SHA256 = "957d557de188190b591f98c28ceb4bc281b60be12765c31514cc3c1340fc5fd1"
EXPECTED_REPO_ID = "physical-intelligence/aloha_pen_uncap_diverse"
EXPECTED_REVISION = "e82d8b40b8ac66c0b40273dd80a077dfc40b732e"
EXPECTED_EPISODES = 123
EXPECTED_ROWS_PER_EPISODE = 300
EXPECTED_FRAMES = 36_900
EXPECTED_DIM = 14
ORDER_NAMES = ("natural", *(f"episode_cluster_seed_{seed}" for seed in range(10)))
BATCH_SIZES = (1, 8, 32, 128)
PRIMARY_BATCH_SIZE = 32
PROJECTION_COLUMNS = (
    "observation.state",
    "action",
    "episode_index",
    "frame_index",
    "timestamp",
)
ZERO_HASH = "0" * 64
STAT_ABSOLUTE_TOLERANCE = 1e-5
EFFECT_ABSOLUTE_TOLERANCE = 2e-5
TIGHT_REPLAY_TOLERANCE = 1e-12


class AuditFailure(RuntimeError):
    """A formal evidence invariant failed closed."""


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _bytes_sha256(_canonical_json_bytes(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditFailure(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value, raw


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class FrozenInputs:
    manifest_sha256: str
    amendment_sha256: str
    section: dict[str, Any]
    files: tuple[dict[str, Any], ...]
    orders: dict[str, tuple[int, ...]]
    expected_counts: dict[str, dict[str, int]]
    feature_names: tuple[str, ...]
    binding: dict[str, Any]
    binding_sha256: str


def _load_frozen_inputs(manifest_path: Path, amendment_path: Path) -> FrozenInputs:
    manifest, manifest_raw = _load_json(manifest_path, "canonical manifest")
    amendment, amendment_raw = _load_json(amendment_path, "freeze amendment")
    manifest_sha = _bytes_sha256(manifest_raw)
    amendment_sha = _bytes_sha256(amendment_raw)
    _require(manifest_sha == EXPECTED_MANIFEST_SHA256, "canonical manifest SHA-256 mismatch")
    _require(amendment_sha == EXPECTED_AMENDMENT_SHA256, "freeze amendment SHA-256 mismatch")
    _require(manifest.get("corpus_id") == "P13-REALDEFECT-8", "corpus identity drifted")
    _require(
        amendment.get("amendment_id") == "P13-REALDEFECT-8-FREEZE-AMENDMENT-V1",
        "freeze amendment identity drifted",
    )
    _require(
        amendment.get("canonical_manifest", {}).get("sha256") == manifest_sha,
        "amendment does not bind the canonical manifest",
    )
    try:
        section = manifest["rd8_openpi_570"]
        dataset = section["dataset"]
        source = section["source"]
        design = section["design"]
        rd8 = amendment["rd8"]
        ordering = rd8["ordering"]
        features = rd8["features"]
        expected_counts = rd8["expected_replay_counts"]
    except (KeyError, TypeError) as error:
        raise AuditFailure(f"frozen RD8 schema is incomplete: {error}") from error

    _require(dataset.get("repo_id") == EXPECTED_REPO_ID, "dataset repo drifted")
    _require(dataset.get("revision") == EXPECTED_REVISION, "dataset revision drifted")
    _require(dataset.get("expected_episodes") == EXPECTED_EPISODES, "episode count drifted")
    _require(dataset.get("expected_frames") == EXPECTED_FRAMES, "frame count drifted")
    _require(dataset.get("projection_columns") == list(PROJECTION_COLUMNS), "projection columns drifted")
    files = dataset.get("files")
    _require(isinstance(files, list) and len(files) == EXPECTED_EPISODES, "file inventory is not 123 entries")
    checked_files: list[dict[str, Any]] = []
    for episode, record in enumerate(files):
        expected_path = f"data/chunk-000/episode_{episode:06d}.parquet"
        _require(isinstance(record, dict) and record.get("path") == expected_path, f"source path drifted at {episode}")
        _require(isinstance(record.get("bytes"), int) and record["bytes"] > 0, f"source bytes invalid at {episode}")
        _require(_is_hex(record.get("git_blob_oid"), 40), f"Git blob invalid at {episode}")
        _require(_is_hex(record.get("lfs_sha256"), 64), f"LFS hash invalid at {episode}")
        checked_files.append(dict(record))
    paths_bytes = "".join(f"{record['path']}\n" for record in checked_files).encode()
    _require(_bytes_sha256(paths_bytes) == dataset.get("file_paths_sha256"), "source path root drifted")

    _require(design.get("orders") == list(ORDER_NAMES), "order labels drifted")
    _require(design.get("batch_sizes") == list(BATCH_SIZES), "batch sizes drifted")
    _require(design.get("primary_batch_size") == PRIMARY_BATCH_SIZE, "primary batch drifted")
    orders_raw = ordering.get("orders")
    _require(isinstance(orders_raw, dict), "frozen orders are absent")
    _require(
        _bytes_sha256(_canonical_json_bytes(orders_raw) + b"\n") == EXPECTED_ORDERS_SHA256,
        "complete frozen order root drifted",
    )
    _require(ordering.get("orders_canonical_sha256") == EXPECTED_ORDERS_SHA256, "declared order root drifted")
    orders: dict[str, tuple[int, ...]] = {}
    for name in ORDER_NAMES:
        values = orders_raw.get(name)
        _require(isinstance(values, list) and sorted(values) == list(range(EXPECTED_EPISODES)), f"invalid order {name}")
        orders[name] = tuple(int(value) for value in values)

    state_feature = features.get("state", {})
    action_feature = features.get("actions", {})
    _require(state_feature.get("source_key") == "observation.state", "state source key drifted")
    _require(action_feature.get("source_key") == "action", "action source key drifted")
    _require(state_feature.get("dtype") == action_feature.get("dtype") == "float32", "feature dtype drifted")
    _require(state_feature.get("shape") == action_feature.get("shape") == [EXPECTED_DIM], "feature shape drifted")
    _require(state_feature.get("names") == action_feature.get("names"), "feature names differ")
    feature_names = tuple(state_feature.get("names", ()))
    _require(len(feature_names) == EXPECTED_DIM and len(set(feature_names)) == EXPECTED_DIM, "feature names invalid")
    for batch_size in BATCH_SIZES:
        expected = {
            "batch_count_including_partial_tail": math.ceil(EXPECTED_FRAMES / batch_size),
            "old_processed_vectors": math.ceil(EXPECTED_FRAMES / batch_size),
            "fixed_processed_vectors": EXPECTED_FRAMES,
        }
        _require(expected_counts.get(str(batch_size)) == expected, f"frozen count drifted for batch {batch_size}")

    binding = {
        "case_id": CASE_ID,
        "manifest_sha256": manifest_sha,
        "freeze_amendment_sha256": amendment_sha,
        "source": source,
        "dataset_repo_id": dataset["repo_id"],
        "dataset_revision": dataset["revision"],
        "file_paths_sha256": dataset["file_paths_sha256"],
        "file_inventory_sha256": _json_sha256(checked_files),
        "orders_sha256": EXPECTED_ORDERS_SHA256,
        "feature_names": list(feature_names),
        "batch_sizes": list(BATCH_SIZES),
    }
    return FrozenInputs(
        manifest_sha256=manifest_sha,
        amendment_sha256=amendment_sha,
        section=section,
        files=tuple(checked_files),
        orders=orders,
        expected_counts={str(key): dict(value) for key, value in expected_counts.items()},
        feature_names=feature_names,
        binding=binding,
        binding_sha256=_json_sha256(binding),
    )


def _verify_result_and_runner(
    result_path: Path,
    formal_script: Path,
    frozen: FrozenInputs,
) -> tuple[dict[str, Any], bytes, str]:
    result, raw = _load_json(result_path, "formal result")
    _require(raw == _canonical_json_bytes(result) + b"\n", "formal result is not canonical JSON")
    _require(result.get("schema_version") == 1, "result schema drifted")
    _require(result.get("case_id") == CASE_ID, "result case identity drifted")
    _require(result.get("evidence_class") == "E2", "result evidence class drifted")
    formal_sha = _file_sha256(formal_script)
    execution = result.get("execution")
    _require(isinstance(execution, dict), "result execution identity is absent")
    _require(execution.get("script_sha256") == formal_sha, "formal runner SHA differs from result")
    _require(execution.get("execution_binding", {}).get("script_sha256") == formal_sha, "execution binding runner SHA differs")
    frozen_result = result.get("frozen_inputs")
    _require(isinstance(frozen_result, dict), "result frozen_inputs absent")
    _require(frozen_result.get("manifest_sha256") == frozen.manifest_sha256, "result manifest root drifted")
    _require(frozen_result.get("freeze_amendment_sha256") == frozen.amendment_sha256, "result amendment root drifted")
    _require(frozen_result.get("binding") == frozen.binding, "result binding content drifted")
    _require(frozen_result.get("binding_sha256") == frozen.binding_sha256, "result binding digest drifted")
    _require(frozen_result.get("dataset") == frozen.section["dataset"], "result dataset freeze drifted")
    _require(frozen_result.get("design") == frozen.section["design"], "result design freeze drifted")
    expected_orders = {name: list(frozen.orders[name]) for name in ORDER_NAMES}
    _require(frozen_result.get("complete_orders") == expected_orders, "result orders drifted")
    _require(frozen_result.get("no_replacement_after_observed_outcome") is True, "result permits replacement")
    return result, raw, formal_sha


def _data_digest(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in zip(
        ("state", "actions", "episode_index", "frame_index", "timestamp"),
        arrays,
        strict=True,
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode() + b"\0")
        digest.update(str(contiguous.dtype).encode() + b"\0")
        digest.update(_canonical_json_bytes(list(contiguous.shape)))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _array_logical_descriptor(name: str, value: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(value)
    return {
        "name": name,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "logical_sha256": _bytes_sha256(contiguous.tobytes(order="C")),
    }


def _logical_projection_record(
    episode: int, source: dict[str, Any], arrays: Sequence[np.ndarray]
) -> dict[str, Any]:
    return {
        "episode": episode,
        "source_path": source["path"],
        "source_lfs_sha256": source["lfs_sha256"],
        "arrays": [
            _array_logical_descriptor(name, value)
            for name, value in zip(
                ("state", "actions", "episode_index", "frame_index", "timestamp"),
                arrays,
                strict=True,
            )
        ],
    }


def _load_projection(path: Path, episode: int) -> tuple[np.ndarray, ...]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            _require(
                set(archive.files) == {"state", "actions", "episode_index", "frame_index", "timestamp"},
                f"projection fields drifted at episode {episode}",
            )
            arrays = tuple(
                np.asarray(archive[name])
                for name in ("state", "actions", "episode_index", "frame_index", "timestamp")
            )
    except (OSError, ValueError) as error:
        raise AuditFailure(f"cannot read projection episode {episode}: {error}") from error
    state, actions, episode_index, frame_index, timestamp = arrays
    _require(state.dtype == np.float32 and state.shape == (EXPECTED_ROWS_PER_EPISODE, EXPECTED_DIM), f"state shape/dtype drifted at {episode}")
    _require(actions.dtype == np.float32 and actions.shape == state.shape, f"actions shape/dtype drifted at {episode}")
    _require(episode_index.dtype == np.int64 and episode_index.shape == (EXPECTED_ROWS_PER_EPISODE,), f"episode_index drifted at {episode}")
    _require(frame_index.dtype == np.int64 and frame_index.shape == (EXPECTED_ROWS_PER_EPISODE,), f"frame_index drifted at {episode}")
    _require(timestamp.dtype == np.float32 and timestamp.shape == (EXPECTED_ROWS_PER_EPISODE,), f"timestamp drifted at {episode}")
    _require(np.all(episode_index == episode), f"foreign episode_index at {episode}")
    _require(np.array_equal(frame_index, np.arange(EXPECTED_ROWS_PER_EPISODE, dtype=np.int64)), f"frame order drifted at {episode}")
    _require(np.all(np.isfinite(state)) and np.all(np.isfinite(actions)), f"non-finite feature at {episode}")
    _require(np.all(np.isfinite(timestamp)) and np.all(np.diff(timestamp.astype(np.float64)) >= 0), f"timestamp invalid at {episode}")
    return arrays


def _rebase_absolute(
    remote_value: Any,
    run_dir: Path,
    directory: str,
    filename: str,
) -> tuple[str, Path]:
    _require(isinstance(remote_value, str), "formal artifact path is not a string")
    remote = PurePosixPath(remote_value)
    _require(remote.is_absolute() and ".." not in remote.parts, "formal artifact path is unsafe")
    _require(remote.parts.count(directory) == 1, f"formal path has invalid {directory} layout")
    marker = remote.parts.index(directory)
    _require(remote.parts[marker:] == (directory, filename), f"formal {directory} suffix drifted")
    root = PurePosixPath(*remote.parts[:marker]).as_posix()
    return root, run_dir / directory / filename


def _verify_projection_artifacts(
    result: dict[str, Any],
    run_dir: Path,
    frozen: FrozenInputs,
) -> tuple[list[tuple[np.ndarray, ...]], dict[str, Any]]:
    projection = result.get("projection")
    _require(isinstance(projection, dict), "result projection section absent")
    inventory = projection.get("inventory")
    logical_result = projection.get("logical_inventory")
    _require(isinstance(inventory, list) and len(inventory) == EXPECTED_EPISODES, "projection inventory is not 123 entries")
    _require(isinstance(logical_result, list) and len(logical_result) == EXPECTED_EPISODES, "logical projection inventory is not 123 entries")
    arrays_by_episode: list[tuple[np.ndarray, ...]] = []
    logical_inventory: list[dict[str, Any]] = []
    checkpoint_inventory: list[dict[str, Any]] = []
    remote_roots: set[str] = set()
    verified: list[dict[str, Any]] = []
    for episode, (source, item) in enumerate(zip(frozen.files, inventory, strict=True)):
        _require(item.get("episode") == episode, f"projection inventory identity drifted at {episode}")
        _require(item.get("source_path") == source["path"] and item.get("source_lfs_sha256") == source["lfs_sha256"], f"projection source drifted at {episode}")
        projection_name = f"episode_{episode:06d}.npz"
        checkpoint_name = f"episode_{episode:06d}.json"
        root_a, projection_path = _rebase_absolute(item.get("projection_file"), run_dir, "projections", projection_name)
        root_b, checkpoint_path = _rebase_absolute(item.get("checkpoint"), run_dir, "projection-checkpoints", checkpoint_name)
        _require(root_a == root_b, f"projection/checkpoint roots differ at {episode}")
        remote_roots.add(root_a)
        _require(projection_path.is_file(), f"projection missing at {episode}")
        _require(checkpoint_path.is_file(), f"projection sidecar missing at {episode}")
        projection_sha = _file_sha256(projection_path)
        checkpoint_sha = _file_sha256(checkpoint_path)
        _require(projection_sha == item.get("projection_file_sha256"), f"NPZ byte hash drifted at {episode}")
        _require(checkpoint_sha == item.get("checkpoint_file_sha256"), f"projection sidecar byte hash drifted at {episode}")
        checkpoint, checkpoint_raw = _load_json(checkpoint_path, f"projection checkpoint {episode}")
        _require(checkpoint_raw == _canonical_json_bytes(checkpoint) + b"\n", f"projection sidecar is not canonical at {episode}")
        _require(checkpoint.get("schema_version") == 1, f"projection sidecar schema drifted at {episode}")
        _require(checkpoint.get("binding_sha256") == frozen.binding_sha256, f"projection sidecar binding drifted at {episode}")
        _require(checkpoint.get("episode") == episode and checkpoint.get("source") == source, f"projection sidecar source drifted at {episode}")
        expected_remote = (
            f"datasets/{EXPECTED_REPO_ID}@{EXPECTED_REVISION}/{source['path']}"
        )
        _require(checkpoint.get("remote") == expected_remote, f"projection remote binding drifted at {episode}")
        _require(checkpoint.get("projection_columns") == list(PROJECTION_COLUMNS), f"projection column seal drifted at {episode}")
        _require(checkpoint.get("rows") == EXPECTED_ROWS_PER_EPISODE, f"projection row seal drifted at {episode}")
        _require(checkpoint.get("shape") == [EXPECTED_ROWS_PER_EPISODE, EXPECTED_DIM], f"projection shape seal drifted at {episode}")
        _require(checkpoint.get("dtype") == "float32", f"projection dtype seal drifted at {episode}")
        _require(checkpoint.get("full_source_parquet_locally_rehashed") is False, f"projection source-rehash seal drifted at {episode}")
        _require(
            checkpoint.get("source_verification")
            == "immutable revision plus frozen Hugging Face LFS metadata",
            f"projection source-verification seal drifted at {episode}",
        )
        _require(checkpoint.get("projection_file_sha256") == projection_sha, f"projection sidecar NPZ hash drifted at {episode}")
        unsigned = dict(checkpoint)
        claimed_content = unsigned.pop("checkpoint_content_sha256", None)
        _require(claimed_content == _json_sha256(unsigned), f"projection sidecar content hash drifted at {episode}")
        _require(claimed_content == item.get("checkpoint_content_sha256"), f"projection inventory content hash drifted at {episode}")
        arrays = _load_projection(projection_path, episode)
        data_sha = _data_digest(arrays)
        _require(data_sha == checkpoint.get("projected_data_sha256") == item.get("projected_data_sha256"), f"projection logical data hash drifted at {episode}")
        logical = _logical_projection_record(episode, source, arrays)
        _require(logical == checkpoint.get("logical_projection") == item.get("logical_projection"), f"projection logical record drifted at {episode}")
        arrays_by_episode.append(arrays)
        logical_inventory.append(logical)
        checkpoint_inventory.append({
            "episode": episode,
            "checkpoint_file_sha256": checkpoint_sha,
            "checkpoint_content_sha256": claimed_content,
        })
        verified.append({
            "episode": episode,
            "projection_file_sha256": projection_sha,
            "checkpoint_file_sha256": checkpoint_sha,
            "projected_data_sha256": data_sha,
            "checkpoint_content_sha256": claimed_content,
        })
    _require(len(remote_roots) == 1, "formal projection artifacts do not share one run root")
    projection_root = _json_sha256(logical_inventory)
    checkpoint_root = _json_sha256(checkpoint_inventory)
    _require(logical_result == logical_inventory, "result logical projection inventory drifted")
    _require(projection.get("root_sha256") == projection_root, "canonical projection root drifted")
    _require(projection.get("checkpoint_root_sha256") == checkpoint_root, "projection checkpoint root drifted")
    _require(projection.get("rows") == EXPECTED_FRAMES and projection.get("episodes") == EXPECTED_EPISODES, "projection counts drifted")
    return arrays_by_episode, {
        "remote_run_root": next(iter(remote_roots)),
        "projection_root_sha256": projection_root,
        "projection_checkpoint_root_sha256": checkpoint_root,
        "inventory": verified,
    }


def _analysis_logical(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition_binding": checkpoint.get("condition_binding"),
        "branches": checkpoint.get("branches"),
        "batch_one_negative_control_exact": checkpoint.get("batch_one_negative_control_exact"),
        "normalized_action_effect": checkpoint.get("normalized_action_effect"),
    }


def _verify_analysis_artifacts(
    result: dict[str, Any],
    run_dir: Path,
    frozen: FrozenInputs,
    projection_root: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conditions = result.get("conditions")
    inventory = result.get("analysis_checkpoint_inventory")
    expected_matrix = [(order, batch) for order in ORDER_NAMES for batch in BATCH_SIZES]
    _require(
        isinstance(conditions, list) and len(conditions) == len(expected_matrix),
        "result does not contain the exact frozen condition count",
    )
    _require(
        isinstance(inventory, list) and len(inventory) == len(expected_matrix),
        "analysis inventory does not contain the exact frozen condition count",
    )
    observed_matrix = [(item.get("order_name"), item.get("batch_size")) for item in inventory]
    _require(observed_matrix == expected_matrix, "analysis inventory matrix/order drifted")
    _require([(item.get("order_name"), item.get("batch_size")) for item in conditions] == expected_matrix, "result condition matrix/order drifted")
    logical_inventory: list[dict[str, Any]] = []
    checkpoint_inventory: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    remote_roots: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for condition, item, (order_name, batch_size) in zip(conditions, inventory, expected_matrix, strict=True):
        filename = f"{order_name}-batch-{batch_size:03d}.json"
        remote_root, path = _rebase_absolute(item.get("checkpoint"), run_dir, "analysis-checkpoints", filename)
        remote_roots.add(remote_root)
        _require(path.is_file(), f"analysis sidecar missing: {order_name}/{batch_size}")
        checkpoint, raw = _load_json(path, f"analysis checkpoint {order_name}/{batch_size}")
        file_sha = _bytes_sha256(raw)
        _require(raw == _canonical_json_bytes(checkpoint) + b"\n", f"analysis sidecar is not canonical: {order_name}/{batch_size}")
        _require(file_sha == item.get("checkpoint_file_sha256"), f"analysis sidecar byte hash drifted: {order_name}/{batch_size}")
        _require(checkpoint.get("schema_version") == 1, f"analysis schema drifted: {order_name}/{batch_size}")
        expected_binding = {
            "binding_sha256": frozen.binding_sha256,
            "projection_root_sha256": projection_root,
            "order_name": order_name,
            "order": list(frozen.orders[order_name]),
            "batch_size": batch_size,
        }
        _require(checkpoint.get("condition_binding") == expected_binding, f"condition binding drifted: {order_name}/{batch_size}")
        binding_sha = _json_sha256(expected_binding)
        _require(checkpoint.get("condition_binding_sha256") == binding_sha == item.get("condition_binding_sha256"), f"condition binding hash drifted: {order_name}/{batch_size}")
        unsigned = dict(checkpoint)
        content_sha = unsigned.pop("checkpoint_content_sha256", None)
        _require(content_sha == _json_sha256(unsigned) == item.get("checkpoint_content_sha256"), f"analysis content hash drifted: {order_name}/{batch_size}")
        logical = _analysis_logical(checkpoint)
        logical_sha = _json_sha256(logical)
        _require(logical_sha == checkpoint.get("analysis_logical_sha256") == item.get("analysis_logical_sha256"), f"analysis logical hash drifted: {order_name}/{batch_size}")
        expected_condition = {
            "order_name": order_name,
            "batch_size": batch_size,
            "branches": checkpoint["branches"],
            "batch_one_negative_control_exact": checkpoint["batch_one_negative_control_exact"],
            "normalized_action_effect": checkpoint["normalized_action_effect"],
        }
        _require(condition == expected_condition, f"result/checkpoint condition differs: {order_name}/{batch_size}")
        logical_inventory.append({"order_name": order_name, "batch_size": batch_size, "analysis_logical_sha256": logical_sha})
        checkpoint_inventory.append({
            "order_name": order_name,
            "batch_size": batch_size,
            "checkpoint_file_sha256": file_sha,
            "checkpoint_content_sha256": content_sha,
        })
        verified.append({
            "order_name": order_name,
            "batch_size": batch_size,
            "checkpoint_file_sha256": file_sha,
            "checkpoint_content_sha256": content_sha,
            "analysis_logical_sha256": logical_sha,
        })
        loaded.append(checkpoint)
    _require(len(remote_roots) == 1, "formal analysis sidecars do not share one run root")
    analysis_root = _json_sha256(logical_inventory)
    checkpoint_root = _json_sha256(checkpoint_inventory)
    _require(result.get("analysis_root_sha256") == analysis_root, "analysis logical root drifted")
    _require(result.get("analysis_checkpoint_root_sha256") == checkpoint_root, "analysis checkpoint root drifted")
    return loaded, {
        "remote_run_root": next(iter(remote_roots)),
        "analysis_root_sha256": analysis_root,
        "analysis_checkpoint_root_sha256": checkpoint_root,
        "inventory": verified,
    }


def _float64_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 2 and array.shape[1] == EXPECTED_DIM, "stats input is not [N,14]")
    mean = np.mean(array, axis=0, dtype=np.float64)
    centered = array - mean
    std = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64))
    return {"count": len(array), "mean": mean, "std": std}


def _historical_moments(
    values: np.ndarray, batch_size: int, branch: str
) -> dict[str, Any]:
    """Independently replay the historical float32 moment accumulator.

    Histogram state cannot affect these moments, so the auditor implements only
    the upstream count/mean/mean-of-squares recurrence and never imports the
    formal runner.
    """

    array = np.asarray(values)
    _require(
        array.dtype == np.float32
        and array.ndim == 2
        and array.shape[1] == EXPECTED_DIM,
        "historical moment input is not float32[N,14]",
    )
    _require(type(batch_size) is int and batch_size > 0, "historical batch size is invalid")
    _require(branch in {"old", "fixed"}, "historical moment branch is invalid")
    count = 0
    batch_count = 0
    mean: np.ndarray | None = None
    mean_of_squares: np.ndarray | None = None
    for start in range(0, len(array), batch_size):
        batch = np.asarray(array[start : start + batch_size])
        selected = np.asarray(batch[0]).reshape(1, -1) if branch == "old" else batch
        num_elements = selected.shape[0]
        if count == 0:
            mean = np.mean(selected, axis=0)
            mean_of_squares = np.mean(selected**2, axis=0)
        count += num_elements
        batch_mean = np.mean(selected, axis=0)
        batch_mean_of_squares = np.mean(selected**2, axis=0)
        assert mean is not None and mean_of_squares is not None
        mean += (batch_mean - mean) * (num_elements / count)
        mean_of_squares += (
            batch_mean_of_squares - mean_of_squares
        ) * (num_elements / count)
        batch_count += 1
    _require(count >= 2 and mean is not None and mean_of_squares is not None, "historical moment replay has fewer than two vectors")
    variance = mean_of_squares - mean**2
    std = np.sqrt(np.maximum(0, variance))
    return {
        "count": count,
        "batch_count": batch_count,
        "mean": np.asarray(mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
    }


def _recorded_stats(checkpoint: dict[str, Any], branch: str, feature: str) -> dict[str, Any]:
    try:
        stats = checkpoint["branches"][branch][feature]["statistics"]
        count = stats["count"]
        batch_count = stats["batch_count"]
        _require(type(count) is int, f"{branch}/{feature} count is not a strict integer")
        _require(type(batch_count) is int, f"{branch}/{feature} batch_count is not a strict integer")
        _require(stats.get("branch") == branch, f"{branch}/{feature} branch identity drifted")
        _require(type(stats.get("batch_size")) is int, f"{branch}/{feature} batch_size is not a strict integer")
        vectors: dict[str, np.ndarray] = {}
        for key in (
            "mean",
            "std",
            "q01",
            "q99",
            "min",
            "max",
            "final_histogram_bin_width",
        ):
            vector = np.asarray(stats[key], dtype=np.float64)
            _require(vector.shape == (EXPECTED_DIM,), f"{branch}/{feature}/{key} shape drifted")
            _require(np.all(np.isfinite(vector)), f"{branch}/{feature}/{key} is non-finite")
            vectors[key] = vector
        return {
            "raw": stats,
            "count": count,
            "batch_count": batch_count,
            "batch_size": stats["batch_size"],
            **vectors,
        }
    except (KeyError, TypeError, ValueError, AuditFailure) as error:
        if isinstance(error, AuditFailure):
            raise
        raise AuditFailure(f"invalid recorded stats for {branch}/{feature}: {error}") from error


def _verify_stat_vector(observed: np.ndarray, expected: np.ndarray, label: str) -> float:
    _require(observed.shape == (EXPECTED_DIM,), f"{label} shape drifted")
    _require(np.all(np.isfinite(observed)), f"{label} is non-finite")
    error = float(np.max(np.abs(observed - expected)))
    _require(error <= STAT_ABSOLUTE_TOLERANCE, f"{label} differs from independent float64 replay by {error}")
    return error


def _verify_tight_vector(
    observed: np.ndarray, expected: np.ndarray, label: str
) -> float:
    _require(observed.shape == expected.shape == (EXPECTED_DIM,), f"{label} shape drifted")
    _require(np.all(np.isfinite(observed)) and np.all(np.isfinite(expected)), f"{label} is non-finite")
    error = float(np.max(np.abs(observed - expected)))
    _require(
        error <= TIGHT_REPLAY_TOLERANCE,
        f"{label} differs from independent historical replay by {error}",
    )
    return error


def _effect(values: np.ndarray, old_mean: np.ndarray, old_std: np.ndarray, fixed_mean: np.ndarray, fixed_std: np.ndarray, thresholds: dict[str, Any]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    old_normalized = (array - old_mean) / (old_std + 1e-6)
    fixed_normalized = (array - fixed_mean) / (fixed_std + 1e-6)
    delta = old_normalized - fixed_normalized
    absolute = np.abs(delta)
    row_l2 = np.linalg.norm(delta, axis=1)
    maximum = float(np.max(absolute))
    median = float(np.median(row_l2))
    return {
        "rule": "z = (x - mean) / (std + 1e-6)",
        "evaluated_rows": len(array),
        "max_absolute_component_delta": maximum,
        "median_row_l2_delta": median,
        "p95_row_l2_delta": float(np.quantile(row_l2, 0.95)),
        "per_component_max_absolute_delta": [float(value) for value in np.max(absolute, axis=0)],
        "thresholds": thresholds,
        "material_under_frozen_threshold": bool(
            maximum >= thresholds["max_absolute_component_delta"]
            and median >= thresholds["median_row_l2_delta"]
        ),
    }


def _verify_numeric_claims(
    result: dict[str, Any],
    checkpoints: Sequence[dict[str, Any]],
    arrays_by_episode: Sequence[tuple[np.ndarray, ...]],
    frozen: FrozenInputs,
) -> dict[str, Any]:
    natural_state = np.concatenate([arrays[0] for arrays in arrays_by_episode], axis=0)
    natural_actions = np.concatenate([arrays[1] for arrays in arrays_by_episode], axis=0)
    _require(natural_state.shape == natural_actions.shape == (EXPECTED_FRAMES, EXPECTED_DIM), "full projection shape drifted")
    state_oracle = _float64_stats(natural_state)
    action_oracle = _float64_stats(natural_actions)
    for feature, oracle in (("state", state_oracle), ("actions", action_oracle)):
        recorded = result.get("oracle", {}).get(feature, {})
        _require(
            type(recorded.get("count")) is int
            and recorded.get("count") == EXPECTED_FRAMES,
            f"result {feature} oracle count drifted",
        )
        _verify_tight_vector(np.asarray(recorded.get("mean"), dtype=np.float64), oracle["mean"], f"result {feature} oracle mean")
        _verify_tight_vector(np.asarray(recorded.get("std"), dtype=np.float64), oracle["std"], f"result {feature} oracle std")

    condition_errors: list[dict[str, Any]] = []
    primary_checkpoint: dict[str, Any] | None = None
    primary_actions: np.ndarray | None = None
    exact_replay_counts = True
    old_undercoverage = True
    fixed_full_coverage = True
    fixed_mean_std = True
    batch_one_exact = True
    aggregation_decisions = True
    for checkpoint, (order_name, batch_size) in zip(
        checkpoints,
        ((order, batch) for order in ORDER_NAMES for batch in BATCH_SIZES),
        strict=True,
    ):
        order = frozen.orders[order_name]
        state = np.concatenate([arrays_by_episode[index][0] for index in order], axis=0)
        actions = np.concatenate([arrays_by_episode[index][1] for index in order], axis=0)
        expected_count = frozen.expected_counts[str(batch_size)]
        maxima: dict[str, float] = {}
        parsed: dict[tuple[str, str], dict[str, Any]] = {}
        for feature, values in (("state", state), ("actions", actions)):
            oracle = state_oracle if feature == "state" else action_oracle
            expected_fixed = _historical_moments(values, batch_size, "fixed")
            expected_old = _historical_moments(values, batch_size, "old")
            for branch, expected_stats, expected_vectors in (
                ("old", expected_old, expected_count["old_processed_vectors"]),
                ("fixed", expected_fixed, expected_count["fixed_processed_vectors"]),
            ):
                observed = _recorded_stats(checkpoint, branch, feature)
                parsed[(branch, feature)] = observed
                _require(observed["batch_size"] == batch_size, f"{order_name}/{batch_size}/{branch}/{feature} batch-size identity drifted")
                count_matches = observed["count"] == expected_vectors
                batch_count_matches = observed["batch_count"] == expected_count["batch_count_including_partial_tail"]
                exact_replay_counts &= count_matches and batch_count_matches
                _require(count_matches, f"{order_name}/{batch_size}/{branch}/{feature} count drifted")
                _require(batch_count_matches, f"{order_name}/{batch_size}/{branch}/{feature} batch count drifted")
                maxima[f"{branch}.{feature}.mean"] = _verify_tight_vector(observed["mean"], expected_stats["mean"], f"{order_name}/{batch_size}/{branch}/{feature}/mean")
                maxima[f"{branch}.{feature}.std"] = _verify_tight_vector(observed["std"], expected_stats["std"], f"{order_name}/{batch_size}/{branch}/{feature}/std")
                mean_error = float(np.max(np.abs(expected_stats["mean"] - oracle["mean"])))
                std_error = float(np.max(np.abs(expected_stats["std"] - oracle["std"])))
                within_tolerance = bool(
                    mean_error <= STAT_ABSOLUTE_TOLERANCE
                    and std_error <= STAT_ABSOLUTE_TOLERANCE
                )
                try:
                    recorded_error = checkpoint["branches"][branch][feature][
                        "oracle_error"
                    ]
                except (KeyError, TypeError) as error:
                    raise AuditFailure(
                        f"missing oracle error for {order_name}/{batch_size}/{branch}/{feature}"
                    ) from error
                for key, expected_error in (
                    ("max_absolute_mean_error", mean_error),
                    ("max_absolute_std_error", std_error),
                ):
                    recorded_value = recorded_error.get(key)
                    _require(
                        isinstance(recorded_value, float)
                        and math.isfinite(recorded_value)
                        and abs(recorded_value - expected_error)
                        <= TIGHT_REPLAY_TOLERANCE,
                        f"{order_name}/{batch_size}/{branch}/{feature}/{key} drifted",
                    )
                _require(
                    type(recorded_error.get("mean_std_within_frozen_tolerance"))
                    is bool
                    and recorded_error["mean_std_within_frozen_tolerance"]
                    == within_tolerance,
                    f"{order_name}/{batch_size}/{branch}/{feature} oracle tolerance decision drifted",
                )
                maxima[f"{branch}.{feature}.oracle_mean_error"] = mean_error
                maxima[f"{branch}.{feature}.oracle_std_error"] = std_error

        old_counts = [parsed[("old", feature)]["count"] for feature in ("state", "actions")]
        fixed_counts = [parsed[("fixed", feature)]["count"] for feature in ("state", "actions")]
        if batch_size == 1:
            for feature in ("state", "actions"):
                old_raw = parsed[("old", feature)]["raw"]
                fixed_raw = parsed[("fixed", feature)]["raw"]
                for key in (
                    "count",
                    "batch_count",
                    "mean",
                    "std",
                    "q01",
                    "q99",
                    "min",
                    "max",
                    "final_histogram_bin_width",
                ):
                    _require(
                        _canonical_json_bytes(old_raw.get(key))
                        == _canonical_json_bytes(fixed_raw.get(key)),
                        f"{order_name}/batch-1/{feature}/{key} old/fixed negative control differs",
                    )
            negative_control = checkpoint.get("batch_one_negative_control_exact") is True
            batch_one_exact &= negative_control
            _require(negative_control, f"{order_name}/batch-1 negative-control flag drifted")
        else:
            old_undercoverage &= all(value < EXPECTED_FRAMES for value in old_counts)
            negative_control = checkpoint.get("batch_one_negative_control_exact") is None
            batch_one_exact &= negative_control
            _require(negative_control, f"{order_name}/{batch_size} negative-control field must be null")
        fixed_full_coverage &= all(value == EXPECTED_FRAMES for value in fixed_counts)
        fixed_mean_std &= all(
            maxima[f"fixed.{feature}.oracle_{statistic}_error"]
            <= STAT_ABSOLUTE_TOLERANCE
            for feature in ("state", "actions")
            for statistic in ("mean", "std")
        )
        for branch in ("old", "fixed"):
            expected_accepted = all(
                parsed[(branch, feature)]["count"] == EXPECTED_FRAMES
                for feature in ("state", "actions")
            )
            recorded_accepted = checkpoint.get("branches", {}).get(branch, {}).get(
                "aggregation_contract_accepted"
            )
            _require(
                type(recorded_accepted) is bool,
                f"{order_name}/{batch_size}/{branch} aggregation decision is not boolean",
            )
            aggregation_decisions &= recorded_accepted == expected_accepted
            _require(
                recorded_accepted == expected_accepted,
                f"{order_name}/{batch_size}/{branch} aggregation decision drifted",
            )
        condition_errors.append({"order_name": order_name, "batch_size": batch_size, "max_absolute_errors": maxima})
        if order_name == "natural" and batch_size == PRIMARY_BATCH_SIZE:
            primary_checkpoint = checkpoint
            primary_actions = actions

    _require(primary_checkpoint is not None and primary_actions is not None, "primary condition absent")
    old = _recorded_stats(primary_checkpoint, "old", "actions")
    fixed = _recorded_stats(primary_checkpoint, "fixed", "actions")
    thresholds = frozen.section["design"]["material_normalized_action_effect_threshold"]
    recomputed_effect = _effect(primary_actions, old["mean"], old["std"], fixed["mean"], fixed["std"], thresholds)
    recorded_effect = primary_checkpoint.get("normalized_action_effect")
    _require(isinstance(recorded_effect, dict), "primary recorded action effect absent")
    for key in ("max_absolute_component_delta", "median_row_l2_delta", "p95_row_l2_delta"):
        _require(abs(float(recorded_effect[key]) - float(recomputed_effect[key])) <= EFFECT_ABSOLUTE_TOLERANCE, f"primary effect {key} drifted")
    _require(np.allclose(recorded_effect["per_component_max_absolute_delta"], recomputed_effect["per_component_max_absolute_delta"], rtol=0, atol=EFFECT_ABSOLUTE_TOLERANCE), "primary per-component action effect drifted")
    for key in ("rule", "evaluated_rows", "thresholds", "material_under_frozen_threshold"):
        _require(recorded_effect.get(key) == recomputed_effect[key], f"primary effect {key} drifted")
    primary_recorded = result.get("primary_condition")
    _require(isinstance(primary_recorded, dict), "result primary condition is absent")
    primary_state_old = _recorded_stats(primary_checkpoint, "old", "state")
    primary_actions_old = _recorded_stats(primary_checkpoint, "old", "actions")
    primary_state_fixed = _recorded_stats(primary_checkpoint, "fixed", "state")
    primary_actions_fixed = _recorded_stats(primary_checkpoint, "fixed", "actions")
    expected_primary = {
        "order_name": "natural",
        "batch_size": PRIMARY_BATCH_SIZE,
        "coverage_scope": (
            "raw one-state/one-action rows under the frozen include-partial-tail E2 protocol"
        ),
        "old_state_row_coverage": primary_state_old["count"] / EXPECTED_FRAMES,
        "old_action_row_coverage": primary_actions_old["count"] / EXPECTED_FRAMES,
        "fixed_state_row_coverage": primary_state_fixed["count"] / EXPECTED_FRAMES,
        "fixed_action_row_coverage": primary_actions_fixed["count"] / EXPECTED_FRAMES,
        "normalized_action_effect": recorded_effect,
    }
    _require(
        _canonical_json_bytes(primary_recorded)
        == _canonical_json_bytes(expected_primary),
        "result primary condition differs from independently reconstructed natural/B32 coverage and effect",
    )

    correctness = {
        "expected_episode_count": EXPECTED_EPISODES,
        "completed_episode_count": len(arrays_by_episode),
        "expected_frame_count": EXPECTED_FRAMES,
        "all_projection_checkpoints_complete": len(arrays_by_episode) == EXPECTED_EPISODES,
        "condition_matrix_exact_11_orders_x_4_batch_sizes": len(checkpoints)
        == len(ORDER_NAMES) * len(BATCH_SIZES),
        "exact_frozen_replay_counts_for_every_condition": bool(exact_replay_counts),
        "old_undercoverage_for_every_order_at_batch_8_32_128": bool(old_undercoverage),
        "fixed_full_coverage_for_every_order_and_batch": bool(fixed_full_coverage),
        "fixed_mean_std_within_1e_5_for_every_order_and_batch": bool(fixed_mean_std),
        "batch_one_old_fixed_exact_for_every_order": bool(batch_one_exact),
        "aggregation_contract_decisions_correct": bool(aggregation_decisions),
    }
    reproduced_keys = (
        "all_projection_checkpoints_complete",
        "condition_matrix_exact_11_orders_x_4_batch_sizes",
        "exact_frozen_replay_counts_for_every_condition",
        "old_undercoverage_for_every_order_at_batch_8_32_128",
        "fixed_full_coverage_for_every_order_and_batch",
        "fixed_mean_std_within_1e_5_for_every_order_and_batch",
        "batch_one_old_fixed_exact_for_every_order",
        "aggregation_contract_decisions_correct",
    )
    correctness["reproduced"] = all(correctness[key] for key in reproduced_keys)
    correctness["primary_normalized_action_effect_material"] = recomputed_effect[
        "material_under_frozen_threshold"
    ]
    official_auxiliary = result.get("oracle", {}).get(
        "official_meta_stats_crosscheck", {}
    ).get("all_mean_std_within_1e_5")
    _require(type(official_auxiliary) is bool, "official metadata auxiliary flag is invalid")
    correctness["official_meta_stats_match_full_projection_within_1e_5"] = official_auxiliary
    recorded_correctness = result.get("correctness")
    _require(isinstance(recorded_correctness, dict), "result correctness object is absent")
    _require(
        set(recorded_correctness) == set(correctness),
        "result correctness field set differs from independent reconstruction",
    )
    for key, expected in correctness.items():
        _require(
            type(recorded_correctness.get(key)) is type(expected)
            and recorded_correctness.get(key) == expected,
            f"result correctness field {key} differs from independent reconstruction",
        )
    expected_status = "REPRODUCED" if correctness["reproduced"] else "NOT_REPRODUCED"
    _require(result.get("status") == expected_status, "formal result status differs from independent gates")
    return {
        "independent_oracle": {
            "state": {"count": state_oracle["count"], "mean": state_oracle["mean"].tolist(), "std": state_oracle["std"].tolist()},
            "actions": {"count": action_oracle["count"], "mean": action_oracle["mean"].tolist(), "std": action_oracle["std"].tolist()},
        },
        "condition_mean_std_errors": condition_errors,
        "primary_normalized_action_effect": recomputed_effect,
        "correctness": correctness,
    }


def _verify_event_ledger(
    path: Path,
    binding_sha256: str,
    result: dict[str, Any],
    result_sha256: str,
    projections: dict[str, Any],
    analyses: dict[str, Any],
) -> dict[str, Any]:
    _require(path.is_file(), "event ledger is missing")
    events: list[dict[str, Any]] = []
    previous = ZERO_HASH
    with path.open("rb") as handle:
        for sequence, raw in enumerate(handle, start=1):
            _require(raw.endswith(b"\n"), f"event line {sequence} is torn")
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                raise AuditFailure(f"event line {sequence} is invalid: {error}") from error
            _require(type(event.get("schema_version")) is int and event.get("schema_version") == 1, f"event schema drifted at {sequence}")
            _require(type(event.get("sequence")) is int and event.get("sequence") == sequence, f"event sequence drifted at {sequence}")
            _require(event.get("binding_sha256") == binding_sha256, f"event binding drifted at {sequence}")
            _require(event.get("previous_event_sha256") == previous, f"event predecessor drifted at {sequence}")
            unsigned = dict(event)
            claimed = unsigned.pop("event_sha256", None)
            _require(claimed == _json_sha256(unsigned), f"event hash drifted at {sequence}")
            previous = claimed
            events.append(event)
    _require(events, "event ledger is empty")
    terminals = [event for event in events if event.get("event") in {"RUN_COMPLETE", "RUN_COMPLETE_RECOVERED"}]
    _require(len(terminals) == 1, "event ledger lacks one unique terminal")
    _require(terminals[0] is events[-1], "unique terminal is not the final ledger event")
    terminal_path = terminals[0].get("payload", {}).get("result")
    _require(isinstance(terminal_path, str) and PurePosixPath(terminal_path).is_absolute(), "terminal result path is invalid")
    expected_terminal = {
        "result": terminal_path,
        "result_sha256": result_sha256,
        "status": result.get("status"),
    }
    _require(
        _canonical_json_bytes(terminals[0].get("payload"))
        == _canonical_json_bytes(expected_terminal),
        "terminal payload does not exactly match the formal result",
    )

    prepared_events = [event for event in events if event.get("event") == "RESULT_PREPARED"]
    _require(len(prepared_events) == 1, "event ledger lacks one unique RESULT_PREPARED")
    expected_prepared = {
        "result": terminal_path,
        "result_sha256": result_sha256,
        "binding_sha256": binding_sha256,
        "execution_binding_sha256": _json_sha256(
            result.get("execution", {}).get("execution_binding")
        ),
        "projection_root_sha256": projections["projection_root_sha256"],
        "projection_checkpoint_root_sha256": projections[
            "projection_checkpoint_root_sha256"
        ],
        "analysis_root_sha256": analyses["analysis_root_sha256"],
        "analysis_checkpoint_root_sha256": analyses[
            "analysis_checkpoint_root_sha256"
        ],
        "condition_count": len(analyses["inventory"]),
    }
    _require(
        _canonical_json_bytes(prepared_events[0].get("payload"))
        == _canonical_json_bytes(expected_prepared),
        "RESULT_PREPARED does not exactly bind result, execution, roots, and condition count",
    )

    valid_projections = {
        item["episode"]: item["projected_data_sha256"]
        for item in projections["inventory"]
    }
    projected: dict[int, str] = {}
    for event in events:
        if event.get("event") not in {
            "PROJECTION_COMPLETE",
            "PROJECTION_CHECKPOINT_RECOVERED",
        }:
            continue
        payload = event.get("payload", {})
        episode = payload.get("episode")
        digest = payload.get("projected_data_sha256")
        _require(type(episode) is int and episode in valid_projections, "ledger names an unknown projection completion")
        _require(digest == valid_projections[episode], f"ledger projection completion conflicts at {episode}")
        _require(episode not in projected or projected[episode] == digest, f"ledger has conflicting duplicate projection completion at {episode}")
        projected[episode] = digest

    valid_analyses = {
        (item["order_name"], item["batch_size"]): (
            item["checkpoint_file_sha256"],
            item["analysis_logical_sha256"],
        )
        for item in analyses["inventory"]
    }
    analyzed: dict[tuple[str, int], tuple[str, str]] = {}
    for event in events:
        if event.get("event") != "ANALYSIS_COMPLETE":
            continue
        payload = event.get("payload", {})
        key = (payload.get("order"), payload.get("batch_size"))
        _require(
            isinstance(key[0], str) and type(key[1]) is int,
            "ledger analysis completion identity has invalid types",
        )
        observed = (
            payload.get("checkpoint_file_sha256"),
            payload.get("analysis_logical_sha256"),
        )
        _require(key in valid_analyses, "ledger names an unknown analysis completion")
        _require(observed == valid_analyses[key], f"ledger analysis completion conflicts at {key}")
        _require(key not in analyzed or analyzed[key] == observed, f"ledger has conflicting duplicate analysis completion at {key}")
        analyzed[key] = observed
    return {
        "event_count": len(events),
        "event_log_sha256": _file_sha256(path),
        "last_event_sha256": previous,
        "result_prepared_sequence": prepared_events[0]["sequence"],
        "terminal_sequence": terminals[0]["sequence"],
        "terminal_event": terminals[0]["event"],
        "projection_completion_bindings": len(projected),
        "analysis_completion_bindings": len(analyzed),
    }


def perform_audit(
    *,
    result_path: Path,
    run_dir: Path,
    formal_script: Path = DEFAULT_FORMAL_SCRIPT,
    manifest_path: Path = DEFAULT_MANIFEST,
    amendment_path: Path = DEFAULT_AMENDMENT,
    created_utc: str | None = None,
) -> dict[str, Any]:
    frozen = _load_frozen_inputs(manifest_path, amendment_path)
    result, result_raw, formal_sha = _verify_result_and_runner(result_path, formal_script, frozen)
    arrays, projection_audit = _verify_projection_artifacts(result, run_dir, frozen)
    checkpoints, analysis_audit = _verify_analysis_artifacts(
        result, run_dir, frozen, projection_audit["projection_root_sha256"]
    )
    _require(projection_audit["remote_run_root"] == analysis_audit["remote_run_root"], "projection and analysis formal roots differ")
    recomputed = _verify_numeric_claims(result, checkpoints, arrays, frozen)
    result_sha = _bytes_sha256(result_raw)
    ledger = _verify_event_ledger(
        run_dir / "events.jsonl",
        frozen.binding_sha256,
        result,
        result_sha,
        projection_audit,
        analysis_audit,
    )
    return {
        "schema_version": 1,
        "audit_id": AUDIT_ID,
        "created_utc": created_utc or _utc_now(),
        "status": "AUDIT_PASS",
        "scope": (
            "offline immutable-artifact, lineage-ledger, and independent outcome audit; "
            "AUDIT_PASS means the evidence and recorded formal determination agree, not that "
            "the defect was REPRODUCED; no network and no formal-run mutation"
        ),
        "inputs": {
            "result": str(result_path),
            "result_sha256": result_sha,
            "run_dir": str(run_dir),
            "formal_script": str(formal_script),
            "formal_script_sha256": formal_sha,
            "manifest": str(manifest_path),
            "manifest_sha256": frozen.manifest_sha256,
            "amendment": str(amendment_path),
            "amendment_sha256": frozen.amendment_sha256,
            "remote_run_root": projection_audit["remote_run_root"],
        },
        "verified": {
            "canonical_formal_result": True,
            "frozen_manifest_amendment_and_binding": True,
            "formal_runner_sha256": True,
            "event_hash_chain_and_unique_matching_terminal": True,
            "all_123_npz_and_projection_sidecars": True,
            "projection_byte_content_logical_roots": True,
            "all_44_analysis_sidecars": True,
            "analysis_byte_content_logical_roots": True,
            "exact_frozen_old_fixed_counts": True,
            "independent_float64_oracle_mean_std": True,
            "historical_float32_streaming_moments_match_recorded": True,
            "per_condition_float64_oracle_errors_recomputed": True,
            "primary_raw_z_score_effect": True,
            "formal_outcome_independently_recomputed": True,
        },
        "frozen_binding_sha256": frozen.binding_sha256,
        "projection": projection_audit,
        "analysis": analysis_audit,
        "event_ledger": ledger,
        "recomputed": recomputed,
        "auditor": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _file_sha256(Path(__file__).resolve()),
            "python": sys.version,
            "numpy": np.__version__,
        },
        "limitations": [
            "AUDIT_PASS certifies evidence integrity and agreement with the independently recomputed formal outcome; it does not mean REPRODUCED, and a trustworthy NOT_REPRODUCED result passes this audit.",
            "The auditor trusts the pre-outcome immutable-revision/LFS bindings and does not download or rehash the complete 72.1 GB image-bearing Parquets.",
            "The independent replay checks counts and exact selected populations plus float64 mean/std; it deliberately does not reimplement the historical 5,000-bin approximate quantiles.",
            "Remote absolute paths are accepted only after exact artifact-directory and filename suffix validation, then rebased onto --run-dir.",
        ],
    }


def _write_create_once(path: Path, payload: dict[str, Any]) -> str:
    serialized = _canonical_json_bytes(payload) + b"\n"
    digest = _bytes_sha256(serialized)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == serialized, f"create-once audit output already differs: {path}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == serialized, "create-once audit race differs")
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _existing_created_utc(path: Path) -> str | None:
    if not path.exists():
        return None
    existing, _ = _load_json(path, "existing audit output")
    _require(existing.get("audit_id") == AUDIT_ID, "existing audit identity drifted")
    created = existing.get("created_utc")
    _require(isinstance(created, str), "existing audit creation time invalid")
    return created


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formal-script", type=Path, default=DEFAULT_FORMAL_SCRIPT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
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
            created_utc=created_utc,
        )
        output_sha = _write_create_once(args.output, payload)
    except Exception as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({
        "status": payload["status"],
        "result_sha256": payload["inputs"]["result_sha256"],
        "projection_count": len(payload["projection"]["inventory"]),
        "condition_count": len(payload["analysis"]["inventory"]),
        "event_count": payload["event_ledger"]["event_count"],
        "output": str(args.output),
        "output_sha256": output_sha,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
