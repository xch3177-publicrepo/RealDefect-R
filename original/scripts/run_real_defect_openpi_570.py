#!/usr/bin/env python3
"""Replay OpenPI issue #570 / PR #619 operators on pinned low-dimensional data.

This is an E2 source-exact replay of the historical reducer and RunningStats
operators on a frozen raw projection.  It deliberately does not claim to run
the full image-bearing OpenPI DataLoader, its historical drop/tail behavior, or
its ALOHA/action-horizon/model transforms.  The frozen harness range-reads only
five Parquet columns, checkpoints every fixed episode, and can be resumed by
repeating the same command.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "contracts" / "real-defect-corpus-manifest.json"
DEFAULT_AMENDMENT = ROOT / "contracts" / "real-defect-corpus-freeze-amendment-v1.json"
DEFAULT_WORK_DIR = ROOT / "runs" / "p13-real-defect-openpi-570"
DEFAULT_OUTPUT = ROOT / "results" / "real-defect-openpi-570.json"

CASE_ID = "RD8-OPENPI-570-619-623"
SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
ZERO_HASH = "0" * 64
HF_ENDPOINT = "https://huggingface.co"
EXPECTED_MANIFEST_SHA256 = "7255524dc1776ef3e96e12fceea138ac76ae3bb15c55df42dc7e5a9e402e0d17"
EXPECTED_AMENDMENT_SHA256 = "58ad7063ddd44872d8fa1535e585f39079537d0a0715967f4784adf6d2adbd0c"
EXPECTED_ORDERS_SHA256 = "957d557de188190b591f98c28ceb4bc281b60be12765c31514cc3c1340fc5fd1"
EXPECTED_REPO_ID = "physical-intelligence/aloha_pen_uncap_diverse"
EXPECTED_REVISION = "e82d8b40b8ac66c0b40273dd80a077dfc40b732e"
EXPECTED_EPISODES = 123
EXPECTED_FRAMES = 36900
EXPECTED_ROWS_PER_EPISODE = 300
EXPECTED_DIM = 14
PROJECTION_COLUMNS = (
    "observation.state",
    "action",
    "episode_index",
    "frame_index",
    "timestamp",
)
ORDER_NAMES = ("natural", *(f"episode_cluster_seed_{seed}" for seed in range(10)))
BATCH_SIZES = (1, 8, 32, 128)
PRIMARY_BATCH_SIZE = 32
MEAN_STD_TOLERANCE = 1e-5
NUM_QUANTILE_BINS = 5000


class FailClosed(RuntimeError):
    """A frozen input, checkpoint, or analysis invariant failed."""


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


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value) + b"\n")


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    """Publish bytes once; an existing unequal artifact is a hard failure."""
    digest = _bytes_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FailClosed(f"immutable artifact differs: {path}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FailClosed(f"immutable artifact race differs: {path}")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _write_immutable_json(path: Path, value: Any) -> str:
    return _write_immutable_bytes(path, _canonical_json_bytes(value) + b"\n")


def _verify_event_log(path: Path, binding_sha256: str) -> tuple[int, str]:
    if not path.exists():
        return 0, ZERO_HASH
    sequence = 1
    previous = ZERO_HASH
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n"):
                raise FailClosed(f"torn event line {line_number}")
            try:
                event = json.loads(raw)
            except json.JSONDecodeError as error:
                raise FailClosed(f"invalid event line {line_number}: {error}") from error
            if event.get("schema_version") != EVENT_SCHEMA_VERSION:
                raise FailClosed(f"event schema mismatch at line {line_number}")
            if event.get("sequence") != sequence:
                raise FailClosed(f"event sequence mismatch at line {line_number}")
            if event.get("binding_sha256") != binding_sha256:
                raise FailClosed(f"event binding mismatch at line {line_number}")
            if event.get("previous_event_sha256") != previous:
                raise FailClosed(f"event predecessor mismatch at line {line_number}")
            claimed = event.get("event_sha256")
            unsigned = dict(event)
            unsigned.pop("event_sha256", None)
            if claimed != _json_sha256(unsigned):
                raise FailClosed(f"event hash mismatch at line {line_number}")
            previous = claimed
            sequence += 1
    return sequence - 1, previous


def _append_event(path: Path, binding_sha256: str, name: str, payload: dict[str, Any]) -> None:
    sequence, previous = _verify_event_log(path, binding_sha256)
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "sequence": sequence + 1,
        "previous_event_sha256": previous,
        "binding_sha256": binding_sha256,
        "utc": _utc_now(),
        "event": name,
        "payload": payload,
    }
    event["event_sha256"] = _json_sha256(event)
    existing = path.read_bytes() if path.exists() else b""
    # Replacing a fully materialized next ledger is crash-atomic: interruption
    # leaves either the complete predecessor or the complete successor, never
    # a partial final JSON line that would make the run permanently unresumable.
    _atomic_write_bytes(path, existing + _canonical_json_bytes(event) + b"\n")


@contextmanager
def _exclusive_lock(work_dir: Path) -> Iterable[None]:
    try:
        import fcntl
    except ImportError as error:
        raise FailClosed("RD8 requires a Unix advisory run lock") from error
    work_dir.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(work_dir / "run.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FailClosed("another RD8 runner holds the run lock") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


@dataclass(frozen=True)
class FrozenInputs:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    amendment_path: Path
    amendment_sha256: str
    amendment: dict[str, Any]
    section: dict[str, Any]
    files: tuple[dict[str, Any], ...]
    orders: dict[str, tuple[int, ...]]
    feature_names: tuple[str, ...]
    expected_counts: dict[str, dict[str, int]]
    binding: dict[str, Any]
    binding_sha256: str


def _load_json_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FailClosed(f"{label} is not a JSON object")
    return value, raw


def load_frozen_inputs(manifest_path: Path, amendment_path: Path) -> FrozenInputs:
    manifest, manifest_raw = _load_json_bytes(manifest_path, "manifest")
    amendment, amendment_raw = _load_json_bytes(amendment_path, "amendment")
    manifest_sha = _bytes_sha256(manifest_raw)
    amendment_sha = _bytes_sha256(amendment_raw)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise FailClosed("canonical manifest SHA-256 mismatch")
    if amendment_sha != EXPECTED_AMENDMENT_SHA256:
        raise FailClosed("freeze amendment SHA-256 mismatch")
    if manifest.get("corpus_id") != "P13-REALDEFECT-8":
        raise FailClosed("unexpected corpus identity")
    if amendment.get("amendment_id") != "P13-REALDEFECT-8-FREEZE-AMENDMENT-V1":
        raise FailClosed("unexpected freeze amendment identity")
    if amendment.get("canonical_manifest", {}).get("sha256") != manifest_sha:
        raise FailClosed("amendment does not bind the canonical manifest")
    if amendment.get("environment") != {
        "authentication": "disabled; public immutable artifacts only",
        "huggingface_hub_endpoint": HF_ENDPOINT,
    }:
        raise FailClosed("frozen Hugging Face environment drifted")
    try:
        section = manifest["rd8_openpi_570"]
        dataset = section["dataset"]
        source = section["source"]
        design = section["design"]
        rd8_amendment = amendment["rd8"]
        ordering = rd8_amendment["ordering"]
        features = rd8_amendment["features"]
        expected_counts = rd8_amendment["expected_replay_counts"]
    except (KeyError, TypeError) as error:
        raise FailClosed(f"malformed frozen RD8 section: {error}") from error
    if section.get("evidence_target") != (
        "E2 exact reducer/RunningStats loop replay on pinned public low-dimensional projection"
    ):
        raise FailClosed("RD8 evidence target drifted")
    if dataset.get("repo_id") != EXPECTED_REPO_ID or dataset.get("revision") != EXPECTED_REVISION:
        raise FailClosed("RD8 dataset identity drifted")
    if dataset.get("expected_episodes") != EXPECTED_EPISODES:
        raise FailClosed("RD8 expected episode count drifted")
    if dataset.get("expected_frames") != EXPECTED_FRAMES:
        raise FailClosed("RD8 expected frame count drifted")
    if dataset.get("projection_columns") != list(PROJECTION_COLUMNS):
        raise FailClosed("RD8 projection columns drifted")
    files = dataset.get("files")
    if not isinstance(files, list) or len(files) != EXPECTED_EPISODES:
        raise FailClosed("RD8 Parquet file inventory is not 123 entries")
    checked_files: list[dict[str, Any]] = []
    for index, record in enumerate(files):
        expected_path = f"data/chunk-000/episode_{index:06d}.parquet"
        if not isinstance(record, dict) or record.get("path") != expected_path:
            raise FailClosed(f"RD8 path inventory drifted at episode {index}")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise FailClosed(f"RD8 source byte size is invalid at episode {index}")
        if not _is_hex(record.get("git_blob_oid"), 40) or not _is_hex(record.get("lfs_sha256"), 64):
            raise FailClosed(f"RD8 source hashes are invalid at episode {index}")
        checked_files.append(dict(record))
    if design.get("batch_sizes") != list(BATCH_SIZES) or design.get("primary_batch_size") != 32:
        raise FailClosed("RD8 batch design drifted")
    if design.get("orders") != list(ORDER_NAMES):
        raise FailClosed("RD8 order labels drifted")
    if design.get("mean_std_absolute_tolerance") != MEAN_STD_TOLERANCE:
        raise FailClosed("RD8 numeric tolerance drifted")
    if ordering.get("orders_canonical_sha256") != EXPECTED_ORDERS_SHA256:
        raise FailClosed("RD8 frozen order root drifted")
    orders_raw = ordering.get("orders")
    orders_digest = _bytes_sha256(_canonical_json_bytes(orders_raw) + b"\n")
    if not isinstance(orders_raw, dict) or orders_digest != EXPECTED_ORDERS_SHA256:
        raise FailClosed("RD8 complete frozen orders differ")
    orders: dict[str, tuple[int, ...]] = {}
    for name in ORDER_NAMES:
        values = orders_raw.get(name)
        if not isinstance(values, list) or sorted(values) != list(range(EXPECTED_EPISODES)):
            raise FailClosed(f"invalid frozen episode order: {name}")
        orders[name] = tuple(int(value) for value in values)
    state_feature = features.get("state")
    action_feature = features.get("actions")
    if state_feature.get("source_key") != "observation.state" or action_feature.get("source_key") != "action":
        raise FailClosed("RD8 feature source keys drifted")
    if state_feature.get("dtype") != "float32" or action_feature.get("dtype") != "float32":
        raise FailClosed("RD8 feature dtype drifted")
    if state_feature.get("shape") != [14] or action_feature.get("shape") != [14]:
        raise FailClosed("RD8 feature shape drifted")
    if state_feature.get("names") != action_feature.get("names"):
        raise FailClosed("RD8 state/action feature order differs")
    feature_names = tuple(state_feature["names"])
    if len(feature_names) != EXPECTED_DIM or len(set(feature_names)) != EXPECTED_DIM:
        raise FailClosed("RD8 feature-name vector is invalid")
    for size in BATCH_SIZES:
        expected = {
            "batch_count_including_partial_tail": math.ceil(EXPECTED_FRAMES / size),
            "old_processed_vectors": math.ceil(EXPECTED_FRAMES / size),
            "fixed_processed_vectors": EXPECTED_FRAMES,
        }
        if expected_counts.get(str(size)) != expected:
            raise FailClosed(f"RD8 expected replay counts drifted for batch {size}")
    metadata = rd8_amendment.get("official_metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
        "meta/info.json", "meta/episodes.jsonl", "meta/stats.json", "meta/tasks.jsonl"
    }:
        raise FailClosed("RD8 official metadata binding is incomplete")
    for path, record in metadata.items():
        if not _is_hex(record.get("downloaded_sha256"), 64):
            raise FailClosed(f"RD8 metadata content hash is invalid: {path}")
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
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        amendment_path=amendment_path,
        amendment_sha256=amendment_sha,
        amendment=amendment,
        section=section,
        files=tuple(checked_files),
        orders=orders,
        feature_names=feature_names,
        expected_counts={str(key): dict(value) for key, value in expected_counts.items()},
        binding=binding,
        binding_sha256=_json_sha256(binding),
    )


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for distribution in ("numpy", "pyarrow", "huggingface-hub", "fsspec"):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _execution_binding() -> dict[str, Any]:
    return {
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }


def _load_state(path: Path, frozen: FrozenInputs, execution: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "binding": frozen.binding,
            "binding_sha256": frozen.binding_sha256,
            "execution_binding": execution,
            "run_created_utc": _utc_now(),
            "status": "INITIALIZED",
            "projection_root_sha256": None,
            "projection_checkpoint_root_sha256": None,
            "analysis_root_sha256": None,
            "analysis_checkpoint_root_sha256": None,
            "result_prepared": None,
            "failure": None,
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(f"invalid RD8 state: {error}") from error
    if state.get("schema_version") != SCHEMA_VERSION:
        raise FailClosed("RD8 state schema drifted")
    if state.get("binding") != frozen.binding or state.get("binding_sha256") != frozen.binding_sha256:
        raise FailClosed("RD8 state frozen binding drifted")
    if state.get("execution_binding") != execution:
        raise FailClosed("RD8 execution environment or script drifted")
    if not isinstance(state.get("run_created_utc"), str):
        raise FailClosed("RD8 run creation time is absent")
    return state


def _episode_from_path(path: str) -> int:
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != path:
        raise FailClosed(f"unsafe RD8 source path: {path}")
    stem = parsed.stem
    if not stem.startswith("episode_"):
        raise FailClosed(f"RD8 source path has no episode identity: {path}")
    return int(stem.removeprefix("episode_"))


def _projection_path(work_dir: Path, episode: int) -> Path:
    return work_dir / "projections" / f"episode_{episode:06d}.npz"


def _projection_checkpoint_path(work_dir: Path, episode: int) -> Path:
    return work_dir / "projection-checkpoints" / f"episode_{episode:06d}.json"


def _analysis_checkpoint_path(work_dir: Path, order_name: str, batch_size: int) -> Path:
    return work_dir / "analysis-checkpoints" / f"{order_name}-batch-{batch_size:03d}.json"


def _data_digest(
    state: np.ndarray,
    actions: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    timestamp: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("state", state),
        ("actions", actions),
        ("episode_index", episode_index),
        ("frame_index", frame_index),
        ("timestamp", timestamp),
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode() + b"\0")
        digest.update(str(contiguous.dtype).encode() + b"\0")
        digest.update(_canonical_json_bytes(list(contiguous.shape)))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _array_logical_descriptor(name: str, value: np.ndarray) -> dict[str, Any]:
    """Describe projected values without paths, timestamps, or container bytes."""
    contiguous = np.ascontiguousarray(value)
    return {
        "name": name,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "logical_sha256": _bytes_sha256(contiguous.tobytes(order="C")),
    }


def _logical_projection_record(
    episode: int,
    record: dict[str, Any],
    arrays: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    names = ("state", "actions", "episode_index", "frame_index", "timestamp")
    return {
        "episode": episode,
        "source_path": record["path"],
        "source_lfs_sha256": record["lfs_sha256"],
        "arrays": [
            _array_logical_descriptor(name, value)
            for name, value in zip(names, arrays, strict=True)
        ],
    }


def _validate_episode_arrays(
    episode: int,
    state: np.ndarray,
    actions: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    timestamp: np.ndarray,
) -> None:
    if state.dtype != np.float32 or state.shape != (EXPECTED_ROWS_PER_EPISODE, EXPECTED_DIM):
        raise FailClosed(f"episode {episode} state is not float32[300,14]")
    if actions.dtype != np.float32 or actions.shape != (EXPECTED_ROWS_PER_EPISODE, EXPECTED_DIM):
        raise FailClosed(f"episode {episode} actions are not float32[300,14]")
    if episode_index.dtype != np.int64 or episode_index.shape != (EXPECTED_ROWS_PER_EPISODE,):
        raise FailClosed(f"episode {episode} episode_index is not int64[300]")
    if frame_index.dtype != np.int64 or frame_index.shape != (EXPECTED_ROWS_PER_EPISODE,):
        raise FailClosed(f"episode {episode} frame_index is not int64[300]")
    if timestamp.dtype != np.float32 or timestamp.shape != (EXPECTED_ROWS_PER_EPISODE,):
        raise FailClosed(f"episode {episode} timestamp is not float32[300]")
    if not np.all(episode_index == episode):
        raise FailClosed(f"episode {episode} contains another episode_index")
    if not np.array_equal(frame_index, np.arange(EXPECTED_ROWS_PER_EPISODE, dtype=np.int64)):
        raise FailClosed(f"episode {episode} frame_index is not exactly 0..299")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(actions)):
        raise FailClosed(f"episode {episode} contains non-finite state/action")
    if not np.all(np.isfinite(timestamp)) or np.any(np.diff(timestamp.astype(np.float64)) < 0):
        raise FailClosed(f"episode {episode} timestamp is non-finite or decreasing")


def _write_projection_npz(
    path: Path,
    state: np.ndarray,
    actions: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    timestamp: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle,
                state=state,
                actions=actions,
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _load_projection_npz(path: Path, episode: int) -> tuple[np.ndarray, ...]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"state", "actions", "episode_index", "frame_index", "timestamp"}:
                raise FailClosed(f"episode {episode} projection fields drifted")
            values = tuple(
                np.asarray(archive[name])
                for name in ("state", "actions", "episode_index", "frame_index", "timestamp")
            )
    except (OSError, ValueError) as error:
        raise FailClosed(f"cannot load episode {episode} projection: {error}") from error
    _validate_episode_arrays(episode, *values)
    return values


def _fixed_list_column_to_numpy(column: Any, expected_name: str) -> np.ndarray:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise FailClosed("RD8 projection requires pyarrow") from error
    chunks: list[np.ndarray] = []
    for chunk in column.chunks:
        if not pa.types.is_fixed_size_list(chunk.type) or chunk.type.list_size != EXPECTED_DIM:
            raise FailClosed(f"{expected_name} is not fixed_size_list<float>[14]")
        if not pa.types.is_float32(chunk.type.value_type):
            raise FailClosed(f"{expected_name} child type is not float32")
        if chunk.null_count or chunk.values.null_count:
            raise FailClosed(f"{expected_name} contains nulls")
        child = chunk.values.slice(chunk.offset * EXPECTED_DIM, len(chunk) * EXPECTED_DIM)
        values = np.asarray(child.to_numpy(zero_copy_only=False), dtype=np.float32)
        chunks.append(values.reshape(len(chunk), EXPECTED_DIM))
    if not chunks:
        raise FailClosed(f"{expected_name} has no Arrow chunks")
    return np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float32)


def _scalar_column_to_numpy(column: Any, dtype: np.dtype, expected_name: str) -> np.ndarray:
    if any(chunk.null_count for chunk in column.chunks):
        raise FailClosed(f"{expected_name} contains nulls")
    values = [np.asarray(chunk.to_numpy(zero_copy_only=False)) for chunk in column.chunks]
    if not values:
        raise FailClosed(f"{expected_name} has no Arrow chunks")
    result = np.concatenate(values)
    if result.dtype != dtype:
        raise FailClosed(f"{expected_name} dtype is {result.dtype}, expected {dtype}")
    return np.ascontiguousarray(result)


def _project_parquet_handle(handle: Any, episode: int) -> tuple[np.ndarray, ...]:
    """Read exactly the frozen low-dimensional projection from a Parquet handle."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise FailClosed("RD8 projection requires pyarrow") from error
    parquet = pq.ParquetFile(handle)
    if parquet.metadata.num_rows != EXPECTED_ROWS_PER_EPISODE:
        raise FailClosed(f"episode {episode} Parquet footer row count is not 300")
    table = parquet.read(columns=list(PROJECTION_COLUMNS), use_threads=False)
    if table.num_rows != EXPECTED_ROWS_PER_EPISODE or table.column_names != list(PROJECTION_COLUMNS):
        raise FailClosed(f"episode {episode} projected table shape/schema drifted")
    state = _fixed_list_column_to_numpy(table["observation.state"], "observation.state")
    actions = _fixed_list_column_to_numpy(table["action"], "action")
    episode_index = _scalar_column_to_numpy(table["episode_index"], np.dtype("int64"), "episode_index")
    frame_index = _scalar_column_to_numpy(table["frame_index"], np.dtype("int64"), "frame_index")
    timestamp = _scalar_column_to_numpy(table["timestamp"], np.dtype("float32"), "timestamp")
    _validate_episode_arrays(episode, state, actions, episode_index, frame_index, timestamp)
    return state, actions, episode_index, frame_index, timestamp


def _project_remote_episode(frozen: FrozenInputs, record: dict[str, Any], episode: int) -> tuple[np.ndarray, ...]:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as error:
        raise FailClosed("RD8 network projection requires huggingface_hub and pyarrow") from error
    filesystem = HfFileSystem(endpoint=HF_ENDPOINT, token=False)
    remote = f"datasets/{EXPECTED_REPO_ID}@{EXPECTED_REVISION}/{record['path']}"
    try:
        with filesystem.open(remote, "rb") as handle:
            return _project_parquet_handle(handle, episode)
    except FailClosed:
        raise
    except Exception as error:
        raise FailClosed(f"range projection failed for episode {episode}: {error}") from error


def _load_projection_checkpoint(
    frozen: FrozenInputs,
    work_dir: Path,
    episode: int,
    record: dict[str, Any],
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]] | None:
    checkpoint_path = _projection_checkpoint_path(work_dir, episode)
    projection_path = _projection_path(work_dir, episode)
    if not checkpoint_path.exists() and not projection_path.exists():
        return None
    if checkpoint_path.exists() and not projection_path.exists():
        raise FailClosed(f"episode {episode} has a partial projection checkpoint")
    if projection_path.exists() and not checkpoint_path.exists():
        arrays = _load_projection_npz(projection_path, episode)
        return arrays, {"unsealed_projection_requires_remote_verification": True}
    checkpoint, _ = _load_json_bytes(checkpoint_path, f"episode {episode} checkpoint")
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise FailClosed(f"episode {episode} projection checkpoint schema drifted")
    if checkpoint.get("binding_sha256") != frozen.binding_sha256:
        raise FailClosed(f"episode {episode} projection checkpoint binding drifted")
    if checkpoint.get("episode") != episode or checkpoint.get("source") != record:
        raise FailClosed(f"episode {episode} projection checkpoint source drifted")
    if _file_sha256(projection_path) != checkpoint.get("projection_file_sha256"):
        raise FailClosed(f"episode {episode} projection file hash drifted")
    arrays = _load_projection_npz(projection_path, episode)
    if _data_digest(*arrays) != checkpoint.get("projected_data_sha256"):
        raise FailClosed(f"episode {episode} projected data hash drifted")
    logical_record = _logical_projection_record(episode, record, arrays)
    if checkpoint.get("logical_projection") != logical_record:
        raise FailClosed(f"episode {episode} logical projection descriptor drifted")
    unsigned = dict(checkpoint)
    claimed = unsigned.pop("checkpoint_content_sha256", None)
    if claimed != _json_sha256(unsigned):
        raise FailClosed(f"episode {episode} checkpoint content digest drifted")
    return arrays, checkpoint


def _seal_projection_checkpoint(
    frozen: FrozenInputs,
    work_dir: Path,
    episode: int,
    record: dict[str, Any],
    arrays: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    projection_path = _projection_path(work_dir, episode)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "binding_sha256": frozen.binding_sha256,
        "episode": episode,
        "source": record,
        "remote": f"datasets/{EXPECTED_REPO_ID}@{EXPECTED_REVISION}/{record['path']}",
        "projection_columns": list(PROJECTION_COLUMNS),
        "rows": EXPECTED_ROWS_PER_EPISODE,
        "shape": [EXPECTED_ROWS_PER_EPISODE, EXPECTED_DIM],
        "dtype": "float32",
        "full_source_parquet_locally_rehashed": False,
        "source_verification": "immutable revision plus frozen Hugging Face LFS metadata",
        "projection_file_sha256": _file_sha256(projection_path),
        "projected_data_sha256": _data_digest(*arrays),
        "logical_projection": _logical_projection_record(episode, record, arrays),
        "completed_utc": _utc_now(),
    }
    checkpoint["checkpoint_content_sha256"] = _json_sha256(checkpoint)
    _write_immutable_json(_projection_checkpoint_path(work_dir, episode), checkpoint)
    return checkpoint


def _complete_projection(
    frozen: FrozenInputs,
    work_dir: Path,
    episode: int,
    record: dict[str, Any],
    allow_download: bool,
    events_path: Path,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    existing = _load_projection_checkpoint(frozen, work_dir, episode, record)
    if existing is not None:
        arrays, checkpoint = existing
        if checkpoint.get("unsealed_projection_requires_remote_verification"):
            if not allow_download:
                raise FailClosed(
                    f"episode {episode} has an unsealed projection; --download is required for source verification"
                )
            remote_arrays = _project_remote_episode(frozen, record, episode)
            if any(
                not np.array_equal(local, remote)
                for local, remote in zip(arrays, remote_arrays, strict=True)
            ):
                raise FailClosed(
                    f"episode {episode} unsealed projection differs from the pinned remote source"
                )
            checkpoint = _seal_projection_checkpoint(
                frozen, work_dir, episode, record, remote_arrays
            )
            _append_event(
                events_path,
                frozen.binding_sha256,
                "PROJECTION_CHECKPOINT_RECOVERED",
                {
                    "episode": episode,
                    "projected_data_sha256": checkpoint["projected_data_sha256"],
                    "verification": "exact array equality after pinned remote re-projection",
                },
            )
            arrays = remote_arrays
        return arrays, checkpoint
    if not allow_download:
        raise FailClosed(f"episode {episode} projection is absent and --download was not authorized")
    _append_event(events_path, frozen.binding_sha256, "PROJECTION_BEGIN", {"episode": episode, "path": record["path"]})
    arrays = _project_remote_episode(frozen, record, episode)
    projection_path = _projection_path(work_dir, episode)
    _write_projection_npz(projection_path, *arrays)
    checkpoint = _seal_projection_checkpoint(
        frozen, work_dir, episode, record, arrays
    )
    _append_event(
        events_path,
        frozen.binding_sha256,
        "PROJECTION_COMPLETE",
        {"episode": episode, "projected_data_sha256": checkpoint["projected_data_sha256"]},
    )
    return arrays, checkpoint


def _ensure_official_metadata(
    frozen: FrozenInputs,
    work_dir: Path,
    allow_download: bool,
) -> dict[str, bytes]:
    records = frozen.amendment["rd8"]["official_metadata"]
    contents: dict[str, bytes] = {}
    for remote_path, record in sorted(records.items()):
        local_path = work_dir / "official-metadata" / remote_path.removeprefix("meta/")
        if local_path.exists():
            content = local_path.read_bytes()
        else:
            if not allow_download:
                raise FailClosed(f"official metadata is absent and --download was not authorized: {remote_path}")
            try:
                from huggingface_hub import hf_hub_download
                downloaded = Path(
                    hf_hub_download(
                        repo_id=EXPECTED_REPO_ID,
                        repo_type="dataset",
                        revision=EXPECTED_REVISION,
                        filename=remote_path,
                        endpoint=HF_ENDPOINT,
                        token=False,
                    )
                )
                content = downloaded.read_bytes()
            except Exception as error:
                raise FailClosed(f"cannot download official metadata {remote_path}: {error}") from error
            _write_immutable_bytes(local_path, content)
        if len(content) != record["bytes"] or _bytes_sha256(content) != record["downloaded_sha256"]:
            raise FailClosed(f"official metadata bytes drifted: {remote_path}")
        contents[remote_path] = content
    return contents


class HistoricalRunningStats:
    """Exact PR #619 RunningStats logic, with an equivalent one-row fast path."""

    def __init__(self) -> None:
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = NUM_QUANTILE_BINS

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch)
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if num_elements == 0:
            raise FailClosed("RunningStats received an empty batch")
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[index] - 1e-10, self._max[index] + 1e-10, self._num_quantile_bins + 1)
                for index in range(vector_length)
            ]
        else:
            assert self._mean is not None and self._min is not None and self._max is not None
            if vector_length != self._mean.size:
                raise FailClosed("RunningStats vector length drifted")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)
        assert self._mean is not None and self._mean_of_squares is not None
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (
            batch_mean_of_squares - self._mean_of_squares
        ) * (num_elements / self._count)
        self._update_histograms(batch)

    def _adjust_histograms(self) -> None:
        assert self._histograms is not None and self._bin_edges is not None
        assert self._min is not None and self._max is not None
        for index in range(len(self._histograms)):
            old_edges = self._bin_edges[index]
            new_edges = np.linspace(self._min[index], self._max[index], self._num_quantile_bins + 1)
            new_histogram, _ = np.histogram(
                old_edges[:-1], bins=new_edges, weights=self._histograms[index]
            )
            self._histograms[index] = new_histogram
            self._bin_edges[index] = new_edges

    @staticmethod
    def _single_value_bin(value: float, edges: np.ndarray) -> int | None:
        if value < edges[0] or value > edges[-1]:
            return None
        if value == edges[-1]:
            return len(edges) - 2
        index = int(np.searchsorted(edges, value, side="right") - 1)
        return index if 0 <= index < len(edges) - 1 else None

    def _update_histograms(self, batch: np.ndarray) -> None:
        assert self._histograms is not None and self._bin_edges is not None
        if batch.shape[0] == 1:
            for index in range(batch.shape[1]):
                bin_index = self._single_value_bin(float(batch[0, index]), self._bin_edges[index])
                if bin_index is not None:
                    self._histograms[index][bin_index] += 1
            return
        for index in range(batch.shape[1]):
            histogram, _ = np.histogram(batch[:, index], bins=self._bin_edges[index])
            self._histograms[index] += histogram

    def statistics(self) -> dict[str, Any]:
        if self._count < 2:
            raise FailClosed("cannot compute RunningStats for fewer than two vectors")
        assert self._mean is not None and self._mean_of_squares is not None
        assert self._min is not None and self._max is not None
        assert self._histograms is not None and self._bin_edges is not None
        variance = self._mean_of_squares - self._mean**2
        std = np.sqrt(np.maximum(0, variance))
        quantiles: list[np.ndarray] = []
        for quantile in (0.01, 0.99):
            target = quantile * self._count
            values: list[float] = []
            for histogram, edges in zip(self._histograms, self._bin_edges, strict=True):
                index = int(np.searchsorted(np.cumsum(histogram), target))
                values.append(float(edges[index]))
            quantiles.append(np.asarray(values, dtype=np.float64))
        bin_widths = np.asarray(
            [float(np.max(np.diff(edges))) for edges in self._bin_edges], dtype=np.float64
        )
        return {
            "count": self._count,
            "mean": np.asarray(self._mean, dtype=np.float64),
            "std": np.asarray(std, dtype=np.float64),
            "q01": quantiles[0],
            "q99": quantiles[1],
            "min": np.asarray(self._min, dtype=np.float64),
            "max": np.asarray(self._max, dtype=np.float64),
            "final_histogram_bin_width": bin_widths,
            "histogram_count_min": int(min(np.sum(value) for value in self._histograms)),
            "histogram_count_max": int(max(np.sum(value) for value in self._histograms)),
        }


def replay_reducer(values: np.ndarray, batch_size: int, branch: str) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != EXPECTED_DIM:
        raise FailClosed("reducer input must be [N,14]")
    if batch_size <= 0:
        raise FailClosed("batch size must be positive")
    stats = HistoricalRunningStats()
    batches = 0
    for start in range(0, len(values), batch_size):
        batch = np.asarray(values[start : start + batch_size])
        if branch == "old":
            historical = np.asarray(batch[0])
            stats.update(historical.reshape(-1, historical.shape[-1]))
        elif branch == "fixed":
            stats.update(np.asarray(batch))
        else:
            raise FailClosed(f"unknown reducer branch: {branch}")
        batches += 1
    result = stats.statistics()
    result["batch_count"] = batches
    result["branch"] = branch
    result["batch_size"] = batch_size
    return result


def independent_oracle(values: np.ndarray) -> dict[str, np.ndarray | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != EXPECTED_DIM or not np.all(np.isfinite(array)):
        raise FailClosed("oracle input is not finite [N,14]")
    mean = np.mean(array, axis=0, dtype=np.float64)
    centered = array - mean
    std = np.sqrt(np.mean(centered * centered, axis=0, dtype=np.float64))
    q01, q99 = np.quantile(array, (0.01, 0.99), axis=0)
    return {"count": len(array), "mean": mean, "std": std, "q01": q01, "q99": q99}


def _array_list(value: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(value).tolist()]


def _stats_to_json(stats: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            result[key] = _array_list(value)
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _stats_from_json(value: dict[str, Any]) -> dict[str, Any]:
    arrays = {"mean", "std", "q01", "q99", "min", "max", "final_histogram_bin_width"}
    return {
        key: np.asarray(item, dtype=np.float64) if key in arrays else item
        for key, item in value.items()
    }


def _stats_error(stats: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    mean_error = np.abs(stats["mean"] - oracle["mean"])
    std_error = np.abs(stats["std"] - oracle["std"])
    q01_error = np.abs(stats["q01"] - oracle["q01"])
    q99_error = np.abs(stats["q99"] - oracle["q99"])
    quantile_tolerance = np.maximum(1e-3, 2.0 * stats["final_histogram_bin_width"])
    return {
        "max_absolute_mean_error": float(np.max(mean_error)),
        "max_absolute_std_error": float(np.max(std_error)),
        "max_absolute_q01_error": float(np.max(q01_error)),
        "max_absolute_q99_error": float(np.max(q99_error)),
        "mean_std_within_frozen_tolerance": bool(
            np.all(mean_error <= MEAN_STD_TOLERANCE)
            and np.all(std_error <= MEAN_STD_TOLERANCE)
        ),
        "quantiles_within_descriptive_histogram_tolerance": bool(
            np.all(q01_error <= quantile_tolerance)
            and np.all(q99_error <= quantile_tolerance)
        ),
        "max_quantile_tolerance": float(np.max(quantile_tolerance)),
    }


def _branch_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["count"] != right["count"] or left["batch_count"] != right["batch_count"]:
        return False
    for key in ("mean", "std", "q01", "q99", "min", "max", "final_histogram_bin_width"):
        if not np.array_equal(left[key], right[key]):
            return False
    return True


def _normalized_action_effect(
    actions: np.ndarray,
    old_stats: dict[str, Any],
    fixed_stats: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    array = np.asarray(actions, dtype=np.float64)
    old_normalized = (array - old_stats["mean"]) / (old_stats["std"] + 1e-6)
    fixed_normalized = (array - fixed_stats["mean"]) / (fixed_stats["std"] + 1e-6)
    absolute = np.abs(old_normalized - fixed_normalized)
    row_l2 = np.linalg.norm(old_normalized - fixed_normalized, axis=1)
    maximum = float(np.max(absolute))
    median_l2 = float(np.median(row_l2))
    component_maxima = np.max(absolute, axis=0)
    material = (
        maximum >= thresholds["max_absolute_component_delta"]
        and median_l2 >= thresholds["median_row_l2_delta"]
    )
    return {
        "rule": "z = (x - mean) / (std + 1e-6)",
        "evaluated_rows": len(array),
        "max_absolute_component_delta": maximum,
        "median_row_l2_delta": median_l2,
        "p95_row_l2_delta": float(np.quantile(row_l2, 0.95)),
        "per_component_max_absolute_delta": _array_list(component_maxima),
        "thresholds": thresholds,
        "material_under_frozen_threshold": bool(material),
    }


def _condition_checkpoint(
    frozen: FrozenInputs,
    work_dir: Path,
    order_name: str,
    batch_size: int,
    projection_root: str,
    state_values: np.ndarray,
    action_values: np.ndarray,
    state_oracle: dict[str, Any],
    action_oracle: dict[str, Any],
    events_path: Path,
) -> dict[str, Any]:
    path = _analysis_checkpoint_path(work_dir, order_name, batch_size)
    condition_binding = {
        "binding_sha256": frozen.binding_sha256,
        "projection_root_sha256": projection_root,
        "order_name": order_name,
        "order": list(frozen.orders[order_name]),
        "batch_size": batch_size,
    }
    condition_sha = _json_sha256(condition_binding)
    if path.exists():
        checkpoint, _ = _load_json_bytes(path, f"analysis checkpoint {order_name}/{batch_size}")
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise FailClosed(f"analysis checkpoint schema drifted: {path}")
        if checkpoint.get("condition_binding") != condition_binding:
            raise FailClosed(f"analysis checkpoint binding drifted: {path}")
        if checkpoint.get("condition_binding_sha256") != condition_sha:
            raise FailClosed(f"analysis checkpoint binding digest drifted: {path}")
        unsigned = dict(checkpoint)
        claimed = unsigned.pop("checkpoint_content_sha256", None)
        if claimed != _json_sha256(unsigned):
            raise FailClosed(f"analysis checkpoint content digest drifted: {path}")
        logical = {
            "condition_binding": checkpoint.get("condition_binding"),
            "branches": checkpoint.get("branches"),
            "batch_one_negative_control_exact": checkpoint.get("batch_one_negative_control_exact"),
            "normalized_action_effect": checkpoint.get("normalized_action_effect"),
        }
        if checkpoint.get("analysis_logical_sha256") != _json_sha256(logical):
            raise FailClosed(f"analysis checkpoint logical digest drifted: {path}")
        return checkpoint

    _append_event(
        events_path,
        frozen.binding_sha256,
        "ANALYSIS_BEGIN",
        {"order": order_name, "batch_size": batch_size},
    )
    branches: dict[str, dict[str, Any]] = {}
    for branch in ("old", "fixed"):
        state_stats = replay_reducer(state_values, batch_size, branch)
        action_stats = replay_reducer(action_values, batch_size, branch)
        branches[branch] = {
            "state": {
                "statistics": _stats_to_json(state_stats),
                "oracle_error": _stats_error(state_stats, state_oracle),
            },
            "actions": {
                "statistics": _stats_to_json(action_stats),
                "oracle_error": _stats_error(action_stats, action_oracle),
            },
            "aggregation_contract_accepted": bool(
                state_stats["count"] == EXPECTED_FRAMES
                and action_stats["count"] == EXPECTED_FRAMES
            ),
        }
    old_action = _stats_from_json(branches["old"]["actions"]["statistics"])
    fixed_action = _stats_from_json(branches["fixed"]["actions"]["statistics"])
    thresholds = frozen.section["design"]["material_normalized_action_effect_threshold"]
    effect = _normalized_action_effect(action_values, old_action, fixed_action, thresholds)
    old_state = _stats_from_json(branches["old"]["state"]["statistics"])
    fixed_state = _stats_from_json(branches["fixed"]["state"]["statistics"])
    logical = {
        "condition_binding": condition_binding,
        "branches": branches,
        "batch_one_negative_control_exact": (
            bool(_branch_equal(old_state, fixed_state) and _branch_equal(old_action, fixed_action))
            if batch_size == 1
            else None
        ),
        "normalized_action_effect": effect,
    }
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "condition_binding_sha256": condition_sha,
        "completed_utc": _utc_now(),
        **logical,
        "analysis_logical_sha256": _json_sha256(logical),
    }
    checkpoint["checkpoint_content_sha256"] = _json_sha256(checkpoint)
    _write_immutable_json(path, checkpoint)
    checkpoint_file_sha256 = _file_sha256(path)
    _append_event(
        events_path,
        frozen.binding_sha256,
        "ANALYSIS_COMPLETE",
        {
            "order": order_name,
            "batch_size": batch_size,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "analysis_logical_sha256": checkpoint["analysis_logical_sha256"],
        },
    )
    return checkpoint


def _oracle_to_json(oracle: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": int(oracle["count"]),
        "mean": _array_list(oracle["mean"]),
        "std": _array_list(oracle["std"]),
        "q01": _array_list(oracle["q01"]),
        "q99": _array_list(oracle["q99"]),
    }


def _official_stats_crosscheck(metadata: dict[str, bytes], state_oracle: dict[str, Any], action_oracle: dict[str, Any]) -> dict[str, Any]:
    try:
        stats = json.loads(metadata["meta/stats.json"])
    except (KeyError, json.JSONDecodeError) as error:
        raise FailClosed(f"official meta/stats.json is invalid: {error}") from error
    result: dict[str, Any] = {}
    for label, key, oracle in (
        ("state", "observation.state", state_oracle),
        ("actions", "action", action_oracle),
    ):
        try:
            official_mean = np.asarray(stats[key]["mean"], dtype=np.float64)
            official_std = np.asarray(stats[key]["std"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise FailClosed(f"official stats entry is invalid for {key}: {error}") from error
        if official_mean.shape != (EXPECTED_DIM,) or official_std.shape != (EXPECTED_DIM,):
            raise FailClosed(f"official stats shape is invalid for {key}")
        if not np.all(np.isfinite(official_mean)) or not np.all(np.isfinite(official_std)):
            raise FailClosed(f"official stats contain non-finite values for {key}")
        if np.any(official_std < 0):
            raise FailClosed(f"official stats contain negative standard deviations for {key}")
        mean_difference = np.abs(official_mean - oracle["mean"])
        std_difference = np.abs(official_std - oracle["std"])
        result[label] = {
            "source_key": key,
            "max_absolute_mean_difference": float(np.max(mean_difference)),
            "max_absolute_std_difference": float(np.max(std_difference)),
            "per_component_absolute_mean_difference": _array_list(mean_difference),
            "per_component_absolute_std_difference": _array_list(std_difference),
            "mean_std_within_1e_5": bool(
                np.all(mean_difference <= MEAN_STD_TOLERANCE)
                and np.all(std_difference <= MEAN_STD_TOLERANCE)
            ),
            "official_mean": _array_list(official_mean),
            "official_std": _array_list(official_std),
        }
    result["all_mean_std_within_1e_5"] = bool(
        all(result[label]["mean_std_within_1e_5"] for label in ("state", "actions"))
    )
    return result


def _evaluate_conditions(
    frozen: FrozenInputs,
    conditions: list[dict[str, Any]],
    completed_episode_count: int,
) -> dict[str, Any]:
    expected_matrix = {(order, batch) for order in ORDER_NAMES for batch in BATCH_SIZES}
    observed_matrix = [(value.get("order_name"), value.get("batch_size")) for value in conditions]
    matrix_exact = len(observed_matrix) == len(expected_matrix) and set(observed_matrix) == expected_matrix
    no_duplicates = len(observed_matrix) == len(set(observed_matrix))
    old_undercoverage = True
    fixed_full_coverage = True
    fixed_mean_std = True
    batch_one_exact = True
    contracts_correct = True
    exact_replay_counts = matrix_exact and no_duplicates
    for condition in conditions:
        order_name = condition.get("order_name")
        batch_size = condition.get("batch_size")
        if (order_name, batch_size) not in expected_matrix:
            exact_replay_counts = False
            continue
        expected = frozen.expected_counts[str(batch_size)]
        old = condition["branches"]["old"]
        fixed = condition["branches"]["fixed"]
        old_counts = [old[name]["statistics"]["count"] for name in ("state", "actions")]
        fixed_counts = [fixed[name]["statistics"]["count"] for name in ("state", "actions")]
        old_batches = [old[name]["statistics"]["batch_count"] for name in ("state", "actions")]
        fixed_batches = [fixed[name]["statistics"]["batch_count"] for name in ("state", "actions")]
        exact_replay_counts &= all(
            value == expected["old_processed_vectors"] for value in old_counts
        )
        exact_replay_counts &= all(
            value == expected["fixed_processed_vectors"] for value in fixed_counts
        )
        exact_replay_counts &= all(
            value == expected["batch_count_including_partial_tail"]
            for value in (*old_batches, *fixed_batches)
        )
        if batch_size > 1:
            old_undercoverage &= all(value < EXPECTED_FRAMES for value in old_counts)
            contracts_correct &= not old["aggregation_contract_accepted"]
            batch_one_exact &= condition.get("batch_one_negative_control_exact") is None
        else:
            batch_one_exact &= condition.get("batch_one_negative_control_exact") is True
            contracts_correct &= old["aggregation_contract_accepted"]
        fixed_full_coverage &= all(value == EXPECTED_FRAMES for value in fixed_counts)
        fixed_mean_std &= all(
            fixed[name]["oracle_error"]["mean_std_within_frozen_tolerance"]
            for name in ("state", "actions")
        )
        contracts_correct &= fixed["aggregation_contract_accepted"]

    correctness = {
        "expected_episode_count": EXPECTED_EPISODES,
        "completed_episode_count": completed_episode_count,
        "expected_frame_count": EXPECTED_FRAMES,
        "all_projection_checkpoints_complete": completed_episode_count == EXPECTED_EPISODES,
        "condition_matrix_exact_11_orders_x_4_batch_sizes": bool(matrix_exact and no_duplicates),
        "exact_frozen_replay_counts_for_every_condition": bool(exact_replay_counts),
        "old_undercoverage_for_every_order_at_batch_8_32_128": bool(old_undercoverage),
        "fixed_full_coverage_for_every_order_and_batch": bool(fixed_full_coverage),
        "fixed_mean_std_within_1e_5_for_every_order_and_batch": bool(fixed_mean_std),
        "batch_one_old_fixed_exact_for_every_order": bool(batch_one_exact),
        "aggregation_contract_decisions_correct": bool(contracts_correct),
    }
    correctness["reproduced"] = all(
        correctness[key]
        for key in (
            "all_projection_checkpoints_complete",
            "condition_matrix_exact_11_orders_x_4_batch_sizes",
            "exact_frozen_replay_counts_for_every_condition",
            "old_undercoverage_for_every_order_at_batch_8_32_128",
            "fixed_full_coverage_for_every_order_and_batch",
            "fixed_mean_std_within_1e_5_for_every_order_and_batch",
            "batch_one_old_fixed_exact_for_every_order",
            "aggregation_contract_decisions_correct",
        )
    )
    return correctness


def _sensitivity_summary(conditions: list[dict[str, Any]]) -> dict[str, Any]:
    statistics = ("mean", "std", "q01", "q99")

    def summarize(group: list[dict[str, Any]]) -> dict[str, float]:
        summary: dict[str, float] = {}
        for branch in ("old", "fixed"):
            for feature in ("state", "actions"):
                for statistic in statistics:
                    vectors = np.asarray(
                        [
                            condition["branches"][branch][feature]["statistics"][statistic]
                            for condition in group
                        ],
                        dtype=np.float64,
                    )
                    component_span = np.max(vectors, axis=0) - np.min(vectors, axis=0)
                    summary[f"{branch}.{feature}.{statistic}.max_component_span"] = float(
                        np.max(component_span)
                    )
        return summary

    by_batch = {
        str(batch): summarize([value for value in conditions if value["batch_size"] == batch])
        for batch in BATCH_SIZES
    }
    by_order = {
        order: summarize([value for value in conditions if value["order_name"] == order])
        for order in ORDER_NAMES
    }
    return {
        "definition": (
            "For each statistic vector, componentwise max-minus-min across the varied axis; "
            "the reported scalar is the maximum component span."
        ),
        "order_sensitivity_by_fixed_batch_size": by_batch,
        "batch_sensitivity_by_fixed_order": by_order,
        "maximum_order_sensitivity": max(max(value.values()) for value in by_batch.values()),
        "maximum_batch_sensitivity": max(max(value.values()) for value in by_order.values()),
    }


def _event_records(path: Path, binding_sha256: str) -> list[dict[str, Any]]:
    _verify_event_log(path, binding_sha256)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _event_payload_exists(
    path: Path,
    binding_sha256: str,
    names: set[str],
    payload: dict[str, Any],
) -> bool:
    matches = [
        event
        for event in _event_records(path, binding_sha256)
        if event.get("event") in names
    ]
    for event in matches:
        existing = event.get("payload", {})
        if existing.get("result_sha256") != payload.get("result_sha256"):
            raise FailClosed("RD8 event ledger contains a conflicting result terminal")
    return any(
        all(event.get("payload", {}).get(key) == value for key, value in payload.items())
        for event in matches
    )


def _bind_root_in_state(state: dict[str, Any], key: str, value: str) -> None:
    if state.get(key) not in (None, value):
        raise FailClosed(f"RD8 {key} drifted across resume")
    state[key] = value


def _reject_unequal_existing_artifact(path: Path, expected: bytes, label: str) -> None:
    if path.exists() and path.read_bytes() != expected:
        raise FailClosed(f"existing {label} does not equal the artifact rebuilt from verified checkpoints")


def _execute_locked(args: argparse.Namespace) -> dict[str, Any]:
    frozen = load_frozen_inputs(args.manifest, args.freeze_amendment)
    execution = _execution_binding()
    work_dir: Path = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "state.json"
    events_path = work_dir / "events.jsonl"
    state = _load_state(state_path, frozen, execution)
    _verify_event_log(events_path, frozen.binding_sha256)
    was_complete = state.get("status") == "COMPLETE"
    output_preexisted = args.output.exists()
    if was_complete and not output_preexisted:
        raise FailClosed("COMPLETE state points to a missing RD8 result")
    if not was_complete:
        event_count, _ = _verify_event_log(events_path, frozen.binding_sha256)
        _append_event(
            events_path,
            frozen.binding_sha256,
            "RUN_RESUME" if event_count else "RUN_START",
            {
                "download_authorized": bool(args.download),
                "endpoint": HF_ENDPOINT,
                "authentication": "disabled",
            },
        )
        state["status"] = "RUNNING"
        state["failure"] = None
        _atomic_write_json(state_path, state)

    metadata = _ensure_official_metadata(frozen, work_dir, bool(args.download))
    episode_arrays: list[tuple[np.ndarray, ...]] = []
    projection_inventory: list[dict[str, Any]] = []
    logical_projection_inventory: list[dict[str, Any]] = []
    projection_checkpoint_inventory: list[dict[str, Any]] = []
    for episode, record in enumerate(frozen.files):
        if _episode_from_path(record["path"]) != episode:
            raise FailClosed(f"source path/episode mismatch at {episode}")
        arrays, checkpoint = _complete_projection(
            frozen, work_dir, episode, record, bool(args.download), events_path
        )
        episode_arrays.append(arrays)
        checkpoint_path = _projection_checkpoint_path(work_dir, episode)
        logical_record = _logical_projection_record(episode, record, arrays)
        if checkpoint["logical_projection"] != logical_record:
            raise FailClosed(f"episode {episode} checkpoint/logical projection mismatch")
        checkpoint_file_sha256 = _file_sha256(checkpoint_path)
        logical_projection_inventory.append(logical_record)
        projection_checkpoint_inventory.append(
            {
                "episode": episode,
                "checkpoint_file_sha256": checkpoint_file_sha256,
                "checkpoint_content_sha256": checkpoint["checkpoint_content_sha256"],
            }
        )
        projection_inventory.append(
            {
                "episode": episode,
                "source_path": record["path"],
                "source_lfs_sha256": record["lfs_sha256"],
                "projection_file": str(_projection_path(work_dir, episode)),
                "projection_file_sha256": checkpoint["projection_file_sha256"],
                "projected_data_sha256": checkpoint["projected_data_sha256"],
                "logical_projection": logical_record,
                "checkpoint": str(checkpoint_path),
                "checkpoint_file_sha256": checkpoint_file_sha256,
                "checkpoint_content_sha256": checkpoint["checkpoint_content_sha256"],
            }
        )
    projection_root = _json_sha256(logical_projection_inventory)
    projection_checkpoint_root = _json_sha256(projection_checkpoint_inventory)
    _bind_root_in_state(state, "projection_root_sha256", projection_root)
    _bind_root_in_state(
        state, "projection_checkpoint_root_sha256", projection_checkpoint_root
    )
    _atomic_write_json(state_path, state)

    natural_state = np.ascontiguousarray(np.concatenate([value[0] for value in episode_arrays], axis=0))
    natural_actions = np.ascontiguousarray(np.concatenate([value[1] for value in episode_arrays], axis=0))
    if natural_state.shape != (EXPECTED_FRAMES, EXPECTED_DIM) or natural_actions.shape != (EXPECTED_FRAMES, EXPECTED_DIM):
        raise FailClosed("RD8 concatenated population is not [36900,14]")
    state_oracle = independent_oracle(natural_state)
    action_oracle = independent_oracle(natural_actions)
    official_crosscheck = _official_stats_crosscheck(metadata, state_oracle, action_oracle)

    conditions: list[dict[str, Any]] = []
    analysis_inventory: list[dict[str, Any]] = []
    for order_name in ORDER_NAMES:
        order = frozen.orders[order_name]
        state_values = np.ascontiguousarray(np.concatenate([episode_arrays[index][0] for index in order], axis=0))
        action_values = np.ascontiguousarray(np.concatenate([episode_arrays[index][1] for index in order], axis=0))
        if independent_oracle(state_values)["count"] != EXPECTED_FRAMES:
            raise FailClosed(f"RD8 order population drifted: {order_name}")
        for batch_size in BATCH_SIZES:
            checkpoint = _condition_checkpoint(
                frozen,
                work_dir,
                order_name,
                batch_size,
                projection_root,
                state_values,
                action_values,
                state_oracle,
                action_oracle,
                events_path,
            )
            conditions.append(
                {
                    "order_name": order_name,
                    "batch_size": batch_size,
                    "branches": checkpoint["branches"],
                    "batch_one_negative_control_exact": checkpoint["batch_one_negative_control_exact"],
                    "normalized_action_effect": checkpoint["normalized_action_effect"],
                }
            )
            path = _analysis_checkpoint_path(work_dir, order_name, batch_size)
            analysis_inventory.append(
                {
                    "order_name": order_name,
                    "batch_size": batch_size,
                    "checkpoint": str(path),
                    "checkpoint_file_sha256": _file_sha256(path),
                    "checkpoint_content_sha256": checkpoint["checkpoint_content_sha256"],
                    "condition_binding_sha256": checkpoint["condition_binding_sha256"],
                    "analysis_logical_sha256": checkpoint["analysis_logical_sha256"],
                }
            )

    analysis_logical_inventory = [
        {
            "order_name": item["order_name"],
            "batch_size": item["batch_size"],
            "analysis_logical_sha256": item["analysis_logical_sha256"],
        }
        for item in analysis_inventory
    ]
    analysis_checkpoint_inventory = [
        {
            "order_name": item["order_name"],
            "batch_size": item["batch_size"],
            "checkpoint_file_sha256": item["checkpoint_file_sha256"],
            "checkpoint_content_sha256": item["checkpoint_content_sha256"],
        }
        for item in analysis_inventory
    ]
    analysis_root = _json_sha256(analysis_logical_inventory)
    analysis_checkpoint_root = _json_sha256(analysis_checkpoint_inventory)
    _bind_root_in_state(state, "analysis_root_sha256", analysis_root)
    _bind_root_in_state(
        state, "analysis_checkpoint_root_sha256", analysis_checkpoint_root
    )
    _atomic_write_json(state_path, state)

    primary = next(
        value for value in conditions
        if value["order_name"] == "natural" and value["batch_size"] == PRIMARY_BATCH_SIZE
    )
    correctness = _evaluate_conditions(frozen, conditions, len(projection_inventory))
    correctness["primary_normalized_action_effect_material"] = primary[
        "normalized_action_effect"
    ]["material_under_frozen_threshold"]
    correctness["official_meta_stats_match_full_projection_within_1e_5"] = official_crosscheck[
        "all_mean_std_within_1e_5"
    ]
    sensitivity = _sensitivity_summary(conditions)
    result = {
        "schema_version": SCHEMA_VERSION,
        "case_id": CASE_ID,
        "case": (
            "OpenPI issue #570 / PR #619 source-exact reducer/RunningStats operators "
            "on a frozen raw projection"
        ),
        "evidence_class": "E2",
        "evidence_class_detail": (
            "source-exact historical reducer/RunningStats operators on all pinned public raw "
            "low-dimensional rows under the frozen include-tail E2 protocol; not the complete "
            "historical image-bearing transformed DataLoader"
        ),
        "created_utc": state["run_created_utc"],
        "status": "REPRODUCED" if correctness["reproduced"] else "NOT_REPRODUCED",
        "frozen_inputs": {
            "manifest": str(frozen.manifest_path),
            "manifest_sha256": frozen.manifest_sha256,
            "freeze_amendment": str(frozen.amendment_path),
            "freeze_amendment_sha256": frozen.amendment_sha256,
            "binding": frozen.binding,
            "binding_sha256": frozen.binding_sha256,
            "dataset": frozen.section["dataset"],
            "design": frozen.section["design"],
            "complete_orders": {name: list(frozen.orders[name]) for name in ORDER_NAMES},
            "no_replacement_after_observed_outcome": True,
        },
        "execution": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": execution["script_sha256"],
            "execution_binding": execution,
            "endpoint": HF_ENDPOINT,
            "authentication": "disabled",
            "full_source_parquets_locally_rehashed": False,
            "projection": "exact-revision Parquet range reads of five frozen columns; no image decode",
            "historical_old": "values=batch[key][0]; update(values.reshape(-1, values.shape[-1]))",
            "historical_fixed": "update(batch[key]); RunningStats flattens all leading dimensions",
        },
        "projection": {
            "root_sha256": projection_root,
            "root_definition": (
                "canonical JSON over episode/source path/source LFS SHA and per-array "
                "name/dtype/shape/logical SHA; excludes paths, timestamps, and NPZ/checkpoint bytes"
            ),
            "checkpoint_root_sha256": projection_checkpoint_root,
            "rows": EXPECTED_FRAMES,
            "episodes": EXPECTED_EPISODES,
            "columns": list(PROJECTION_COLUMNS),
            "feature_names": list(frozen.feature_names),
            "logical_inventory": logical_projection_inventory,
            "inventory": projection_inventory,
        },
        "oracle": {
            "method": "full-array float64 two-pass mean/population-std and NumPy q01/q99",
            "official_crosscheck_role": (
                "auxiliary consistency signal reported in correctness; the independent full-projection "
                "oracle remains the frozen formal gate"
            ),
            "state": _oracle_to_json(state_oracle),
            "actions": _oracle_to_json(action_oracle),
            "official_meta_stats_crosscheck": official_crosscheck,
        },
        "conditions": conditions,
        "sensitivity": sensitivity,
        "primary_condition": {
            "order_name": "natural",
            "batch_size": PRIMARY_BATCH_SIZE,
            "coverage_scope": (
                "raw one-state/one-action rows under the frozen include-partial-tail E2 protocol"
            ),
            "old_state_row_coverage": primary["branches"]["old"]["state"]["statistics"]["count"] / EXPECTED_FRAMES,
            "old_action_row_coverage": primary["branches"]["old"]["actions"]["statistics"]["count"] / EXPECTED_FRAMES,
            "fixed_state_row_coverage": primary["branches"]["fixed"]["state"]["statistics"]["count"] / EXPECTED_FRAMES,
            "fixed_action_row_coverage": primary["branches"]["fixed"]["actions"]["statistics"]["count"] / EXPECTED_FRAMES,
            "normalized_action_effect": primary["normalized_action_effect"],
        },
        "analysis_root_sha256": analysis_root,
        "analysis_checkpoint_root_sha256": analysis_checkpoint_root,
        "analysis_checkpoint_inventory": analysis_inventory,
        "correctness": correctness,
        "limitations": [
            "E2 replays source-exact reducer/RunningStats operators, not the complete historical OpenPI DataLoader.",
            "The frozen include-partial-tail replay policy is an experimental policy and does not claim the historical DataLoader's exact batch/drop behavior.",
            "The raw 14-component projection has one action per row; it does not construct the historical 50-step action horizon or apply ALOHA/delta/model transforms.",
            "Immutable revision and LFS metadata bind each 72.1 GB image-bearing source Parquet; range reads do not locally rehash every complete source file.",
            "Quantile errors are descriptive because the historical 5,000-bin histogram is approximate and order-sensitive.",
            "Normalized-action effects are descriptive raw low-dimensional z-score effects; coverage failure establishes the governance defect independently.",
        ],
        "sources": {
            "issue": "https://github.com/Physical-Intelligence/openpi/issues/570",
            "fix_pull_request": "https://github.com/Physical-Intelligence/openpi/pull/619",
            "test_pull_request": "https://github.com/Physical-Intelligence/openpi/pull/623",
            "dataset": f"https://huggingface.co/datasets/{EXPECTED_REPO_ID}/tree/{EXPECTED_REVISION}",
        },
    }
    result_bytes = _canonical_json_bytes(result) + b"\n"
    result_sha = _bytes_sha256(result_bytes)
    _reject_unequal_existing_artifact(args.output, result_bytes, "RD8 result")
    prepared = {
        "result": str(args.output),
        "result_sha256": result_sha,
        "binding_sha256": frozen.binding_sha256,
        "execution_binding_sha256": _json_sha256(execution),
        "projection_root_sha256": projection_root,
        "projection_checkpoint_root_sha256": projection_checkpoint_root,
        "analysis_root_sha256": analysis_root,
        "analysis_checkpoint_root_sha256": analysis_checkpoint_root,
        "condition_count": len(conditions),
    }
    if state.get("result_prepared") not in (None, prepared):
        raise FailClosed("RD8 prepared-result anchor drifted across resume")
    if not _event_payload_exists(
        events_path,
        frozen.binding_sha256,
        {"RESULT_PREPARED"},
        prepared,
    ):
        _append_event(
            events_path,
            frozen.binding_sha256,
            "RESULT_PREPARED",
            prepared,
        )
    state["result_prepared"] = prepared
    _atomic_write_json(state_path, state)
    published_sha = _write_immutable_bytes(args.output, result_bytes)
    if published_sha != result_sha:
        raise FailClosed("published RD8 result hash differs from prepared result")
    state["status"] = "COMPLETE"
    state["result_path"] = str(args.output)
    state["result_sha256"] = result_sha
    _atomic_write_json(state_path, state)
    terminal_payload = {
        "result": str(args.output),
        "result_sha256": result_sha,
        "status": result["status"],
    }
    if not _event_payload_exists(
        events_path,
        frozen.binding_sha256,
        {"RUN_COMPLETE", "RUN_COMPLETE_RECOVERED"},
        terminal_payload,
    ):
        _append_event(
            events_path,
            frozen.binding_sha256,
            "RUN_COMPLETE_RECOVERED" if (was_complete or output_preexisted) else "RUN_COMPLETE",
            terminal_payload,
        )
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    with _exclusive_lock(args.work_dir):
        try:
            return _execute_locked(args)
        except Exception as error:
            try:
                frozen = load_frozen_inputs(args.manifest, args.freeze_amendment)
                execution = _execution_binding()
                state_path = args.work_dir / "state.json"
                state = _load_state(state_path, frozen, execution)
                state["status"] = "FAILED"
                state["failure"] = {
                    "utc": _utc_now(),
                    "class": type(error).__name__,
                    "message": str(error),
                }
                _atomic_write_json(state_path, state)
                _append_event(args.work_dir / "events.jsonl", frozen.binding_sha256, "RUN_FAILED", state["failure"])
            except Exception as recording_error:
                print(f"RD8 failure evidence could not be recorded: {recording_error}", file=sys.stderr)
            raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--freeze-amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="authorize anonymous exact-revision HF range reads; absent means offline-only resume",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as error:
        print(f"RD8 failed closed: {error}", file=sys.stderr)
        return 2
    summary = {
        "status": result["status"],
        "reproduced": result["correctness"]["reproduced"],
        "episodes": result["correctness"]["completed_episode_count"],
        "primary_old_action_row_coverage": result["primary_condition"]["old_action_row_coverage"],
        "primary_material_effect": result["primary_condition"]["normalized_action_effect"]["material_under_frozen_threshold"],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if result["correctness"]["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
