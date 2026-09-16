#!/usr/bin/env python3
"""Replay the GR00T #172 / PR #373 timestamp-selection defect.

This is deliberately an E2 replay: it reproduces the two historical
``torchvision.io.VideoReader`` loops, but it does not import or claim to run
the complete historical GR00T stack.  A separate full-file PyAV decode is the
PTS-and-RGB oracle.

The formal input set comes only from ``contracts/real-defect-corpus-manifest.json``.
Downloads are opt-in, exact-revision Hugging Face downloads.  Hugging Face's
local-dir cache retains ``.incomplete`` files, so invoking the same command
again resumes an interrupted transfer.  Completed videos are represented by
immutable per-video checkpoints and are not replayed on resume.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import statistics
import sys
import tempfile
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "contracts" / "real-defect-corpus-manifest.json"
DEFAULT_FREEZE_AMENDMENT = (
    ROOT / "contracts" / "real-defect-corpus-freeze-amendment-v1.json"
)
DEFAULT_WORK_DIR = ROOT / "runs" / "p13-real-defect-groot-172"
DEFAULT_OUTPUT = ROOT / "results" / "real-defect-groot-172.json"

CASE_ID = "RD7-GR00T-172-373"
SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
ZERO_HASH = "0" * 64

EXPECTED_SOURCE = {
    "repo": "NVIDIA/Isaac-GR00T",
    "issue": 172,
    "pull_request": 373,
    "parent_commit": "d69b446ef33766fca481ff0ada48292706af5065",
    "fix_commit": "cdf6d6e3c48a9c033e2b188c8c3ff4cccc5ad04e",
    "merge_commit": "c8647d1bfe873b068e25e8ecdb9d33b935f063d1",
    "path": "gr00t/utils/video.py",
    "parent_blob_sha256": "2b119dcfeb164f9f840b8fe2483dbf177b989dc7e0ac921e7f324a3be155a3d2",
    "fixed_blob_sha256": "5fc5cfbbf3e6c0c3860333d285a1a08af35d232e35cf679e989e43124bb3f106",
}
EXPECTED_DATASET_REPO = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
EXPECTED_DATASET_REVISION = "ea7ac0b68f87da62f1e726771bba0fe74300802f"
EXPECTED_REGULAR_SECONDS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
EXPECTED_SPARSE_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
EXPECTED_LATE_FRACTIONS = [0.8, 0.85, 0.9, 0.95, 0.99]
EXPECTED_HF_ENDPOINT = "https://huggingface.co"
EXPECTED_CANONICAL_MANIFEST_SHA256 = (
    "7255524dc1776ef3e96e12fceea138ac76ae3bb15c55df42dc7e5a9e402e0d17"
)
EXPECTED_FREEZE_AMENDMENT_SHA256 = (
    "58ad7063ddd44872d8fa1535e585f39079537d0a0715967f4784adf6d2adbd0c"
)
EXPECTED_CANDIDATE_SIDECAR_COMPRESSED_SHA256 = (
    "f7bbec200ec62622cf8355f2b4f1bf151f710c14425dc6ade466fee7e75891ba"
)
EXPECTED_TIME_SEMANTICS = {
    "cross_group_duplicate_targets": "retain and count separately by query-group ID",
    "duplicate_pts_equivalence": "set of RGB SHA-256 digests at selected PTS",
    "duration_span": "last decoded PTS seconds - first decoded PTS seconds",
    "fraction_target": "first_pts + fraction * duration_span",
    "pts_seconds": "float(frame.pts * stream.time_base)",
    "target_after_last": "select last decoded frame",
    "target_before_first": "select first decoded frame",
}


class FailClosed(RuntimeError):
    """An input or recovery invariant failed and execution must stop."""


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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_json(path: Path, value: Any) -> str:
    """Publish a complete JSON file without ever replacing an existing one."""

    payload = _canonical_json_bytes(value) + b"\n"
    payload_sha256 = _bytes_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FailClosed(f"immutable checkpoint differs: {path}")
        return payload_sha256

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link supplies atomic create-if-absent
            # semantics.  Unlike replace(), it cannot overwrite evidence.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FailClosed(f"immutable checkpoint race differs: {path}")
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload_sha256


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise FailClosed("selected video path is missing")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise FailClosed(f"unsafe selected video path: {value!r}")
    normalized = parsed.as_posix()
    if normalized != value or not normalized.endswith(".mp4"):
        raise FailClosed(f"non-canonical selected video path: {value!r}")
    return normalized


@dataclass(frozen=True)
class FrozenManifest:
    path: Path
    sha256: str
    payload: dict[str, Any]
    source: dict[str, Any]
    dataset: dict[str, Any]
    queries: dict[str, Any]
    selected: tuple[dict[str, Any], ...]
    amendment_path: Path
    amendment_sha256: str
    amendment: dict[str, Any]
    candidate_sidecar: dict[str, Any]
    binding: dict[str, Any]
    binding_sha256: str


def _load_and_verify_freeze_amendment(
    amendment_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    selected: Sequence[dict[str, Any]],
    candidate_paths_sha256: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    try:
        raw = amendment_path.read_bytes()
        amendment = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(f"cannot read freeze amendment {amendment_path}: {error}") from error
    if amendment.get("amendment_id") != "P13-REALDEFECT-8-FREEZE-AMENDMENT-V1":
        raise FailClosed("unexpected P13 freeze amendment identity")
    amendment_sha256 = _bytes_sha256(raw)
    if amendment_sha256 != EXPECTED_FREEZE_AMENDMENT_SHA256:
        raise FailClosed("freeze amendment complete-file SHA-256 mismatch")
    if amendment.get("status") != "pre-outcome freeze strengthening; input membership unchanged":
        raise FailClosed("unexpected P13 freeze amendment status")
    if manifest_sha256 != EXPECTED_CANONICAL_MANIFEST_SHA256:
        raise FailClosed("canonical manifest SHA-256 mismatch")
    canonical = amendment.get("canonical_manifest")
    if canonical != {
        "corpus_id": "P13-REALDEFECT-8",
        "path": "contracts/real-defect-corpus-manifest.json",
        "sha256": manifest_sha256,
    }:
        raise FailClosed(
            f"freeze amendment does not bind the exact manifest bytes: {manifest_path}"
        )
    environment = amendment.get("environment")
    if environment != {
        "authentication": "disabled; public immutable artifacts only",
        "huggingface_hub_endpoint": EXPECTED_HF_ENDPOINT,
    }:
        raise FailClosed("freeze amendment download environment drifted")
    rd7_amendment = amendment.get("rd7")
    if not isinstance(rd7_amendment, dict):
        raise FailClosed("freeze amendment lacks RD7")
    if rd7_amendment.get("time_semantics") != EXPECTED_TIME_SEMANTICS:
        raise FailClosed("freeze amendment time semantics drifted")
    if rd7_amendment.get("decode_failure_rule") != (
        "retain and report the fixed input; never replace it"
    ):
        raise FailClosed("freeze amendment decode-failure rule drifted")
    if rd7_amendment.get("eligibility_scope") != (
        "every .mp4 path in the pinned dataset repository"
    ):
        raise FailClosed("freeze amendment eligibility scope drifted")
    expected_consistency = {
        "head_merge_blob_sha256": EXPECTED_SOURCE["fixed_blob_sha256"],
        "merge_parents": [
            EXPECTED_SOURCE["parent_commit"],
            EXPECTED_SOURCE["fix_commit"],
        ],
    }
    if rd7_amendment.get("source_consistency") != expected_consistency:
        raise FailClosed("freeze amendment source consistency drifted")

    sidecar = rd7_amendment.get("candidate_sidecar")
    if not isinstance(sidecar, dict):
        raise FailClosed("freeze amendment lacks the RD7 candidate sidecar")
    if sidecar.get("path") != "contracts/p13-groot-mp4-candidates.txt.gz":
        raise FailClosed("candidate-sidecar path drifted")
    if sidecar.get("compression") != "gzip level 9, mtime 0":
        raise FailClosed("candidate-sidecar compression binding drifted")
    if sidecar.get("candidate_count") != 50656:
        raise FailClosed("candidate-sidecar count binding drifted")
    if sidecar.get("uncompressed_sha256") != candidate_paths_sha256:
        raise FailClosed("candidate-sidecar digest differs from the canonical manifest")
    if not _is_lower_hex(sidecar.get("compressed_sha256"), 64):
        raise FailClosed("candidate-sidecar compressed digest is invalid")
    if sidecar["compressed_sha256"] != EXPECTED_CANDIDATE_SIDECAR_COMPRESSED_SHA256:
        raise FailClosed("candidate-sidecar frozen compressed digest drifted")

    sidecar_path = (ROOT / sidecar["path"]).resolve()
    try:
        sidecar_path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise FailClosed("candidate sidecar escapes the repository") from error
    try:
        compressed = sidecar_path.read_bytes()
        uncompressed = gzip.decompress(compressed)
        decoded = uncompressed.decode("utf-8")
    except (OSError, gzip.BadGzipFile, UnicodeDecodeError) as error:
        raise FailClosed(f"cannot verify candidate sidecar: {error}") from error
    if _bytes_sha256(compressed) != sidecar["compressed_sha256"]:
        raise FailClosed("candidate-sidecar compressed SHA-256 mismatch")
    if _bytes_sha256(uncompressed) != sidecar["uncompressed_sha256"]:
        raise FailClosed("candidate-sidecar uncompressed SHA-256 mismatch")
    candidate_paths = decoded.splitlines()
    if len(candidate_paths) != sidecar["candidate_count"]:
        raise FailClosed("candidate-sidecar decoded count mismatch")
    if candidate_paths != sorted(candidate_paths) or len(candidate_paths) != len(
        set(candidate_paths)
    ):
        raise FailClosed("candidate-sidecar paths are not unique lexical order")
    if any(_safe_relative_path(value) != value for value in candidate_paths):
        raise FailClosed("candidate-sidecar contains a noncanonical path")
    if candidate_paths[:10] != [record["path"] for record in selected]:
        raise FailClosed("manifest selection is not the sidecar's lexical first ten")
    return amendment, amendment_sha256, dict(sidecar)


def load_frozen_manifest(
    path: Path, amendment_path: Path = DEFAULT_FREEZE_AMENDMENT
) -> FrozenManifest:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(f"cannot read frozen manifest {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("corpus_id") != "P13-REALDEFECT-8":
        raise FailClosed("unexpected frozen corpus identity")
    try:
        section = payload["rd7_groot_172"]
        source = section["source"]
        dataset = section["dataset"]
        queries = section["queries"]
        selected = dataset["selected"]
    except (KeyError, TypeError) as error:
        raise FailClosed(f"malformed RD7 manifest section: {error}") from error

    if section.get("evidence_target") != "E2 exact historical loop replay on pinned public videos":
        raise FailClosed("RD7 evidence target drifted")
    for key, expected in EXPECTED_SOURCE.items():
        if source.get(key) != expected:
            raise FailClosed(f"RD7 source binding drifted at {key}")
    if dataset.get("repo_id") != EXPECTED_DATASET_REPO:
        raise FailClosed("RD7 dataset repository drifted")
    if dataset.get("revision") != EXPECTED_DATASET_REVISION:
        raise FailClosed("RD7 dataset revision drifted")
    if dataset.get("eligibility") != "all repository paths ending in .mp4":
        raise FailClosed("RD7 eligibility rule drifted")
    if dataset.get("selection") != "lexicographically first 10 eligible paths":
        raise FailClosed("RD7 sample selection rule drifted")
    if not isinstance(dataset.get("candidate_count"), int) or dataset["candidate_count"] < 10:
        raise FailClosed("RD7 candidate count is invalid")
    if not _is_lower_hex(dataset.get("candidate_paths_sha256"), 64):
        raise FailClosed("RD7 candidate-path digest is invalid")
    if not isinstance(selected, list) or len(selected) != 10:
        raise FailClosed("RD7 must contain exactly ten frozen videos")

    checked: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in selected:
        if not isinstance(record, dict):
            raise FailClosed("RD7 selected record is not an object")
        selected_path = _safe_relative_path(record.get("path"))
        if selected_path in seen_paths:
            raise FailClosed(f"duplicate frozen video: {selected_path}")
        seen_paths.add(selected_path)
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise FailClosed(f"invalid byte size for {selected_path}")
        if not _is_lower_hex(record.get("git_blob_oid"), 40):
            raise FailClosed(f"invalid Git blob for {selected_path}")
        if not _is_lower_hex(record.get("lfs_sha256"), 64):
            raise FailClosed(f"invalid LFS SHA-256 for {selected_path}")
        checked.append(dict(record))
    if [record["path"] for record in checked] != sorted(seen_paths):
        raise FailClosed("frozen videos are not in lexical order")

    if queries.get("reported_regular_seconds") != EXPECTED_REGULAR_SECONDS:
        raise FailClosed("RD7 regular query schedule drifted")
    if queries.get("sparse_duration_fractions") != EXPECTED_SPARSE_FRACTIONS:
        raise FailClosed("RD7 sparse query schedule drifted")
    if queries.get("late_duration_fractions") != EXPECTED_LATE_FRACTIONS:
        raise FailClosed("RD7 late query schedule drifted")
    if queries.get("negative_control") != "every decoded source PTS in original order":
        raise FailClosed("RD7 negative control drifted")
    if queries.get("oracle_selection") != "latest source PTS <= target; first frame if none":
        raise FailClosed("RD7 oracle rule drifted")
    if queries.get("duplicate_pts_equivalence") != "all frames at the selected PTS are equivalent":
        raise FailClosed("RD7 duplicate-PTS rule drifted")

    manifest_sha256 = _bytes_sha256(raw)
    amendment, amendment_sha256, candidate_sidecar = _load_and_verify_freeze_amendment(
        amendment_path,
        path,
        manifest_sha256,
        checked,
        dataset["candidate_paths_sha256"],
    )
    binding = {
        "case_id": CASE_ID,
        "manifest_sha256": manifest_sha256,
        "freeze_amendment_sha256": amendment_sha256,
        "candidate_sidecar_compressed_sha256": candidate_sidecar[
            "compressed_sha256"
        ],
        "candidate_sidecar_uncompressed_sha256": candidate_sidecar[
            "uncompressed_sha256"
        ],
        "source": source,
        "dataset_repo_id": dataset["repo_id"],
        "dataset_revision": dataset["revision"],
        "candidate_paths_sha256": dataset["candidate_paths_sha256"],
        "selected_sha256": _json_sha256(checked),
        "queries_sha256": _json_sha256(queries),
    }
    return FrozenManifest(
        path=path,
        sha256=manifest_sha256,
        payload=payload,
        source=dict(source),
        dataset=dict(dataset),
        queries=dict(queries),
        selected=tuple(checked),
        amendment_path=amendment_path,
        amendment_sha256=amendment_sha256,
        amendment=amendment,
        candidate_sidecar=candidate_sidecar,
        binding=binding,
        binding_sha256=_json_sha256(binding),
    )


def _safe_local_video_path(video_root: Path, relative_path: str) -> Path:
    relative_path = _safe_relative_path(relative_path)
    root = video_root.resolve()
    candidate = (video_root / PurePosixPath(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FailClosed(f"video path escapes root: {relative_path}") from error
    return candidate


def _verify_video_file(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        raise FailClosed(f"frozen video is missing: {record['path']}")
    observed_bytes = path.stat().st_size
    if observed_bytes != record["bytes"]:
        raise FailClosed(
            f"byte-size mismatch for {record['path']}: "
            f"expected {record['bytes']}, observed {observed_bytes}"
        )
    observed_sha256 = _file_sha256(path)
    if observed_sha256 != record["lfs_sha256"]:
        raise FailClosed(f"LFS SHA-256 mismatch for {record['path']}")
    return {
        "path": record["path"],
        "expected_bytes": record["bytes"],
        "observed_bytes": observed_bytes,
        "git_blob_oid": record["git_blob_oid"],
        "expected_lfs_sha256": record["lfs_sha256"],
        "observed_sha256": observed_sha256,
        "exact_match": True,
    }


def _ensure_video(
    frozen: FrozenManifest,
    record: dict[str, Any],
    video_root: Path,
    allow_download: bool,
) -> tuple[Path, dict[str, Any], bool]:
    path = _safe_local_video_path(video_root, record["path"])
    downloaded = False
    if not path.exists():
        if not allow_download:
            raise FailClosed(
                f"frozen video is absent and network download was not authorized: {record['path']}"
            )
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise FailClosed("--download requires huggingface_hub") from error
        video_root.mkdir(parents=True, exist_ok=True)
        try:
            returned = Path(
                hf_hub_download(
                    repo_id=frozen.dataset["repo_id"],
                    repo_type="dataset",
                    revision=frozen.dataset["revision"],
                    filename=record["path"],
                    local_dir=video_root,
                    force_download=False,
                    endpoint=EXPECTED_HF_ENDPOINT,
                    token=False,
                )
            )
        except Exception as error:
            raise FailClosed(f"download failed for {record['path']}: {error}") from error
        if returned.resolve() != path:
            raise FailClosed(
                f"download returned an unexpected path for {record['path']}: {returned}"
            )
        downloaded = True
    return path, _verify_video_file(path, record), downloaded


def _verify_event_log(path: Path, binding_sha256: str) -> tuple[int, str]:
    if not path.exists():
        return 0, ZERO_HASH
    expected_sequence = 1
    previous_hash = ZERO_HASH
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise FailClosed(f"torn event line {line_number}")
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise FailClosed(f"invalid event line {line_number}: {error}") from error
            if event.get("schema_version") != EVENT_SCHEMA_VERSION:
                raise FailClosed(f"event schema mismatch at line {line_number}")
            if event.get("sequence") != expected_sequence:
                raise FailClosed(f"event sequence mismatch at line {line_number}")
            if event.get("binding_sha256") != binding_sha256:
                raise FailClosed(f"event binding mismatch at line {line_number}")
            if event.get("previous_event_sha256") != previous_hash:
                raise FailClosed(f"event predecessor mismatch at line {line_number}")
            claimed_hash = event.get("event_sha256")
            unsigned = dict(event)
            unsigned.pop("event_sha256", None)
            observed_hash = _json_sha256(unsigned)
            if claimed_hash != observed_hash:
                raise FailClosed(f"event hash mismatch at line {line_number}")
            previous_hash = claimed_hash
            expected_sequence += 1
    return expected_sequence - 1, previous_hash


def _append_event(
    path: Path,
    binding_sha256: str,
    event_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    sequence, previous_hash = _verify_event_log(path, binding_sha256)
    event = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "sequence": sequence + 1,
        "previous_event_sha256": previous_hash,
        "binding_sha256": binding_sha256,
        "utc": _utc_now(),
        "event": event_type,
        "payload": payload,
    }
    event["event_sha256"] = _json_sha256(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        with os.fdopen(descriptor, "ab", closefd=True) as handle:
            handle.write(_canonical_json_bytes(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        # fdopen owns the descriptor unless constructing the file object fails.
        pass
    _fsync_directory(path.parent)
    return event


def _load_state(
    path: Path,
    frozen: FrozenManifest,
    execution_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "binding": frozen.binding,
            "binding_sha256": frozen.binding_sha256,
            "execution_binding": execution_binding,
            "run_created_utc": _utc_now(),
            "status": "INITIALIZED",
            "completed": {},
            "failure": None,
        }
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(f"invalid run state: {error}") from error
    if state.get("schema_version") != SCHEMA_VERSION:
        raise FailClosed("run-state schema mismatch")
    if state.get("binding") != frozen.binding:
        raise FailClosed("run-state binding differs from the frozen manifest")
    if state.get("binding_sha256") != frozen.binding_sha256:
        raise FailClosed("run-state binding digest mismatch")
    if execution_binding is not None and state.get("execution_binding") != execution_binding:
        raise FailClosed("run-state execution environment or replay backend drifted")
    if not isinstance(state.get("run_created_utc"), str):
        raise FailClosed("run-state creation time is missing")
    if not isinstance(state.get("completed"), dict):
        raise FailClosed("run-state completion index is invalid")
    return state


@dataclass(frozen=True)
class OracleFrame:
    source_index: int
    pts: int
    time_base_numerator: int
    time_base_denominator: int
    rgb_sha256: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def time(self) -> Fraction:
        return Fraction(
            self.pts * self.time_base_numerator, self.time_base_denominator
        )

    @property
    def seconds(self) -> float:
        return float(self.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "pts": self.pts,
            "time_base": f"{self.time_base_numerator}/{self.time_base_denominator}",
            "seconds": self.seconds,
            "rgb_sha256": self.rgb_sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class DecodedOracle:
    frames: tuple[OracleFrame, ...]
    duration_origin: Fraction
    duration: Fraction
    duration_rule: str
    stream_index: int
    stream_declared_frames: int | None
    oracle_sha256: str


def _rgb_array_details(value: Any, *, channel_first_allowed: bool) -> tuple[str, tuple[int, ...], str]:
    """Return RGB digest, HWC shape, and dtype without importing NumPy."""

    array = value
    if hasattr(array, "detach"):
        array = array.detach()
    if hasattr(array, "cpu"):
        array = array.cpu()
    shape = tuple(int(dimension) for dimension in getattr(array, "shape", ()))
    if channel_first_allowed and len(shape) == 3 and shape[0] == 3 and shape[-1] != 3:
        if hasattr(array, "permute"):
            array = array.permute(1, 2, 0)
        elif hasattr(array, "transpose"):
            array = array.transpose(1, 2, 0)
        shape = tuple(int(dimension) for dimension in getattr(array, "shape", ()))
    if hasattr(array, "contiguous"):
        array = array.contiguous()
    if hasattr(array, "numpy"):
        array = array.numpy()
    shape = tuple(int(dimension) for dimension in getattr(array, "shape", shape))
    dtype = str(getattr(array, "dtype", ""))
    if len(shape) != 3 or shape[-1] != 3:
        raise FailClosed(f"decoded frame is not HWC RGB: shape={shape}")
    if dtype not in {"uint8", "torch.uint8"}:
        raise FailClosed(f"decoded frame is not uint8: dtype={dtype}")
    try:
        raw = array.tobytes(order="C")
    except TypeError:
        raw = array.tobytes()
    except AttributeError as error:
        raise FailClosed("decoded frame cannot expose RGB bytes") from error
    return _bytes_sha256(raw), shape, "uint8"


def decode_oracle_with_pyav(path: Path, av_module: Any | None = None) -> DecodedOracle:
    """Fully decode one video and build an independent PTS+RGB oracle."""

    if av_module is None:
        try:
            import av as av_module  # type: ignore[no-redef]
        except ImportError as error:
            raise FailClosed("formal RD7 execution requires PyAV") from error
    try:
        container = av_module.open(str(path))
    except Exception as error:
        raise FailClosed(f"PyAV could not open {path}: {error}") from error

    frames: list[OracleFrame] = []
    try:
        video_streams = list(container.streams.video)
        if len(video_streams) != 1:
            raise FailClosed(
                f"expected exactly one video stream in {path}, found {len(video_streams)}"
            )
        stream = video_streams[0]
        default_time_base = getattr(stream, "time_base", None)
        if default_time_base is None:
            raise FailClosed("PyAV video stream has no time base")
        stream_time_base = Fraction(default_time_base)
        for source_index, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise FailClosed(f"source frame {source_index} has no PTS")
            frame_time_base = getattr(frame, "time_base", None)
            if frame_time_base is not None and Fraction(frame_time_base) != stream_time_base:
                raise FailClosed(
                    f"source frame {source_index} time base differs from its stream"
                )
            rgb = frame.to_ndarray(format="rgb24")
            rgb_sha256, shape, dtype = _rgb_array_details(
                rgb, channel_first_allowed=False
            )
            frames.append(
                OracleFrame(
                    source_index=source_index,
                    pts=int(frame.pts),
                    time_base_numerator=stream_time_base.numerator,
                    time_base_denominator=stream_time_base.denominator,
                    rgb_sha256=rgb_sha256,
                    shape=shape,
                    dtype=dtype,
                )
            )
        stream_index = int(getattr(stream, "index", 0))
        declared = getattr(stream, "frames", None)
        stream_declared_frames = int(declared) if declared not in (None, 0) else None
    finally:
        container.close()

    if len(frames) < 2:
        raise FailClosed(f"video has fewer than two decoded frames: {path}")
    for previous, current in zip(frames, frames[1:]):
        if current.time < previous.time:
            raise FailClosed(
                f"source PTS is not nondecreasing at frame {current.source_index}"
            )
    duration_origin = frames[0].time
    duration = frames[-1].time - duration_origin
    if duration <= 0:
        raise FailClosed(f"decoded PTS span is not positive: {path}")
    oracle_payload = [frame.as_dict() for frame in frames]
    return DecodedOracle(
        frames=tuple(frames),
        duration_origin=duration_origin,
        duration=duration,
        duration_rule="first_source_pts + fraction * (last_source_pts - first_source_pts)",
        stream_index=stream_index,
        stream_declared_frames=stream_declared_frames,
        oracle_sha256=_json_sha256(oracle_payload),
    )


@dataclass(frozen=True)
class ReplayFrame:
    loaded_index: int
    pts_seconds: float
    rgb_sha256: str
    shape: tuple[int, ...]
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "loaded_index": self.loaded_index,
            "pts_seconds": self.pts_seconds,
            "rgb_sha256": self.rgb_sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }


@dataclass(frozen=True)
class ReplayResult:
    loaded_count: int
    returned: tuple[ReplayFrame, ...]


def _close_video_reader(reader: Any) -> None:
    container = getattr(reader, "container", None)
    close = getattr(container, "close", None)
    if close is not None:
        close()


def _read_replay_frame(raw: Any, loaded_index: int) -> ReplayFrame:
    try:
        pts_seconds = float(raw["pts"])
        data = raw["data"]
    except (KeyError, TypeError, ValueError) as error:
        raise FailClosed(f"VideoReader returned a malformed frame: {error}") from error
    if not math.isfinite(pts_seconds):
        raise FailClosed("VideoReader returned a non-finite PTS")
    rgb_sha256, shape, dtype = _rgb_array_details(data, channel_first_allowed=True)
    return ReplayFrame(
        loaded_index=loaded_index,
        pts_seconds=pts_seconds,
        rgb_sha256=rgb_sha256,
        shape=shape,
        dtype=dtype,
    )


def replay_old_stop_on_count(reader: Any, targets: Sequence[Fraction]) -> ReplayResult:
    """Exact parent loop: seek, then stop on last target or request count."""

    if not targets:
        raise FailClosed("historical replay received no target timestamps")
    reader.seek(float(targets[0]), keyframes_only=True)
    loaded: list[ReplayFrame] = []
    try:
        for raw in reader:
            frame = _read_replay_frame(raw, len(loaded))
            loaded.append(frame)
            if frame.pts_seconds >= float(targets[-1]):
                break
            if len(loaded) >= len(targets):
                break
    finally:
        _close_video_reader(reader)
    # The historical parent returned the loaded frames in order, with no
    # timestamp-to-request selection pass.
    return ReplayResult(loaded_count=len(loaded), returned=tuple(loaded))


def replay_fixed_load_through_last(reader: Any, targets: Sequence[Fraction]) -> ReplayResult:
    """Exact PR #373 loop: load through last, then choose latest <= target."""

    if not targets:
        raise FailClosed("historical replay received no target timestamps")
    reader.seek(float(targets[0]), keyframes_only=True)
    loaded: list[ReplayFrame] = []
    try:
        for raw in reader:
            frame = _read_replay_frame(raw, len(loaded))
            loaded.append(frame)
            if frame.pts_seconds >= float(targets[-1]):
                break
    finally:
        _close_video_reader(reader)
    if not loaded:
        raise FailClosed("fixed replay loaded no frames")

    selected: list[ReplayFrame] = []
    for target in targets:
        eligible = [frame for frame in loaded if frame.pts_seconds <= float(target)]
        selected.append(eligible[-1] if eligible else loaded[0])
    return ReplayResult(loaded_count=len(loaded), returned=tuple(selected))


def _torchvision_reader(path: Path) -> Any:
    try:
        import torchvision
    except ImportError as error:
        raise FailClosed("formal RD7 replay requires torchvision") from error
    try:
        torchvision.set_video_backend("pyav")
        return torchvision.io.VideoReader(str(path), "video")
    except Exception as error:
        raise FailClosed(f"cannot create torchvision/PyAV VideoReader: {error}") from error


class PyAVHistoricalReader:
    """Small VideoReader-compatible adapter for the two historical loops.

    torchvision 0.26 no longer exposes ``io.VideoReader``.  This adapter is a
    declared E2 backend compatibility substitution: it retains the historical
    keyframe-backward seek and loop control flow, and exposes each decoded
    frame as ``{"pts": seconds, "data": RGB}``.  It is not presented as the
    complete historical GR00T or torchvision stack.
    """

    def __init__(self, path: Path, av_module: Any | None = None) -> None:
        if av_module is None:
            try:
                import av as av_module  # type: ignore[no-redef]
            except ImportError as error:
                raise FailClosed("PyAV historical-reader backend requires PyAV") from error
        try:
            self.container = av_module.open(str(path))
        except Exception as error:
            raise FailClosed(f"PyAV historical reader could not open {path}: {error}") from error
        streams = list(self.container.streams.video)
        if len(streams) != 1:
            self.container.close()
            raise FailClosed(
                f"historical reader expected one video stream, found {len(streams)}"
            )
        self.stream = streams[0]
        if getattr(self.stream, "time_base", None) is None:
            self.container.close()
            raise FailClosed("historical reader stream has no time base")
        self.time_base = Fraction(self.stream.time_base)

    def seek(self, seconds: float, keyframes_only: bool = True) -> "PyAVHistoricalReader":
        if not math.isfinite(seconds):
            raise FailClosed("historical reader received a non-finite seek target")
        target = Fraction(str(seconds))
        offset = math.floor(target / self.time_base)
        try:
            self.container.seek(
                offset,
                backward=True,
                any_frame=not keyframes_only,
                stream=self.stream,
            )
        except Exception as error:
            raise FailClosed(f"PyAV historical keyframe seek failed: {error}") from error
        return self

    def __iter__(self) -> Iterable[dict[str, Any]]:
        for frame in self.container.decode(self.stream):
            if frame.pts is None:
                raise FailClosed("historical reader decoded a frame without PTS")
            yield {
                # The freeze amendment deliberately binds stream.time_base,
                # matching the historical VideoReader seconds interface.
                "pts": float(int(frame.pts) * self.time_base),
                "data": frame.to_ndarray(format="rgb24"),
            }


def _reader_factory_for_backend(name: str) -> Callable[[Path], Any]:
    if name == "pyav_historical_compat":
        return PyAVHistoricalReader
    if name == "torchvision_videoreader":
        return _torchvision_reader
    raise FailClosed(f"unsupported replay backend: {name}")


def _fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "fraction": f"{value.numerator}/{value.denominator}",
        "seconds": float(value),
    }


def build_query_groups(
    frozen: FrozenManifest, oracle: DecodedOracle
) -> dict[str, tuple[Fraction, ...]]:
    regular = tuple(Fraction(str(value)) for value in frozen.queries["reported_regular_seconds"])
    sparse = tuple(
        oracle.duration_origin + oracle.duration * Fraction(str(value))
        for value in frozen.queries["sparse_duration_fractions"]
    )
    late = tuple(
        oracle.duration_origin + oracle.duration * Fraction(str(value))
        for value in frozen.queries["late_duration_fractions"]
    )
    continuous = tuple(frame.time for frame in oracle.frames)
    groups = {
        "regular": regular,
        "sparse": sparse,
        "late": late,
        "continuous_pts_negative_control": continuous,
    }
    for group_name, values in groups.items():
        if not values:
            raise FailClosed(f"empty query group: {group_name}")
        if any(current < previous for previous, current in zip(values, values[1:])):
            raise FailClosed(f"query group is not nondecreasing: {group_name}")
    return groups


def _oracle_selection(
    oracle: DecodedOracle, target: Fraction
) -> tuple[Fraction, tuple[OracleFrame, ...]]:
    eligible_times = [frame.time for frame in oracle.frames if frame.time <= target]
    selected_time = max(eligible_times) if eligible_times else oracle.frames[0].time
    equivalent = tuple(frame for frame in oracle.frames if frame.time == selected_time)
    if not equivalent:
        raise AssertionError("oracle selected a PTS absent from the full decode")
    return selected_time, equivalent


def _local_frame_interval(oracle: DecodedOracle, selected_time: Fraction) -> Fraction | None:
    unique_times = sorted({frame.time for frame in oracle.frames})
    if len(unique_times) < 2:
        return None
    position = unique_times.index(selected_time)
    intervals: list[Fraction] = []
    if position > 0:
        intervals.append(selected_time - unique_times[position - 1])
    if position + 1 < len(unique_times):
        intervals.append(unique_times[position + 1] - selected_time)
    positive = [value for value in intervals if value > 0]
    return min(positive) if positive else None


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
        "median_absolute_timestamp_error_seconds": statistics.median(errors) if errors else None,
        "p95_absolute_timestamp_error_seconds": _percentile(errors, 0.95),
        "max_absolute_timestamp_error_seconds": max(errors) if errors else None,
        "errors_exceeding_one_local_frame_interval": sum(
            bool(row["exceeds_one_local_frame_interval"]) for row in rows
        ),
    }


def evaluate_replay(
    oracle: DecodedOracle,
    group_name: str,
    targets: Sequence[Fraction],
    replay: ReplayResult,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    structural_shapes = {frame.shape for frame in oracle.frames}
    for index, target in enumerate(targets):
        selected_time, equivalent = _oracle_selection(oracle, target)
        interval = _local_frame_interval(oracle, selected_time)
        actual = replay.returned[index] if index < len(replay.returned) else None
        if actual is None:
            timestamp_error = None
            pts_match = False
            digest_match = False
            shape_match = False
            dtype_match = False
            exceeds = True
        else:
            timestamp_error = abs(actual.pts_seconds - float(selected_time))
            # VideoReader exposes seconds as a float.  One millionth of a
            # source time-base tick is ample for float conversion noise and
            # far below an adjacent presentation timestamp.
            tick = min(
                Fraction(frame.time_base_numerator, frame.time_base_denominator)
                for frame in equivalent
            )
            tolerance = max(1e-9, float(tick) * 1e-6)
            pts_match = timestamp_error <= tolerance
            digest_match = actual.rgb_sha256 in {
                frame.rgb_sha256 for frame in equivalent
            }
            shape_match = actual.shape in structural_shapes
            dtype_match = actual.dtype == "uint8"
            exceeds = interval is not None and timestamp_error > float(interval) + tolerance
        rows.append(
            {
                "group": group_name,
                "query_index": index,
                "target": _fraction_record(target),
                "oracle": {
                    "selected_pts": _fraction_record(selected_time),
                    "equivalent_source_indices": [frame.source_index for frame in equivalent],
                    "equivalent_rgb_sha256": [frame.rgb_sha256 for frame in equivalent],
                    "local_frame_interval_seconds": float(interval) if interval is not None else None,
                },
                "actual": actual.as_dict() if actual is not None else None,
                "pts_match": pts_match,
                "rgb_digest_match": digest_match,
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "correct": pts_match and digest_match,
                "wrong_frame": not (pts_match and digest_match),
                "absolute_timestamp_error_seconds": timestamp_error,
                "exceeds_one_local_frame_interval": exceeds,
            }
        )
    extra_returns = max(0, len(replay.returned) - len(targets))
    structure = {
        "readable": all(
            frame.dtype == "uint8" and len(frame.shape) == 3 and frame.shape[-1] == 3
            for frame in replay.returned
        ),
        "return_count": len(replay.returned),
        "expected_return_count": len(targets),
        "return_count_matches": len(replay.returned) == len(targets),
        "extra_return_count": extra_returns,
        "all_shapes_match_oracle": all(
            frame.shape in structural_shapes for frame in replay.returned
        ),
        "all_dtypes_uint8": all(frame.dtype == "uint8" for frame in replay.returned),
    }
    structure["accepted"] = all(
        [
            structure["readable"],
            structure["return_count_matches"],
            structure["all_shapes_match_oracle"],
            structure["all_dtypes_uint8"],
        ]
    )
    return {
        "loaded_count": replay.loaded_count,
        "structure_baseline": structure,
        "metrics": _metric_summary(rows),
        "queries": rows,
    }


def _combine_group_rows(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for group in groups.values()
        for row in group["queries"]
    ]


def run_one_video(
    frozen: FrozenManifest,
    index: int,
    path: Path,
    input_record: dict[str, Any],
    reader_factory: Callable[[Path], Any] = PyAVHistoricalReader,
    reader_backend: str = "pyav_historical_compat",
    av_module: Any | None = None,
) -> dict[str, Any]:
    oracle = decode_oracle_with_pyav(path, av_module=av_module)
    query_groups = build_query_groups(frozen, oracle)
    branches: dict[str, dict[str, Any]] = {"old": {}, "fixed": {}}
    for group_name, targets in query_groups.items():
        old_replay = replay_old_stop_on_count(reader_factory(path), targets)
        fixed_replay = replay_fixed_load_through_last(reader_factory(path), targets)
        branches["old"][group_name] = evaluate_replay(
            oracle, group_name, targets, old_replay
        )
        branches["fixed"][group_name] = evaluate_replay(
            oracle, group_name, targets, fixed_replay
        )

    branch_summaries: dict[str, Any] = {}
    for branch_name, groups in branches.items():
        all_rows = _combine_group_rows(groups)
        outside_rows = [
            row
            for group_name, group in groups.items()
            if group_name != "continuous_pts_negative_control"
            for row in group["queries"]
        ]
        negative_rows = groups["continuous_pts_negative_control"]["queries"]
        all_structural = all(
            group["structure_baseline"]["accepted"] for group in groups.values()
        )
        branch_summaries[branch_name] = {
            "all_groups_structure_baseline_accepted": all_structural,
            "all_queries": _metric_summary(all_rows),
            "outside_negative_control": _metric_summary(outside_rows),
            "negative_control": _metric_summary(negative_rows),
            "temporal_contract_accepted": all_structural
            and all(row["correct"] for row in all_rows),
        }

    old_wrong_outside = (
        branch_summaries["old"]["outside_negative_control"]["wrong_frame_count"] > 0
    )
    fixed_all_correct = (
        branch_summaries["fixed"]["all_queries"]["wrong_frame_count"] == 0
    )
    old_negative_control_correct = (
        branch_summaries["old"]["negative_control"]["wrong_frame_count"] == 0
    )
    both_structural = all(
        branch_summaries[name]["all_groups_structure_baseline_accepted"]
        for name in ("old", "fixed")
    )
    video_correctness = {
        "old_wrong_outside_negative_control": old_wrong_outside,
        "old_negative_control_all_correct": old_negative_control_correct,
        "fixed_all_queries_correct": fixed_all_correct,
        "both_branches_structurally_readable": both_structural,
        "temporal_contract_rejects_old": not branch_summaries["old"][
            "temporal_contract_accepted"
        ],
        "temporal_contract_accepts_fixed": branch_summaries["fixed"][
            "temporal_contract_accepted"
        ],
    }
    video_correctness["reproduced_on_video"] = (
        old_wrong_outside
        and old_negative_control_correct
        and fixed_all_correct
        and both_structural
        and video_correctness["temporal_contract_rejects_old"]
        and video_correctness["temporal_contract_accepts_fixed"]
    )

    return {
        "video_index": index,
        "selected_path": frozen.selected[index]["path"],
        "replay_backend": reader_backend,
        "replay_backend_evidence": (
            "E2 compatibility substitution preserving keyframe-backward PyAV seek, "
            "historical loop stops, PTS seconds, and RGB outputs; not the complete "
            "historical torchvision/GR00T stack"
            if reader_backend == "pyav_historical_compat"
            else "historical torchvision.io.VideoReader with pyav backend"
        ),
        "input": input_record,
        "oracle": {
            "decoder": "PyAV full sequential decode",
            "frame_count": len(oracle.frames),
            "stream_index": oracle.stream_index,
            "stream_declared_frames": oracle.stream_declared_frames,
            "duration_origin": _fraction_record(oracle.duration_origin),
            "decoded_duration": _fraction_record(oracle.duration),
            "duration_rule": oracle.duration_rule,
            "pts_nondecreasing": True,
            "all_frame_time_bases_equal_stream": True,
            "pts_seconds_rule": "float(frame.pts * stream.time_base)",
            "duplicate_pts_count": len(oracle.frames)
            - len({frame.time for frame in oracle.frames}),
            "pts_rgb_oracle_sha256": oracle.oracle_sha256,
            "frames": [frame.as_dict() for frame in oracle.frames],
        },
        "query_groups": {
            name: [_fraction_record(value) for value in values]
            for name, values in query_groups.items()
        },
        "branches": branches,
        "branch_summaries": branch_summaries,
        "correctness": video_correctness,
    }


def _checkpoint_path(work_dir: Path, index: int, selected_path: str) -> Path:
    identity = hashlib.sha256(selected_path.encode("utf-8")).hexdigest()[:12]
    return work_dir / "video-checkpoints" / f"video-{index:02d}-{identity}.json"


def _load_video_checkpoint(
    path: Path,
    frozen: FrozenManifest,
    index: int,
    record: dict[str, Any],
) -> tuple[dict[str, Any], str] | None:
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        checkpoint = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(f"invalid immutable video checkpoint {path}: {error}") from error
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise FailClosed(f"video checkpoint schema mismatch: {path}")
    if checkpoint.get("binding_sha256") != frozen.binding_sha256:
        raise FailClosed(f"video checkpoint binding mismatch: {path}")
    if checkpoint.get("video_index") != index or checkpoint.get("selected") != record:
        raise FailClosed(f"video checkpoint selection mismatch: {path}")
    result = checkpoint.get("result")
    if not isinstance(result, dict):
        raise FailClosed(f"video checkpoint lacks its result: {path}")
    if result.get("video_index") != index or result.get("selected_path") != record["path"]:
        raise FailClosed(f"video checkpoint result identity mismatch: {path}")
    if result.get("input", {}).get("observed_sha256") != record["lfs_sha256"]:
        raise FailClosed(f"video checkpoint input hash mismatch: {path}")
    return result, _bytes_sha256(raw)


def _package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for distribution in ("av", "torchvision", "torch", "huggingface-hub"):
        try:
            values[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            values[distribution] = None
    return values


def _execution_binding(replay_backend: str) -> dict[str, Any]:
    return {
        "replay_backend": replay_backend,
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }


def aggregate_results(
    frozen: FrozenManifest,
    video_results: Sequence[dict[str, Any]],
    checkpoint_inventory: Sequence[dict[str, Any]],
    *,
    created_utc: str | None = None,
    execution_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(video_results) != 10:
        raise FailClosed("aggregate requires all ten frozen videos")
    for expected_index, result in enumerate(video_results):
        if result.get("video_index") != expected_index:
            raise FailClosed("video results are not in frozen order")
    replay_backends = {result.get("replay_backend") for result in video_results}
    if len(replay_backends) != 1 or None in replay_backends:
        raise FailClosed("video checkpoints mix replay backends")

    branch_aggregate: dict[str, Any] = {}
    for branch_name in ("old", "fixed"):
        rows_by_group: dict[str, list[dict[str, Any]]] = {}
        all_rows: list[dict[str, Any]] = []
        for result in video_results:
            for group_name, group in result["branches"][branch_name].items():
                rows_by_group.setdefault(group_name, []).extend(group["queries"])
                all_rows.extend(group["queries"])
        branch_aggregate[branch_name] = {
            "all_queries": _metric_summary(all_rows),
            "by_group": {
                group_name: _metric_summary(rows)
                for group_name, rows in sorted(rows_by_group.items())
            },
            "all_videos_structure_baseline_accepted": all(
                result["branch_summaries"][branch_name][
                    "all_groups_structure_baseline_accepted"
                ]
                for result in video_results
            ),
        }

    videos_with_old_error = sum(
        result["correctness"]["old_wrong_outside_negative_control"]
        for result in video_results
    )
    all_old_negative_controls_correct = all(
        result["correctness"]["old_negative_control_all_correct"]
        for result in video_results
    )
    all_fixed_correct = all(
        result["correctness"]["fixed_all_queries_correct"]
        for result in video_results
    )
    all_structural = all(
        result["correctness"]["both_branches_structurally_readable"]
        for result in video_results
    )
    correctness = {
        "fixed_input_count": 10,
        "completed_input_count": len(video_results),
        "all_input_sizes_and_lfs_hashes_exact": all(
            result["input"]["exact_match"] for result in video_results
        ),
        "videos_with_old_error_outside_negative_control": videos_with_old_error,
        "all_old_continuous_pts_negative_controls_correct": all_old_negative_controls_correct,
        "all_fixed_queries_correct": all_fixed_correct,
        "both_branches_structurally_readable_on_all_videos": all_structural,
        "temporal_contract_rejects_old": videos_with_old_error > 0,
        "temporal_contract_accepts_fixed": all_fixed_correct and all_structural,
    }
    correctness["reproduced"] = all(
        [
            correctness["all_input_sizes_and_lfs_hashes_exact"],
            videos_with_old_error > 0,
            all_old_negative_controls_correct,
            all_fixed_correct,
            all_structural,
            correctness["temporal_contract_rejects_old"],
            correctness["temporal_contract_accepts_fixed"],
        ]
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "case": "GR00T issue #172 / PR #373 timestamp-to-frame selection",
        "case_id": CASE_ID,
        "evidence_class": "E2",
        "evidence_class_detail": (
            "self-contained exact historical loop replay on pinned public data; "
            "not execution of the complete historical GR00T API or stack"
        ),
        "created_utc": created_utc or _utc_now(),
        "status": "REPRODUCED" if correctness["reproduced"] else "NOT_REPRODUCED",
        "frozen_inputs": {
            "manifest": str(frozen.path),
            "manifest_sha256": frozen.sha256,
            "freeze_amendment": str(frozen.amendment_path),
            "freeze_amendment_sha256": frozen.amendment_sha256,
            "candidate_sidecar": frozen.candidate_sidecar,
            "time_semantics": frozen.amendment["rd7"]["time_semantics"],
            "binding": frozen.binding,
            "binding_sha256": frozen.binding_sha256,
            "source": frozen.source,
            "dataset": {
                key: frozen.dataset[key]
                for key in (
                    "repo_id",
                    "revision",
                    "eligibility",
                    "candidate_count",
                    "candidate_paths_sha256",
                    "selection",
                    "selected",
                )
            },
            "queries": frozen.queries,
            "no_replacement_after_observed_outcome": True,
        },
        "execution": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _file_sha256(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
            "execution_binding": execution_binding,
            "replay_backend": (
                video_results[0]["replay_backend"] if video_results else None
            ),
            "backend_compatibility": (
                "The default PyAVHistoricalReader is an explicit E2 compatibility "
                "substitution because torchvision 0.26 removed VideoReader. It "
                "preserves the historical keyframe-backward seek and selection loops "
                "but is not the complete historical software stack."
                if video_results
                and video_results[0]["replay_backend"] == "pyav_historical_compat"
                else "torchvision.io.VideoReader with its pyav backend was used"
            ),
            "historical_replay": {
                "old": "seek(first); load until current>=last or loaded_count>=request_count; return loaded order",
                "fixed": "seek(first); load until current>=last; for each target return latest loaded PTS<=target, else first",
            },
            "oracle": "independent full sequential PyAV decode with exact PTS and RGB-byte SHA-256",
            "local_frame_interval_rule": "minimum positive gap to an adjacent distinct source PTS",
        },
        "checkpoint_inventory": [
            {
                key: value
                for key, value in item.items()
                if key != "resumed"
            }
            for item in checkpoint_inventory
        ],
        "videos": list(video_results),
        "aggregate": branch_aggregate,
        "correctness": correctness,
        "limitations": [
            "E2 replays the exact parent and PR #373 selection loops but does not execute the complete historical GR00T stack.",
            "The ten videos are the manifest's mechanically selected official inputs; no failed, null, or inconvenient input is replaced.",
            "RGB digests require the pinned PyAV/torchvision decode environment recorded above; an incompatible decoder is a reported failure, not a reason to weaken matching.",
            "The top-level natural unit is a video; timestamp-query aggregates are descriptive within-video diagnostics.",
        ],
        "sources": {
            "issue": "https://github.com/NVIDIA/Isaac-GR00T/issues/172",
            "pull_request": "https://github.com/NVIDIA/Isaac-GR00T/pull/373",
            "dataset": f"https://huggingface.co/datasets/{EXPECTED_DATASET_REPO}/tree/{EXPECTED_DATASET_REVISION}",
        },
    }


@contextmanager
def _exclusive_run_lock(work_dir: Path) -> Iterable[None]:
    try:
        import fcntl
    except ImportError as error:
        raise FailClosed("RD7 resumability requires a Unix advisory lock") from error
    work_dir.mkdir(parents=True, exist_ok=True)
    lock_path = work_dir / "run.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FailClosed(f"another RD7 runner holds {lock_path}") from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_completed_output(
    args: argparse.Namespace,
    frozen: FrozenManifest,
    state: dict[str, Any],
    events_path: Path,
) -> dict[str, Any] | None:
    if not args.output.exists():
        if state.get("status") == "COMPLETE":
            raise FailClosed("COMPLETE state points to a missing result")
        return None
    try:
        raw = args.output.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise FailClosed(f"existing result is invalid and cannot be overwritten: {error}") from error
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise FailClosed("existing result schema mismatch")
    if payload.get("case_id") != CASE_ID:
        raise FailClosed("existing result case identity mismatch")
    frozen_inputs = payload.get("frozen_inputs", {})
    if frozen_inputs.get("binding") != frozen.binding:
        raise FailClosed("existing result frozen binding mismatch")
    if frozen_inputs.get("binding_sha256") != frozen.binding_sha256:
        raise FailClosed("existing result binding digest mismatch")
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) != 10:
        raise FailClosed("existing result does not contain all ten videos")
    if [value.get("selected_path") for value in videos] != [
        record["path"] for record in frozen.selected
    ]:
        raise FailClosed("existing result video membership or order mismatch")
    inventory = payload.get("checkpoint_inventory")
    if not isinstance(inventory, list) or len(inventory) != 10:
        raise FailClosed("existing result checkpoint inventory is incomplete")
    work_root = args.work_dir.resolve()
    for index, item in enumerate(inventory):
        if item.get("video_index") != index:
            raise FailClosed("existing checkpoint inventory order mismatch")
        checkpoint = Path(item.get("checkpoint", "")).resolve()
        try:
            checkpoint.relative_to(work_root)
        except ValueError as error:
            raise FailClosed("existing checkpoint escapes the run directory") from error
        if not checkpoint.is_file() or _file_sha256(checkpoint) != item.get(
            "checkpoint_sha256"
        ):
            raise FailClosed(f"existing checkpoint hash mismatch: {checkpoint}")
    observed_result_sha256 = _bytes_sha256(raw)
    if state.get("status") == "COMPLETE":
        if state.get("result_path") != str(args.output):
            raise FailClosed("COMPLETE state result path mismatch")
        if state.get("result_sha256") != observed_result_sha256:
            raise FailClosed("COMPLETE state result hash mismatch")
        return payload

    # The result was atomically published but the mutable state/event tail did
    # not survive.  Repair bookkeeping only; never replay a completed video.
    state["status"] = "COMPLETE"
    state["result_path"] = str(args.output)
    state["result_sha256"] = observed_result_sha256
    _atomic_write_json(args.work_dir / "state.json", state)
    _append_event(
        events_path,
        frozen.binding_sha256,
        "RUN_COMPLETE_RECOVERED_FROM_IMMUTABLE_RESULT",
        {
            "result_path": str(args.output),
            "result_sha256": observed_result_sha256,
            "status": payload.get("status"),
        },
    )
    return payload


def _execute_locked(args: argparse.Namespace) -> dict[str, Any]:
    frozen = load_frozen_manifest(args.manifest, args.freeze_amendment)
    work_dir: Path = args.work_dir
    video_root: Path = args.video_root or (work_dir / "downloads")
    state_path = work_dir / "state.json"
    events_path = work_dir / "events.jsonl"
    work_dir.mkdir(parents=True, exist_ok=True)
    execution_binding = _execution_binding(args.replay_backend)
    state = _load_state(state_path, frozen, execution_binding)
    _verify_event_log(events_path, frozen.binding_sha256)
    completed_output = _validate_completed_output(args, frozen, state, events_path)
    if completed_output is not None:
        return completed_output
    event_sequence, _ = _verify_event_log(events_path, frozen.binding_sha256)
    event_type = "RUN_RESUME" if event_sequence else "RUN_START"
    _append_event(
        events_path,
        frozen.binding_sha256,
        event_type,
        {
            "manifest_sha256": frozen.sha256,
            "freeze_amendment_sha256": frozen.amendment_sha256,
            "selected_count": len(frozen.selected),
            "download_authorized": bool(args.download),
            "download_endpoint": EXPECTED_HF_ENDPOINT,
            "download_authentication": "disabled",
            "replay_backend": args.replay_backend,
        },
    )
    state["status"] = "RUNNING"
    state["failure"] = None
    _atomic_write_json(state_path, state)

    results: list[dict[str, Any]] = []
    checkpoint_inventory: list[dict[str, Any]] = []
    for index, record in enumerate(frozen.selected):
        checkpoint_path = _checkpoint_path(work_dir, index, record["path"])
        existing = _load_video_checkpoint(
            checkpoint_path, frozen, index, record
        )
        if existing is not None:
            result, checkpoint_sha256 = existing
            if result.get("replay_backend") != args.replay_backend:
                raise FailClosed(
                    f"completed checkpoint replay backend drifted: {record['path']}"
                )
            # Rehash the selected input on every resume.  The checkpoint is
            # reusable only while the exact frozen bytes remain present.
            local_path = _safe_local_video_path(video_root, record["path"])
            observed = _verify_video_file(local_path, record)
            if result["input"] != observed:
                raise FailClosed(f"completed input record drifted: {record['path']}")
            results.append(result)
            state["completed"][record["path"]] = {
                "video_index": index,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "input_sha256": observed["observed_sha256"],
                "oracle_sha256": result["oracle"]["pts_rgb_oracle_sha256"],
            }
            _atomic_write_json(state_path, state)
            checkpoint_inventory.append(
                {
                    "video_index": index,
                    "selected_path": record["path"],
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "resumed": True,
                }
            )
            continue

        _append_event(
            events_path,
            frozen.binding_sha256,
            "VIDEO_INPUT_BEGIN",
            {"video_index": index, "selected_path": record["path"]},
        )
        local_path, observed, downloaded = _ensure_video(
            frozen, record, video_root, bool(args.download)
        )
        _append_event(
            events_path,
            frozen.binding_sha256,
            "VIDEO_INPUT_VERIFIED",
            {
                "video_index": index,
                "selected_path": record["path"],
                "observed_bytes": observed["observed_bytes"],
                "observed_sha256": observed["observed_sha256"],
                "downloaded_this_invocation": downloaded,
            },
        )
        result = run_one_video(
            frozen,
            index,
            local_path,
            observed,
            reader_factory=_reader_factory_for_backend(args.replay_backend),
            reader_backend=args.replay_backend,
        )
        checkpoint = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "binding_sha256": frozen.binding_sha256,
            "video_index": index,
            "selected": record,
            "completed_utc": _utc_now(),
            "result": result,
        }
        checkpoint_sha256 = _write_immutable_json(checkpoint_path, checkpoint)
        state["completed"][record["path"]] = {
            "video_index": index,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "input_sha256": observed["observed_sha256"],
            "oracle_sha256": result["oracle"]["pts_rgb_oracle_sha256"],
        }
        _atomic_write_json(state_path, state)
        _append_event(
            events_path,
            frozen.binding_sha256,
            "VIDEO_COMPLETE",
            {
                "video_index": index,
                "selected_path": record["path"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "oracle_sha256": result["oracle"]["pts_rgb_oracle_sha256"],
                "reproduced_on_video": result["correctness"]["reproduced_on_video"],
            },
        )
        results.append(result)
        checkpoint_inventory.append(
            {
                "video_index": index,
                "selected_path": record["path"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "resumed": False,
            }
        )

    result_payload = aggregate_results(
        frozen,
        results,
        checkpoint_inventory,
        created_utc=state["run_created_utc"],
        execution_binding=execution_binding,
    )
    result_sha256 = _write_immutable_json(args.output, result_payload)
    state["status"] = "COMPLETE"
    state["result_path"] = str(args.output)
    state["result_sha256"] = result_sha256
    _atomic_write_json(state_path, state)
    _append_event(
        events_path,
        frozen.binding_sha256,
        "RUN_COMPLETE",
        {
            "result_path": str(args.output),
            "result_sha256": state["result_sha256"],
            "status": result_payload["status"],
        },
    )
    return result_payload


def execute(args: argparse.Namespace) -> dict[str, Any]:
    with _exclusive_run_lock(args.work_dir):
        try:
            return _execute_locked(args)
        except Exception as error:
            # Best-effort failure evidence is written while the same run lock
            # is still held.  A malformed binding or event ledger remains a
            # hard failure and is never repaired or truncated here.
            try:
                frozen = load_frozen_manifest(
                    args.manifest, args.freeze_amendment
                )
                state_path = args.work_dir / "state.json"
                state = _load_state(
                    state_path,
                    frozen,
                    _execution_binding(args.replay_backend),
                )
                state["status"] = "FAILED"
                state["failure"] = {
                    "utc": _utc_now(),
                    "class": type(error).__name__,
                    "message": str(error),
                }
                _atomic_write_json(state_path, state)
                _append_event(
                    args.work_dir / "events.jsonl",
                    frozen.binding_sha256,
                    "RUN_FAILED",
                    state["failure"],
                )
            except Exception as recording_error:
                print(
                    f"RD7 failure evidence could not be recorded: {recording_error}",
                    file=sys.stderr,
                )
            raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--freeze-amendment", type=Path, default=DEFAULT_FREEZE_AMENDMENT
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--video-root",
        type=Path,
        default=None,
        help="root containing the exact manifest-relative paths (default: WORK_DIR/downloads)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--replay-backend",
        choices=("pyav_historical_compat", "torchvision_videoreader"),
        default="pyav_historical_compat",
        help=(
            "default compatibility reader preserves the historical PyAV seek/loop; "
            "torchvision_videoreader is optional when that API still exists"
        ),
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="allow exact-revision Hugging Face downloads; absent means strictly offline",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = execute(args)
    except Exception as error:
        print(f"RD7 failed closed: {error}", file=sys.stderr)
        return 2
    summary = {
        "status": payload["status"],
        "reproduced": payload["correctness"]["reproduced"],
        "videos": payload["correctness"]["completed_input_count"],
        "videos_with_old_error": payload["correctness"][
            "videos_with_old_error_outside_negative_control"
        ],
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if payload["correctness"]["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
