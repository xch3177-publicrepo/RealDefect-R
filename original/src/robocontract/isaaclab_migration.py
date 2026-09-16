"""Audit Isaac Lab's legacy-HDF5 quaternion migration on public data.

The audit deliberately executes the frozen upstream converter source.  Small
module stubs satisfy imports unavailable on macOS; the HDF5 traversal and
field-selection logic under test remain the upstream implementation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any


@dataclass(frozen=True)
class QuaternionObligation:
    family: str
    path: str
    source_wxyz: Any
    target_xyzw: Any


def _dependencies() -> tuple[Any, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise RuntimeError("Isaac Lab migration audit requires h5py and numpy") from error
    return h5py, np


def _upstream_import_stubs() -> tuple[dict[str, Any], list[str]]:
    """Install only the import surfaces needed to load the frozen source."""

    _, np = _dependencies()
    names = [
        "torch",
        "isaaclab",
        "isaaclab.utils",
        "isaaclab.utils.math",
        "isaaclab.utils.datasets",
        "isaaclab.utils.datasets.dataset_file_handler_base",
        "isaaclab.utils.datasets.episode_data",
        "isaaclab.utils.datasets.hdf5_dataset_file_handler",
    ]
    previous = {name: sys.modules.get(name) for name in names}

    torch_module = types.ModuleType("torch")
    isaaclab_module = types.ModuleType("isaaclab")
    utils_module = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")
    datasets_module = types.ModuleType("isaaclab.utils.datasets")
    base_module = types.ModuleType("isaaclab.utils.datasets.dataset_file_handler_base")
    episode_module = types.ModuleType("isaaclab.utils.datasets.episode_data")

    # This is the documented WXYZ -> XYZW permutation.  It substitutes only
    # for the unavailable torch-based helper; the upstream migration's HDF5
    # recursion, key matching, version marking, and writes execute unchanged.
    def convert_quat(quaternion: Any, to: str = "xyzw") -> Any:
        if to != "xyzw":
            raise ValueError(f"audit stub only supports conversion to xyzw, got {to}")
        return np.roll(quaternion, -1, axis=-1)

    class DatasetFileHandlerBase:
        pass

    class EpisodeData:
        pass

    math_module.convert_quat = convert_quat
    base_module.DatasetFileHandlerBase = DatasetFileHandlerBase
    episode_module.EpisodeData = EpisodeData
    isaaclab_module.__path__ = []
    utils_module.__path__ = []
    datasets_module.__path__ = []

    sys.modules.update(
        {
            "torch": torch_module,
            "isaaclab": isaaclab_module,
            "isaaclab.utils": utils_module,
            "isaaclab.utils.math": math_module,
            "isaaclab.utils.datasets": datasets_module,
            "isaaclab.utils.datasets.dataset_file_handler_base": base_module,
            "isaaclab.utils.datasets.episode_data": episode_module,
        }
    )
    return previous, names


def run_frozen_upstream_converter(
    converter_source: Path, source_hdf5: Path, target_hdf5: Path
) -> None:
    """Load the fixed-revision module and invoke its static converter method."""

    previous, names = _upstream_import_stubs()
    module_name = "isaaclab.utils.datasets.hdf5_dataset_file_handler"
    try:
        spec = importlib.util.spec_from_file_location(module_name, converter_source)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load upstream source from {converter_source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        module.HDF5DatasetFileHandler.convert_dataset_to_xyzw(
            str(source_hdf5), str(target_hdf5)
        )
    finally:
        for name in names:
            old = previous[name]
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _datasets(handle: Any) -> dict[str, Any]:
    h5py, _ = _dependencies()
    found: dict[str, Any] = {}

    def visit(name: str, value: Any) -> None:
        if isinstance(value, h5py.Dataset):
            found[name] = value

    handle.visititems(visit)
    return found


def collect_obligations(source_hdf5: Path, target_hdf5: Path) -> list[QuaternionObligation]:
    """Collect versioned WXYZ->XYZW obligations established by v2.3.2 task code."""

    h5py, np = _dependencies()
    obligations: list[QuaternionObligation] = []
    with h5py.File(source_hdf5, "r") as source, h5py.File(target_hdf5, "r") as target:
        source_datasets = _datasets(source)
        for path, dataset in source_datasets.items():
            values = np.asarray(dataset)
            target_values = np.asarray(target[path])
            leaf = path.rsplit("/", 1)[-1]
            if leaf == "root_pose" and values.ndim >= 1 and values.shape[-1] == 7:
                obligations.append(
                    QuaternionObligation(
                        "root_pose",
                        path,
                        values[..., 3:7].reshape(-1, 4),
                        target_values[..., 3:7].reshape(-1, 4),
                    )
                )
            elif path.endswith("/obs/eef_quat") and values.shape[-1] == 4:
                obligations.append(
                    QuaternionObligation(
                        "eef_quat", path, values.reshape(-1, 4), target_values.reshape(-1, 4)
                    )
                )
            elif path.endswith("/obs/cube_orientations") and values.shape[-1] % 4 == 0:
                obligations.append(
                    QuaternionObligation(
                        "cube_orientations",
                        path,
                        values.reshape(-1, 4),
                        target_values.reshape(-1, 4),
                    )
                )
            elif path.endswith("/obs/object") and values.shape[-1] >= 21:
                source_quaternions = np.concatenate(
                    [values[..., start : start + 4] for start in (3, 10, 17)], axis=0
                ).reshape(-1, 4)
                target_quaternions = np.concatenate(
                    [target_values[..., start : start + 4] for start in (3, 10, 17)], axis=0
                ).reshape(-1, 4)
                obligations.append(
                    QuaternionObligation(
                        "object_embedded", path, source_quaternions, target_quaternions
                    )
                )
    return obligations


def _normalize(quaternions: Any) -> Any:
    _, np = _dependencies()
    values = np.asarray(quaternions, dtype="f8")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("zero-norm quaternion in declared orientation field")
    return values / norms


def _matrices_wxyz(quaternions: Any) -> Any:
    _, np = _dependencies()
    q = _normalize(quaternions)
    w, x, y, z = np.moveaxis(q, -1, 0)
    result = np.empty(q.shape[:-1] + (3, 3), dtype="f8")
    result[..., 0, 0] = 1 - 2 * (y * y + z * z)
    result[..., 0, 1] = 2 * (x * y - z * w)
    result[..., 0, 2] = 2 * (x * z + y * w)
    result[..., 1, 0] = 2 * (x * y + z * w)
    result[..., 1, 1] = 1 - 2 * (x * x + z * z)
    result[..., 1, 2] = 2 * (y * z - x * w)
    result[..., 2, 0] = 2 * (x * z - y * w)
    result[..., 2, 1] = 2 * (y * z + x * w)
    result[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def _matrices_xyzw(quaternions: Any) -> Any:
    _, np = _dependencies()
    q = _normalize(quaternions)
    x, y, z, w = np.moveaxis(q, -1, 0)
    result = np.empty(q.shape[:-1] + (3, 3), dtype="f8")
    result[..., 0, 0] = 1 - 2 * (y * y + z * z)
    result[..., 0, 1] = 2 * (x * y - z * w)
    result[..., 0, 2] = 2 * (x * z + y * w)
    result[..., 1, 0] = 2 * (x * y + z * w)
    result[..., 1, 1] = 1 - 2 * (x * x + z * z)
    result[..., 1, 2] = 2 * (y * z - x * w)
    result[..., 2, 0] = 2 * (x * z - y * w)
    result[..., 2, 1] = 2 * (y * z + x * w)
    result[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def physical_errors_deg(source_wxyz: Any, target_xyzw: Any) -> Any:
    """Independent SO(3) oracle using explicit Hamilton rotation matrices."""

    _, np = _dependencies()
    source_matrices = _matrices_wxyz(source_wxyz)
    target_matrices = _matrices_xyzw(target_xyzw)
    relative = np.matmul(np.swapaxes(source_matrices, -1, -2), target_matrices)
    traces = np.trace(relative, axis1=-2, axis2=-1)
    cosines = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


def audit_migration(source_hdf5: Path, target_hdf5: Path) -> dict[str, Any]:
    h5py, np = _dependencies()
    obligations = collect_obligations(source_hdf5, target_hdf5)
    family_counts: dict[str, int] = defaultdict(int)
    family_discharged: dict[str, int] = defaultdict(int)
    failures: list[dict[str, Any]] = []
    all_errors: list[Any] = []

    for obligation in obligations:
        expected = np.roll(obligation.source_wxyz, -1, axis=-1)
        matches = np.all(np.isclose(obligation.target_xyzw, expected, atol=1e-7), axis=-1)
        count = int(len(matches))
        discharged = int(np.count_nonzero(matches))
        family_counts[obligation.family] += count
        family_discharged[obligation.family] += discharged
        if discharged != count:
            errors = physical_errors_deg(
                obligation.source_wxyz[~matches], obligation.target_xyzw[~matches]
            )
            all_errors.append(errors)
            failures.append(
                {
                    "family": obligation.family,
                    "path": obligation.path,
                    "instances": count,
                    "undischarged": count - discharged,
                    "mean_orientation_error_deg": float(np.mean(errors)),
                }
            )

    with h5py.File(source_hdf5, "r") as source, h5py.File(target_hdf5, "r") as target:
        source_datasets = _datasets(source)
        target_datasets = _datasets(target)
        same_paths = set(source_datasets) == set(target_datasets)
        shape_equal = sum(
            source_datasets[path].shape == target_datasets[path].shape
            for path in source_datasets
            if path in target_datasets
        )
        dtype_equal = sum(
            source_datasets[path].dtype == target_datasets[path].dtype
            for path in source_datasets
            if path in target_datasets
        )
        bytewise_equal = sum(
            np.array_equal(source_datasets[path][...], target_datasets[path][...])
            for path in source_datasets
            if path in target_datasets
        )
        source_total = int(source["data"].attrs["total"])
        episode_names = sorted(source["data"].keys())
        declared_samples = sum(
            int(source[f"data/{name}"].attrs["num_samples"]) for name in episode_names
        )
        action_rows = sum(len(source[f"data/{name}/actions"]) for name in episode_names)
        target_version = int(target.attrs.get("format_version", 0))

    converted = sum(family_discharged.values())
    total = sum(family_counts.values())
    errors = np.concatenate(all_errors) if all_errors else np.asarray([], dtype="f8")
    schema_baseline_accepts = (
        same_paths
        and shape_equal == len(source_datasets)
        and dtype_equal == len(source_datasets)
        and target_version == 1
    )
    return {
        "schema_baseline": {
            "accepted": schema_baseline_accepts,
            "source_dataset_count": len(source_datasets),
            "target_dataset_count": len(target_datasets),
            "same_dataset_paths": same_paths,
            "same_shape_count": shape_equal,
            "same_dtype_count": dtype_equal,
            "bytewise_unchanged_count": bytewise_equal,
            "target_format_version": target_version,
        },
        "independent_metadata_observation": {
            "data_total_attribute": source_total,
            "sum_episode_num_samples": declared_samples,
            "sum_action_rows": action_rows,
            "consistent": source_total == declared_samples == action_rows,
            "excluded_from_quaternion_contract_decision": True,
        },
        "contract": {
            "accepted": converted == total,
            "declared_instances": total,
            "converted_instances": converted,
            "undischarged_instances": total - converted,
            "instance_weighted_coverage": converted / total if total else 0.0,
            "family_instances": dict(sorted(family_counts.items())),
            "family_converted_instances": dict(sorted(family_discharged.items())),
            "failed_paths": failures,
        },
        "physical_oracle": {
            "undischarged_instances_evaluated": int(len(errors)),
            "nonzero_error_instances": int(np.count_nonzero(errors > 1e-6)),
            "over_90_degree_instances": int(np.count_nonzero(errors > 90.0)),
            "mean_error_deg": float(np.mean(errors)) if len(errors) else 0.0,
            "median_error_deg": float(np.median(errors)) if len(errors) else 0.0,
            "p95_error_deg": float(np.percentile(errors, 95)) if len(errors) else 0.0,
            "max_error_deg": float(np.max(errors)) if len(errors) else 0.0,
        },
    }
