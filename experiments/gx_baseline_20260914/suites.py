"""GX-BASE-20260914 step 2: the three frozen expectation suites.

Everything here is executed natively by Great Expectations. Arm C's semantic rules are implemented as
Expectation classes registered into GX through `MulticolumnMapExpectation` with pandas metric
providers, so the domain knowledge lives inside the framework rather than in a wrapper script.

Arm A  declared-schema structure and range (parameters from the artifact's own declared schema)
Arm B  disjoint-reference-fitted (parameters fitted mechanically on a disjoint fixture)
Arm C  information-enriched (reference information as batch columns and/or expectation parameters)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import great_expectations.expectations as gxe
from great_expectations.execution_engine import PandasExecutionEngine
from great_expectations.expectations.expectation import MulticolumnMapExpectation
from great_expectations.expectations.metrics.map_metric_provider import (
    MulticolumnMapMetricProvider,
    multicolumn_condition_partial,
)

TWO_PI = 2.0 * np.pi


# ==============================================================================================
# Arm C custom Expectations, registered into GX by subclassing.
# ==============================================================================================
class _WindowSelectionRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.window_selection_matches_clamped_request"
    condition_domain_keys = ("column_list",)

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, **kwargs):
        df = column_list
        request = df["absolute_index"] + df["delta_index"]
        start, end = df["episode_start"], df["episode_end"]
        expected_index = request.clip(lower=start, upper=end - 1)
        expected_pad = (request < start) | (request >= end)
        return (df["selected_index"] == expected_index) & (df["is_pad"] == expected_pad)


class ExpectWindowSelectionToMatchClampedRequest(MulticolumnMapExpectation):
    """Selected row identity and padding flag must follow the clamp-and-pad rule over episode bounds."""

    map_metric = "multicolumn_values.window_selection_matches_clamped_request"


class _GlobalIntervalChainRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.global_row_intervals_chain"
    condition_domain_keys = ("column_list",)

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, **kwargs):
        df = column_list.sort_values("episode_index")
        lengths = df["length"].to_numpy()
        prefix = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        ok = (df["dataset_from_index"].to_numpy() == prefix) & (
            df["dataset_to_index"].to_numpy() == prefix + lengths
        )
        return pd.Series(ok, index=df.index).reindex(column_list.index)


class ExpectGlobalRowIntervalsToChain(MulticolumnMapExpectation):
    """Episode row intervals must form one contiguous global chain starting at zero."""

    map_metric = "multicolumn_values.global_row_intervals_chain"


class _IntervalLengthRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.row_interval_length_matches_declared_length"
    condition_domain_keys = ("column_list",)

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, **kwargs):
        df = column_list
        return (df["dataset_to_index"] - df["dataset_from_index"]) == df["length"]


class ExpectRowIntervalLengthToMatchDeclaredLength(MulticolumnMapExpectation):
    """Each episode's declared row interval width must equal its declared frame count."""

    map_metric = "multicolumn_values.row_interval_length_matches_declared_length"


def _as_wxyz(q, order):
    """Return normalised (w, vector) components from a quaternion array in the given order."""
    q = np.asarray(q, dtype="f8")
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    if order == "wxyz":
        return q[:, 0], q[:, 1:4]
    return q[:, 3], q[:, 0:3]


def quaternion_geodesic_deg(q_source, source_order, q_target, target_order):
    """Geodesic SO(3) angle between two orientations, evaluated stably near the identity.

    Mathematically identical to the trace/arccos rotation distance, but computed from the relative
    quaternion as `2*atan2(||vec||, |w|)`. `arccos` near an argument of 1 loses roughly half the
    available precision, which puts a floor of order 1e-6 degrees on near-identical rotations;
    `atan2` of a small numerator has no such floor, so a 1e-6-degree decision boundary is meaningful.
    Taking `|w|` makes the result invariant under the q / -q double cover.
    """
    w1, v1 = _as_wxyz(q_source, source_order)
    w2, v2 = _as_wxyz(q_target, target_order)
    # Relative quaternion q_source^-1 (x) q_target, for unit quaternions.
    rel_w = w1 * w2 + np.einsum("ij,ij->i", v1, v2)
    rel_v = w1[:, None] * v2 - w2[:, None] * v1 - np.cross(v1, v2)
    return np.degrees(2.0 * np.arctan2(np.linalg.norm(rel_v, axis=1), np.abs(rel_w)))


class _QuaternionConventionRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.quaternion_matches_source_under_conventions"
    condition_domain_keys = ("column_list",)
    condition_value_keys = ("max_error_deg",)

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, max_error_deg, **kwargs):
        df = column_list
        error = quaternion_geodesic_deg(
            df[["source_qw", "source_qx", "source_qy", "source_qz"]].to_numpy(), "wxyz",
            df[["qx", "qy", "qz", "qw"]].to_numpy(), "xyzw")
        return pd.Series(error <= max_error_deg, index=df.index)


class ExpectQuaternionToMatchSourceUnderDeclaredConventions(MulticolumnMapExpectation):
    """Stored orientation must represent the source rotation once both conventions are applied."""

    map_metric = "multicolumn_values.quaternion_matches_source_under_conventions"
    success_keys = ("max_error_deg", "mostly")
    max_error_deg: float = 1e-6


class _DisplacementFrameRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.world_delta_equals_rotated_displacement"
    condition_domain_keys = ("column_list",)
    condition_value_keys = ("base_rotation", "base_translation", "atol")

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, base_rotation, base_translation, atol, **kwargs):
        df = column_list
        rotation = np.asarray(base_rotation, dtype="f8").reshape(3, 3)
        delta = df[["delta_x", "delta_y", "delta_z"]].to_numpy(dtype="f8")
        observed = df[["world_delta_x", "world_delta_y", "world_delta_z"]].to_numpy(dtype="f8")
        # A displacement is rotated by the base frame; it is not offset by the base origin.
        expected = delta @ rotation.T
        return pd.Series(np.linalg.norm(observed - expected, axis=1) <= atol, index=df.index)


class ExpectWorldDeltaToEqualRotatedDisplacement(MulticolumnMapExpectation):
    """A commanded displacement must be rotated into the world frame without the base translation."""

    map_metric = "multicolumn_values.world_delta_equals_rotated_displacement"
    success_keys = ("base_rotation", "base_translation", "atol", "mostly")
    base_rotation: list = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    base_translation: list = [0.0, 0.0, 0.0]
    atol: float = 1e-9


class _CalibrationAnchorRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.calibration_anchors_map_to_endpoints"
    condition_domain_keys = ("column_list",)
    condition_value_keys = ("atol",)

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, atol, **kwargs):
        df = column_list
        is_anchor = (df["row_kind"] == "anchor").to_numpy()
        deviation = (df["gripper_norm"] - df["expected_norm"]).abs().to_numpy()
        # Only the calibration anchor rows carry an expected endpoint; data rows are unconstrained.
        ok = np.where(is_anchor, deviation <= atol, True).astype(bool)
        return pd.Series(ok, index=df.index)


class ExpectCalibrationAnchorsToMapToEndpoints(MulticolumnMapExpectation):
    """Documented encoder anchors must normalise to their calibrated closed/open endpoints."""

    map_metric = "multicolumn_values.calibration_anchors_map_to_endpoints"
    success_keys = ("atol", "mostly")
    atol: float = 1e-3


class _FrameCorrespondenceRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.returned_frame_matches_requested_time"
    condition_domain_keys = ("column_list",)

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, **kwargs):
        df = column_list
        pts_ok = (df["returned_pts_seconds"] - df["expected_pts_seconds"]).abs() <= (
            0.5 * df["local_frame_interval_seconds"]
        )
        identity_ok = [
            str(got) in str(allowed).split("|")
            for got, allowed in zip(df["returned_rgb_sha256"], df["expected_rgb_sha256_set"])
        ]
        return pts_ok & pd.Series(identity_ok, index=df.index)


class ExpectReturnedFrameToMatchRequestedTime(MulticolumnMapExpectation):
    """The returned frame's timestamp and image identity must correspond to the requested time."""

    map_metric = "multicolumn_values.returned_frame_matches_requested_time"


class _ReferenceStatisticRule(MulticolumnMapMetricProvider):
    condition_metric_name = "multicolumn_values.statistics_agree_with_reference"
    condition_domain_keys = ("column_list",)

    @multicolumn_condition_partial(engine=PandasExecutionEngine)
    def _pandas(cls, column_list, **kwargs):
        df = column_list
        tol = df["absolute_tolerance"]
        return ((df["mean"] - df["reference_mean"]).abs() <= tol) & (
            (df["std"] - df["reference_std"]).abs() <= tol
        )


class ExpectStatisticsToAgreeWithReference(MulticolumnMapExpectation):
    """Published per-component mean/std must agree with an independent reference within tolerance."""

    map_metric = "multicolumn_values.statistics_agree_with_reference"


CUSTOM_TYPES = {
    "expect_window_selection_to_match_clamped_request",
    "expect_global_row_intervals_to_chain",
    "expect_row_interval_length_to_match_declared_length",
    "expect_quaternion_to_match_source_under_declared_conventions",
    "expect_world_delta_to_equal_rotated_displacement",
    "expect_calibration_anchors_to_map_to_endpoints",
    "expect_returned_frame_to_match_requested_time",
    "expect_statistics_to_agree_with_reference",
}


# ==============================================================================================
# Arm A - declared-schema structure and range
# ==============================================================================================
def _schema_scaffold(columns: dict[str, str]) -> list:
    """Column set, non-null and dtype expectations from a declared column -> dtype mapping."""
    out = [gxe.ExpectTableColumnsToMatchSet(column_set=sorted(columns))]
    for name, dtype in columns.items():
        out.append(gxe.ExpectColumnValuesToNotBeNull(column=name))
        out.append(gxe.ExpectColumnValuesToBeOfType(column=name, type_=dtype))
    return out


def arm_a(case: str, notes: dict) -> list:
    if case == "RD1":
        total = notes["declared_total_frames"]
        exp = _schema_scaffold({
            "absolute_index": "int64", "delta_index": "int64",
            "selected_index": "int64", "is_pad": "bool",
        })
        exp += [
            gxe.ExpectColumnValuesToBeBetween(column="absolute_index", min_value=0, max_value=total - 1),
            gxe.ExpectColumnValuesToBeBetween(column="selected_index", min_value=0, max_value=total - 1),
            gxe.ExpectTableRowCountToEqual(value=notes["declared_episode_bounds"][1]
                                          - notes["declared_episode_bounds"][0]),
        ]
        return exp

    if case == "RD2":
        exp = _schema_scaffold({
            "episode_index": "int64", "length": "int64",
            "dataset_from_index": "int64", "dataset_to_index": "int64",
        })
        exp += [
            gxe.ExpectColumnValuesToBeUnique(column="episode_index"),
            gxe.ExpectColumnValuesToBeIncreasing(column="episode_index"),
            gxe.ExpectColumnValuesToBeBetween(column="length", min_value=1),
            gxe.ExpectColumnValuesToBeBetween(column="dataset_from_index", min_value=0),
            gxe.ExpectColumnValuesToBeBetween(column="dataset_to_index", min_value=0),
            gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="dataset_to_index", column_B="dataset_from_index"),
            # LeRobot v3 declares the interval width to be the episode's frame count.
            ExpectRowIntervalLengthToMatchDeclaredLength(
                column_list=["length", "dataset_from_index", "dataset_to_index"]),
        ]
        return exp

    if case == "RD4":
        exp = _schema_scaffold({
            "field_kind": "object", "field_path": "object", "instance_index": "int64",
            "qx": "float64", "qy": "float64", "qz": "float64", "qw": "float64", "norm": "float64",
        })
        for c in ("qx", "qy", "qz", "qw"):
            exp.append(gxe.ExpectColumnValuesToBeBetween(column=c, min_value=-1.0, max_value=1.0))
        exp.append(gxe.ExpectColumnValuesToBeBetween(
            column="norm", min_value=1.0 - 1e-6, max_value=1.0 + 1e-6))
        return exp

    if case == "RD5":
        lo = min(notes["declared_controller_output_min"])
        hi = max(notes["declared_controller_output_max"])
        exp = _schema_scaffold({
            "demo_id": "object", "demo_number": "int64", "step": "int64",
            "action_x": "float64", "action_y": "float64", "action_z": "float64",
            "world_delta_x": "float64", "world_delta_y": "float64", "world_delta_z": "float64",
            "world_delta_norm_m": "float64",
        })
        for c in ("action_x", "action_y", "action_z"):
            exp.append(gxe.ExpectColumnValuesToBeBetween(column=c, min_value=-1.0, max_value=1.0))
        for c in ("world_delta_x", "world_delta_y", "world_delta_z"):
            exp.append(gxe.ExpectColumnValuesToBeBetween(column=c, min_value=lo, max_value=hi))
        exp += [
            gxe.ExpectColumnValuesToBeBetween(
                column="world_delta_norm_m", min_value=0.0,
                max_value=notes["declared_max_displacement_norm_m"]),
            gxe.ExpectColumnValuesToBeBetween(column="step", min_value=0),
        ]
        return exp

    if case == "RD6":
        exp = _schema_scaffold({
            "row_kind": "object", "side": "object", "source_index": "int64",
            "gripper_raw": "float64", "gripper_norm": "float64",
        })
        exp += [
            gxe.ExpectColumnDistinctValuesToBeInSet(column="row_kind", value_set=["anchor", "data"]),
            # Both pinned branches document the normalised gripper output as 0-1.
            gxe.ExpectColumnValuesToBeBetween(column="gripper_norm", min_value=0.0, max_value=1.0),
            gxe.ExpectColumnValuesToBeBetween(column="gripper_raw", min_value=-TWO_PI, max_value=TWO_PI),
        ]
        return exp

    if case == "RD7":
        exp = _schema_scaffold({
            "video_index": "int64", "group": "object", "query_index": "int64",
            "requested_seconds": "float64", "returned_pts_seconds": "float64",
            "returned_height": "int64", "returned_width": "int64", "returned_channels": "int64",
            "returned_dtype": "object", "returned_rgb_sha256": "object",
        })
        exp += [
            gxe.ExpectColumnValuesToBeInSet(column="returned_height", value_set=[256]),
            gxe.ExpectColumnValuesToBeInSet(column="returned_width", value_set=[256]),
            gxe.ExpectColumnValuesToBeInSet(column="returned_channels", value_set=[3]),
            gxe.ExpectColumnDistinctValuesToBeInSet(column="returned_dtype", value_set=["uint8"]),
            gxe.ExpectColumnValuesToBeBetween(column="requested_seconds", min_value=0.0),
            gxe.ExpectColumnValuesToBeBetween(column="returned_pts_seconds", min_value=0.0),
            gxe.ExpectColumnValueLengthsToEqual(column="returned_rgb_sha256", value=64),
        ]
        return exp

    if case == "RD8":
        exp = _schema_scaffold({
            "feature_group": "object", "component_index": "int64", "component_name": "object",
            "mean": "float64", "std": "float64", "q01": "float64", "q99": "float64",
        })
        exp += [
            gxe.ExpectColumnValuesToBeBetween(column="std", min_value=0.0),
            gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A="q99", column_B="q01", or_equal=True),
            gxe.ExpectTableRowCountToEqual(value=28),
            gxe.ExpectColumnDistinctValuesToBeInSet(
                column="feature_group", value_set=["actions", "state"]),
        ]
        return exp

    raise KeyError(case)


# ==============================================================================================
# Arm B - mechanical fit on a disjoint reference fixture
# ==============================================================================================
MAX_DISTINCT_FOR_SET = 20


def arm_b(fixture: pd.DataFrame) -> list:
    """The frozen, case-independent generation rule from PROTOCOL.md section 5."""
    exp = [gxe.ExpectTableColumnsToMatchSet(column_set=sorted(fixture.columns))]
    for name in fixture.columns:
        col = fixture[name]
        if not col.isna().any():
            exp.append(gxe.ExpectColumnValuesToNotBeNull(column=name))
        if pd.api.types.is_bool_dtype(col):
            exp.append(gxe.ExpectColumnValuesToBeOfType(column=name, type_="bool"))
        elif pd.api.types.is_integer_dtype(col):
            exp.append(gxe.ExpectColumnValuesToBeOfType(column=name, type_="int64"))
        elif pd.api.types.is_float_dtype(col):
            exp.append(gxe.ExpectColumnValuesToBeOfType(column=name, type_="float64"))
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            finite = col.dropna()
            if len(finite):
                exp.append(gxe.ExpectColumnValuesToBeBetween(
                    column=name, min_value=float(finite.min()), max_value=float(finite.max())))
        else:
            distinct = col.dropna().unique()
            if 0 < len(distinct) <= MAX_DISTINCT_FOR_SET:
                exp.append(gxe.ExpectColumnDistinctValuesToBeInSet(
                    column=name, value_set=sorted(bool(v) if isinstance(v, (bool, np.bool_)) else v
                                                  for v in distinct)))
    return exp


# Pre-declared disjoint splits: (fixture predicate, validated predicate) over a table's rows.
def arm_b_split(case: str, df: pd.DataFrame):
    if case == "RD1":
        return df.index < 59, df.index >= 59
    if case == "RD2":
        return df["episode_index"] <= 2, df["episode_index"] > 2
    if case == "RD4":
        half = len(df) // 2
        return df.index < half, df.index >= half
    if case == "RD5":
        half = int(df["demo_number"].max()) // 2
        return df["demo_number"] <= half, df["demo_number"] > half
    if case == "RD6":
        data = df["row_kind"] == "data"
        half = int(df.loc[data, "source_index"].max()) // 2
        fixture = data & (df["source_index"] <= half)
        return fixture, ~fixture
    if case == "RD7":
        return df["video_index"] <= 4, df["video_index"] > 4
    if case == "RD8":
        # No exchangeable split exists for a 28-record per-component artifact; the fixture is the
        # dataset's own published companion statistics table (see PROTOCOL.md section 5).
        return None, pd.Series(True, index=df.index)
    raise KeyError(case)


# ==============================================================================================
# Arm C - information-enriched suites
# ==============================================================================================
def arm_c(case: str, notes: dict) -> list:
    if case == "RD1":
        return [ExpectWindowSelectionToMatchClampedRequest(column_list=[
            "absolute_index", "delta_index", "selected_index", "is_pad",
            "episode_start", "episode_end"])]
    if case == "RD2":
        return [ExpectGlobalRowIntervalsToChain(column_list=[
            "episode_index", "length", "dataset_from_index", "dataset_to_index"])]
    if case == "RD4":
        return [ExpectQuaternionToMatchSourceUnderDeclaredConventions(
            column_list=["qx", "qy", "qz", "qw", "source_qw", "source_qx", "source_qy", "source_qz"],
            # Original frozen decision boundary, restored. See amendment A3: the boundary is
            # unchanged and the angular evaluation was made numerically stable instead.
            max_error_deg=1e-6)]
    if case == "RD5":
        return [ExpectWorldDeltaToEqualRotatedDisplacement(
            column_list=["delta_x", "delta_y", "delta_z",
                         "world_delta_x", "world_delta_y", "world_delta_z"],
            base_rotation=notes["base_rotation"], base_translation=notes["base_translation"],
            atol=1e-9)]
    if case == "RD6":
        return [ExpectCalibrationAnchorsToMapToEndpoints(
            column_list=["row_kind", "gripper_norm", "expected_norm"], atol=1e-3)]
    if case == "RD7":
        return [ExpectReturnedFrameToMatchRequestedTime(column_list=[
            "returned_pts_seconds", "expected_pts_seconds", "local_frame_interval_seconds",
            "returned_rgb_sha256", "expected_rgb_sha256_set"])]
    if case == "RD8":
        return [ExpectStatisticsToAgreeWithReference(column_list=[
            "mean", "std", "reference_mean", "reference_std", "absolute_tolerance"])]
    raise KeyError(case)
