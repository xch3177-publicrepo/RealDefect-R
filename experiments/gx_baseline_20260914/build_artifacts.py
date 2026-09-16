"""GX-BASE-20260914 step 1: materialise paired faulty/reference artifacts for the GX baseline.

Runs in `.venv-build` (NumPy 1.26.4, h5py 3.11.0, PyArrow 14.0.2, pandas 2.1.4). Produces, for every
eligible case, four Parquet tables under `experiments/work/gx_baseline_20260914/artifacts/`:

    <case>__faulty_public.parquet       <case>__reference_public.parquet
    <case>__faulty_enriched.parquet     <case>__reference_enriched.parquet

`_public` carries only what a downstream consumer obtains from the artifact itself. `_enriched` adds
the case's reference-information columns. Every table is SHA-256 hashed into
`artifact_manifest.json`, which the validation runner asserts before validating.

No frozen result file, historical archive, or restored original source is modified. Row-level content
comes from the archived case results where those already contain it, and from re-executing the pinned
per-case transform code on hash-verified re-downloaded inputs where they do not (RD4, RD5).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = REPO / "reproducibility" / "results"
WORK = REPO / "experiments" / "work" / "gx_baseline_20260914"
INPUTS = WORK / "inputs"
OUT = WORK / "artifacts"
RD8_PROJ = REPO / "experiments" / "work" / "rd8_diagnostic_20260914" / "projections"
RD8_META = REPO / "experiments" / "work" / "rd8_diagnostic_20260914" / "official-metadata"
ORIGINAL_SRC = REPO / "original" / "src"

sys.path.insert(0, str(ORIGINAL_SRC))

PINNED_INPUT_SHA = {
    "franka_stack_v51.hdf5": "e0954647328801b091b81604915acd4eff2093b2b26ff0b05753c8f85fbb61b3",
    "hdf5_dataset_file_handler_beta2_patch1.py": "b5c4365cfd46107dfe56288b092bbdb03d993dbb83e23415742965db7df8d793",
    "low_dim_v15.hdf5": "0d302b37fb08e99b00de287b833925589ad3b65d199306e55562938bcbd746f1",
}

MANIFEST: dict = {"cases": {}, "inputs": {}, "provenance": {}}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_result(name: str) -> dict:
    path = RESULTS / name
    MANIFEST["provenance"][str(path.relative_to(REPO))] = sha256_file(path)
    return json.loads(path.read_bytes())


def emit(case: str, tables: dict[str, pd.DataFrame], notes: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    entry = {"notes": notes, "tables": {}}
    for kind, df in tables.items():
        path = OUT / f"{case}__{kind}.parquet"
        df.to_parquet(path, index=False)
        entry["tables"][kind] = {
            "path": str(path.relative_to(REPO)),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sha256": sha256_file(path),
        }
    MANIFEST["cases"][case] = entry
    print(f"{case}: " + ", ".join(f"{k}={len(v)}r" for k, v in tables.items()))


# ----------------------------------------------------------------------------------------------
# RD1 - LeRobot episode-filtered temporal windows
# ----------------------------------------------------------------------------------------------
def build_rd1() -> None:
    doc = load_result("real-defect-lerobot-2610.json")
    rows, summary, frozen = doc["rows"], doc["summary"], doc["frozen_inputs"]
    start, end = summary["episode_start"], summary["episode_end"]

    # Declared dataset length, taken from the same pinned PushT episode metadata file the case used.
    meta = REPO / "reproducibility/light_inputs/data/raw/lerobot/pusht/meta/episodes/chunk-000/file-000.parquet"
    assert sha256_file(meta) == frozen["metadata_sha256"], "PushT metadata hash mismatch"
    MANIFEST["inputs"][str(meta.relative_to(REPO))] = frozen["metadata_sha256"]
    episodes = pd.read_parquet(meta)
    declared_total_frames = int(episodes["length"].sum())

    def table(branch: str, enriched: bool) -> pd.DataFrame:
        df = pd.DataFrame({
            "absolute_index": np.array([r["absolute_index"] for r in rows], dtype="int64"),
            "delta_index": np.zeros(len(rows), dtype="int64"),
            "selected_index": np.array([r[f"{branch}_query_index"] for r in rows], dtype="int64"),
            "is_pad": np.array([r[f"{branch}_is_pad"] for r in rows], dtype="bool"),
        })
        if enriched:
            req = df["absolute_index"] + df["delta_index"]
            df["episode_start"] = np.int64(start)
            df["episode_end"] = np.int64(end)
            df["expected_selected_index"] = np.clip(req, start, end - 1).astype("int64")
            df["expected_is_pad"] = ((req < start) | (req >= end)).astype("bool")
        return df

    emit("RD1", {
        "faulty_public": table("buggy", False),
        "reference_public": table("fixed", False),
        "faulty_enriched": table("buggy", True),
        "reference_enriched": table("fixed", True),
    }, {
        "artifact": "windowed-sample selection output for PushT episode 1",
        "declared_total_frames": declared_total_frames,
        "declared_episode_bounds": [start, end],
        "reference_information": "episode row bounds from meta/episodes; clamp-and-pad rule",
        "source": "row-level records archived in real-defect-lerobot-2610.json",
    })


# ----------------------------------------------------------------------------------------------
# RD2 - LeRobot v2.1 -> v3 global continuity
# ----------------------------------------------------------------------------------------------
def build_rd2() -> None:
    doc = load_result("real-defect-lerobot-2057.json")

    def table(rows: list[dict], enriched: bool) -> pd.DataFrame:
        df = pd.DataFrame({
            "episode_index": np.array([r["episode_index"] for r in rows], dtype="int64"),
            "length": np.array([r["length"] for r in rows], dtype="int64"),
            "dataset_from_index": np.array([r["dataset_from_index"] for r in rows], dtype="int64"),
            "dataset_to_index": np.array([r["dataset_to_index"] for r in rows], dtype="int64"),
        })
        if enriched:
            # Reference information: the global row counter is a prefix sum over episode lengths.
            prefix = np.concatenate([[0], np.cumsum(df["length"].to_numpy())[:-1]])
            df["expected_from_index"] = prefix.astype("int64")
            df["expected_to_index"] = (prefix + df["length"].to_numpy()).astype("int64")
        return df

    emit("RD2", {
        "faulty_public": table(doc["buggy_rows"], False),
        "reference_public": table(doc["fixed_rows"], False),
        "faulty_enriched": table(doc["buggy_rows"], True),
        "reference_enriched": table(doc["fixed_rows"], True),
    }, {
        "artifact": "meta/episodes global row-interval fields across an output-file boundary",
        "reference_information": "global prefix sum over episode lengths; intervals must chain",
        "source": "row-level records archived in real-defect-lerobot-2057.json",
    })


# ----------------------------------------------------------------------------------------------
# RD4 - Isaac Lab quaternion migration coverage
# ----------------------------------------------------------------------------------------------
def build_rd4() -> None:
    from robocontract.isaaclab_migration import (  # noqa: E402
        collect_obligations, physical_errors_deg, run_frozen_upstream_converter,
    )

    source = INPUTS / "franka_stack_v51.hdf5"
    converter = INPUTS / "hdf5_dataset_file_handler_beta2_patch1.py"
    for p in (source, converter):
        got = sha256_file(p)
        assert got == PINNED_INPUT_SHA[p.name], f"{p.name} hash mismatch: {got}"
        MANIFEST["inputs"][str(p.relative_to(REPO))] = got

    target = WORK / "rd4_native_output.hdf5"
    target.parent.mkdir(parents=True, exist_ok=True)
    run_frozen_upstream_converter(converter, source, target)

    obligations = collect_obligations(source, target)
    kinds, paths, idx, src, tgt = [], [], [], [], []
    for ob in obligations:
        n = np.asarray(ob.source_wxyz).shape[0]
        kinds += [ob.family] * n
        paths += [ob.path] * n
        idx += list(range(n))
        src.append(np.asarray(ob.source_wxyz, dtype="f8"))
        tgt.append(np.asarray(ob.target_xyzw, dtype="f8"))
    src_arr = np.concatenate(src, axis=0)
    tgt_arr = np.concatenate(tgt, axis=0)

    # Reference (fully converted) target: the declared WXYZ -> XYZW permutation of every source tuple.
    ref_arr = src_arr[:, [1, 2, 3, 0]]

    def table(values: np.ndarray, enriched: bool) -> pd.DataFrame:
        df = pd.DataFrame({
            "field_kind": pd.Series(kinds, dtype="object"),
            "field_path": pd.Series(paths, dtype="object"),
            "instance_index": np.array(idx, dtype="int64"),
            "qx": values[:, 0], "qy": values[:, 1], "qz": values[:, 2], "qw": values[:, 3],
        })
        df["norm"] = np.linalg.norm(values, axis=1)
        if enriched:
            df["source_qw"] = src_arr[:, 0]
            df["source_qx"] = src_arr[:, 1]
            df["source_qy"] = src_arr[:, 2]
            df["source_qz"] = src_arr[:, 3]
            df["source_convention"] = pd.Series(["wxyz"] * len(df), dtype="object")
            df["target_convention"] = pd.Series(["xyzw"] * len(df), dtype="object")
            df["orientation_error_deg"] = physical_errors_deg(src_arr, values)
        return df

    faulty = table(tgt_arr, False)
    reference = table(ref_arr, False)
    unconverted = int(np.sum(np.all(np.isclose(tgt_arr, src_arr, atol=1e-7), axis=1)))
    emit("RD4", {
        "faulty_public": faulty,
        "reference_public": reference,
        "faulty_enriched": table(tgt_arr, True),
        "reference_enriched": table(ref_arr, True),
    }, {
        "artifact": "quaternion instances stored by the pinned upstream converter body",
        "declared_instances": int(len(faulty)),
        "instances_equal_to_source_coefficients": unconverted,
        "reference_information": "source tuple plus declared source/target conventions",
        "source": "re-executed pinned upstream converter body on the hash-verified official HDF5",
    })


# ----------------------------------------------------------------------------------------------
# RD5 - robosuite Mink delta-action reference frame
# ----------------------------------------------------------------------------------------------
def build_rd5() -> None:
    import h5py
    from robocontract.rigid_transform import (  # noqa: E402
        rotation_matrix_from_wxyz, transform_delta_as_absolute_pose_bug, transform_delta_correct,
    )

    dataset = INPUTS / "low_dim_v15.hdf5"
    got = sha256_file(dataset)
    assert got == PINNED_INPUT_SHA[dataset.name], f"robomimic hash mismatch: {got}"
    MANIFEST["inputs"][str(dataset.relative_to(REPO))] = got

    def base_transform(model_xml: str):
        root = ET.fromstring(model_xml)
        body = next(e for e in root.iter("body") if e.attrib.get("name") == "robot0_base")
        t = [float(v) for v in body.attrib.get("pos", "0 0 0").split()]
        q = [float(v) for v in body.attrib.get("quat", "1 0 0 0").split()]
        return np.asarray(t), np.asarray(rotation_matrix_from_wxyz(q))

    recs = []
    with h5py.File(dataset, "r") as handle:
        env_args = json.loads(str(handle["data"].attrs["env_args"]))
        ctl = env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]
        in_min = np.broadcast_to(np.asarray(ctl["input_min"], dtype="f8"), (3,))
        in_max = np.broadcast_to(np.asarray(ctl["input_max"], dtype="f8"), (3,))
        out_min = np.asarray(ctl["output_min"], dtype="f8")[:3]
        out_max = np.asarray(ctl["output_max"], dtype="f8")[:3]
        for demo_id in sorted(handle["data"].keys(), key=lambda v: int(v.rsplit("_", 1)[1])):
            demo = handle[f"data/{demo_id}"]
            raw = np.asarray(demo["actions"], dtype="f8")[:, :3]
            frac = (np.clip(raw, in_min, in_max) - in_min) / (in_max - in_min)
            deltas = out_min + frac * (out_max - out_min)
            t, R = base_transform(str(demo.attrs["model_file"]))
            for step, d in enumerate(deltas):
                recs.append({
                    "demo_id": demo_id, "demo_number": int(demo_id.rsplit("_", 1)[1]), "step": step,
                    "action_x": raw[step, 0], "action_y": raw[step, 1], "action_z": raw[step, 2],
                    "delta_x": d[0], "delta_y": d[1], "delta_z": d[2],
                    "faulty": transform_delta_as_absolute_pose_bug(d, t, R),
                    "reference": transform_delta_correct(d, R),
                    "R": R, "t": t,
                })

    declared_max_norm = float(np.linalg.norm(np.maximum(np.abs(out_min), np.abs(out_max))))

    def table(branch: str, enriched: bool) -> pd.DataFrame:
        w = np.asarray([r[branch] for r in recs], dtype="f8")
        df = pd.DataFrame({
            "demo_id": pd.Series([r["demo_id"] for r in recs], dtype="object"),
            "demo_number": np.array([r["demo_number"] for r in recs], dtype="int64"),
            "step": np.array([r["step"] for r in recs], dtype="int64"),
            "action_x": [r["action_x"] for r in recs],
            "action_y": [r["action_y"] for r in recs],
            "action_z": [r["action_z"] for r in recs],
            "world_delta_x": w[:, 0], "world_delta_y": w[:, 1], "world_delta_z": w[:, 2],
        })
        df["world_delta_norm_m"] = np.linalg.norm(w, axis=1)
        if enriched:
            # Reference information: the commanded displacement in the base frame, which the check
            # needs in order to distinguish a displacement from an absolute position.
            df["delta_x"] = [r["delta_x"] for r in recs]
            df["delta_y"] = [r["delta_y"] for r in recs]
            df["delta_z"] = [r["delta_z"] for r in recs]
        return df

    emit("RD5", {
        "faulty_public": table("faulty", False),
        "reference_public": table("reference", False),
        "faulty_enriched": table("faulty", True),
        "reference_enriched": table("reference", True),
    }, {
        "artifact": "world-frame displacement produced by the Mink delta-action branch",
        "declared_controller_output_min": out_min.tolist(),
        "declared_controller_output_max": out_max.tolist(),
        "declared_max_displacement_norm_m": declared_max_norm,
        "base_rotation": np.asarray(recs[0]["R"], dtype="f8").reshape(-1).tolist(),
        "base_translation": np.asarray(recs[0]["t"], dtype="f8").tolist(),
        "base_transform_constant_across_demos": bool(
            all(np.array_equal(r["R"], recs[0]["R"]) and np.array_equal(r["t"], recs[0]["t"])
                for r in recs)),
        "reference_information": "base transform (R,t) and the absolute-vs-displacement distinction",
        "source": "re-executed pinned transform functions on the hash-verified robomimic Lift HDF5",
    })


# ----------------------------------------------------------------------------------------------
# RD6 - OpenPI ALOHA gripper calibration (DERIVED target, outside RD6's frozen replay scope)
# ----------------------------------------------------------------------------------------------
def _load_pinned_gripper_transform(pinned_name: str):
    """Extract the pinned `_normalize`/`_unnormalize`/`_gripper_to_angular` bodies verbatim.

    The pinned file is the full upstream ALOHA policy module and imports `einops`/`openpi`, which are
    not installed here. Rather than substituting our own arithmetic, we parse the hash-verified
    pinned source and execute only those three self-contained pure functions with NumPy in scope.
    Nothing in their bodies is edited; the surrounding module's unrelated imports are simply not run.
    """
    import ast

    path = REPO / "reproducibility/light_inputs/upstream_pinned_sources" / f"{pinned_name}.py"
    MANIFEST["inputs"][str(path.relative_to(REPO))] = sha256_file(path)
    source = path.read_text()
    tree = ast.parse(source)
    wanted = {"_normalize", "_unnormalize", "_gripper_to_angular"}
    picked = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in picked} == wanted, f"pinned functions missing in {pinned_name}"
    namespace: dict = {"np": np}
    exec(compile(ast.Module(body=picked, type_ignores=[]), str(path), "exec"), namespace)
    bodies = {n.name: ast.get_source_segment(source, n) for n in picked}
    return namespace["_gripper_to_angular"], bodies


def build_rd6() -> None:
    doc = load_result("real-defect-openpi-557.json")
    frozen = doc["frozen_inputs"]
    counts = frozen["encoder_counts"]
    zero, cpr = frozen["encoder_zero"], frozen["counts_per_revolution"]
    anchor_rad = [(c - zero) / cpr * 2.0 * np.pi for c in counts]
    anchor_expected = [0.0, 1.0]
    old_bounds, fixed_bounds = (0.4, 1.5), (0.5476, 1.6296)

    old_fn, old_bodies = _load_pinned_gripper_transform("RD6_old")
    fixed_fn, fixed_bodies = _load_pinned_gripper_transform("RD6_fixed")
    branch_fn = {"old": old_fn, "fixed": fixed_fn}
    # The two pinned variants must differ only in the declared normalisation bounds.
    assert old_bodies["_normalize"] == fixed_bodies["_normalize"]
    assert "0.4, max_val=1.5" in old_bodies["_gripper_to_angular"]
    assert "0.5476, max_val=1.6296" in fixed_bodies["_gripper_to_angular"]

    names = json.loads((RESULTS / "real-defect-openpi-570.json").read_bytes())["projection"]["feature_names"]
    gi = {"right_gripper": names.index("right_gripper"), "left_gripper": names.index("left_gripper")}

    files = sorted(RD8_PROJ.glob("*.npz"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    assert len(files) == 123, "RD6 derived target needs the 123 verified RD8 projections"
    side, src_idx, rad = [], [], []
    row0 = 0
    for path in files:
        with np.load(path) as z:
            st = z["state"]
        for name, col in gi.items():
            side += [name] * st.shape[0]
            src_idx += list(range(row0, row0 + st.shape[0]))
            rad.append(np.asarray(st[:, col], dtype="f8"))
        row0 += st.shape[0]
    raw_arr = np.concatenate(rad, axis=0)

    def table(branch: str, enriched: bool) -> pd.DataFrame:
        fn = branch_fn[branch]
        lo, hi = (old_bounds if branch == "old" else fixed_bounds)
        df = pd.DataFrame({
            "row_kind": pd.Series(["data"] * len(raw_arr), dtype="object"),
            "side": pd.Series(side, dtype="object"),
            "source_index": np.array(src_idx, dtype="int64"),
            "gripper_raw": raw_arr,
            "gripper_norm": np.asarray(fn(raw_arr), dtype="f8"),
        })
        # The two calibration anchors are already angular, so only the declared bound normalisation
        # of the pinned branch applies to them.
        anchors = pd.DataFrame({
            "row_kind": pd.Series(["anchor", "anchor"], dtype="object"),
            "side": pd.Series(["calibration", "calibration"], dtype="object"),
            "source_index": np.array([-1, -2], dtype="int64"),
            "gripper_raw": np.asarray(anchor_rad, dtype="f8"),
            "gripper_norm": (np.asarray(anchor_rad, dtype="f8") - lo) / (hi - lo),
        })
        df = pd.concat([df, anchors], ignore_index=True)
        if enriched:
            df["expected_norm"] = np.concatenate([
                np.full(len(raw_arr), np.nan), np.asarray(anchor_expected, dtype="f8")])
            df["anchor_encoder_counts"] = np.concatenate([
                np.full(len(raw_arr), np.nan), np.asarray(counts, dtype="f8")])
        return df

    emit("RD6", {
        "faulty_public": table("old", False),
        "reference_public": table("fixed", False),
        "faulty_enriched": table("old", True),
        "reference_enriched": table("fixed", True),
    }, {
        "artifact": "DERIVED normalised gripper column over 36,900 pinned rows plus two anchor rows",
        "transform": (
            "pinned upstream _gripper_to_angular bodies extracted verbatim from the hash-verified "
            "RD6_old.py / RD6_fixed.py and executed with NumPy"
        ),
        "declared_output_range": "[0,1]; both pinned branches document the gripper output as normalised to 0-1",
        "derived_target_disclaimer": (
            "not part of RD6's frozen replay scope; RD6's recorded result is unchanged"
        ),
        "old_bounds_rad": list(old_bounds),
        "fixed_bounds_rad": list(fixed_bounds),
        "anchor_encoder_counts": counts,
        "anchor_rad": anchor_rad,
        "anchor_expected_norm": anchor_expected,
        "reference_information": "documented encoder anchors and their calibrated endpoints",
    })


# ----------------------------------------------------------------------------------------------
# RD7 - Isaac-GR00T requested-time frame selection
# ----------------------------------------------------------------------------------------------
def build_rd7() -> None:
    doc = load_result("real-defect-groot-172.json")

    def table(branch: str, enriched: bool) -> pd.DataFrame:
        recs = []
        for video in doc["videos"]:
            for group, payload in video["branches"][branch].items():
                for q in payload["queries"]:
                    actual, oracle = q["actual"], q["oracle"]
                    shape = actual["shape"]
                    rec = {
                        "video_index": video["video_index"],
                        "group": group,
                        "query_index": q["query_index"],
                        "requested_seconds": float(q["target"]["seconds"]),
                        "returned_pts_seconds": float(actual["pts_seconds"]),
                        "returned_height": int(shape[0]),
                        "returned_width": int(shape[1]),
                        "returned_channels": int(shape[2]),
                        "returned_dtype": actual["dtype"],
                        "returned_rgb_sha256": actual["rgb_sha256"],
                    }
                    if enriched:
                        rec["expected_pts_seconds"] = float(oracle["selected_pts"]["seconds"])
                        rec["expected_rgb_sha256_set"] = "|".join(oracle["equivalent_rgb_sha256"])
                        rec["local_frame_interval_seconds"] = float(oracle["local_frame_interval_seconds"])
                    recs.append(rec)
        df = pd.DataFrame(recs).sort_values(["video_index", "group", "query_index"]).reset_index(drop=True)
        return df

    emit("RD7", {
        "faulty_public": table("old", False),
        "reference_public": table("fixed", False),
        "faulty_enriched": table("old", True),
        "reference_enriched": table("fixed", True),
    }, {
        "artifact": "per-request returned-frame identity table across ten pinned videos",
        "reference_information": "requested-time correspondence: expected frame PTS and RGB identity",
        "source": "per-query records archived in real-defect-groot-172.json",
    })


# ----------------------------------------------------------------------------------------------
# RD8 - OpenPI normalisation statistics artifact
# ----------------------------------------------------------------------------------------------
def build_rd8() -> None:
    doc = load_result("real-defect-openpi-570.json")
    names = doc["projection"]["feature_names"]
    primary = doc["primary_condition"]
    cond = next(c for c in doc["conditions"]
                if c["order_name"] == primary["order_name"] and c["batch_size"] == primary["batch_size"])

    def table(branch: str, enriched: bool) -> pd.DataFrame:
        recs = []
        for group in ("state", "actions"):
            st = cond["branches"][branch][group]["statistics"]
            ref = doc["oracle"][group]
            for i, name in enumerate(names):
                rec = {
                    "feature_group": group, "component_index": i, "component_name": name,
                    "mean": float(st["mean"][i]), "std": float(st["std"][i]),
                    "q01": float(st["q01"][i]), "q99": float(st["q99"][i]),
                }
                if enriched:
                    rec["reference_mean"] = float(ref["mean"][i])
                    rec["reference_std"] = float(ref["std"][i])
                    rec["absolute_tolerance"] = 1e-5
                recs.append(rec)
        return pd.DataFrame(recs)

    # Arm B fitting fixture: the dataset's own published companion statistics.
    stats_path = RD8_META / "stats.json"
    published = json.loads(stats_path.read_bytes())
    MANIFEST["inputs"][str(stats_path.relative_to(REPO))] = sha256_file(stats_path)
    key = {"state": "observation.state", "actions": "action"}
    pub_recs = []
    for group, pkey in key.items():
        entry = published[pkey]
        for i, name in enumerate(names):
            pub_recs.append({
                "feature_group": group, "component_index": i, "component_name": name,
                "mean": float(entry["mean"][i]), "std": float(entry["std"][i]),
                "q01": float(entry["q01"][i]) if "q01" in entry else float("nan"),
                "q99": float(entry["q99"][i]) if "q99" in entry else float("nan"),
            })
    published_df = pd.DataFrame(pub_recs)

    emit("RD8", {
        "faulty_public": table("old", False),
        "reference_public": table("fixed", False),
        "faulty_enriched": table("old", True),
        "reference_enriched": table("fixed", True),
        "armb_fixture": published_df,
    }, {
        "artifact": "norm_stats per-component statistics at the primary condition",
        "primary_condition": {"order_name": primary["order_name"], "batch_size": primary["batch_size"]},
        "published_artifact_has_no_count_field": True,
        "armb_fixture_source": "dataset-published meta/stats.json recovered from the frozen evidence archive",
        "reference_information": "independently computed float64 full-array mean/std and the retained 1e-5 tolerance",
        "source": "condition records archived in real-defect-openpi-570.json",
    })


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (build_rd1, build_rd2, build_rd4, build_rd5, build_rd6, build_rd7, build_rd8):
        fn()
    MANIFEST["generated_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST["environment"] = {
        "python_version": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    MANIFEST["excluded_cases"] = {
        "RD3": "NA - single 4-element configuration constant, not a multi-record data artifact; "
               "its population analogue is RD4"
    }
    (HERE / "artifact_manifest.json").write_text(json.dumps(MANIFEST, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {(HERE / 'artifact_manifest.json').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
