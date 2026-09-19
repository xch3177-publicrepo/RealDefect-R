#!/usr/bin/env python3
"""Derive source-prefix chart data from pinned Cosmos episode metadata.

This analysis does not identify institutions, execute a consumer, or reread
trajectory/video data. It preserves the original seven identity predicates and
independently checks whether their values match source-prefix-local numbering.
Existing frozen evidence is read only. Each run requires a new output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REVISION = "dabaaffe428d67cf93fd355b82658934ee59bfec"
IDENTITY_FIELDS = [
    "stats/episode_index/min", "stats/episode_index/max",
    "stats/episode_index/mean", "stats/episode_index/std",
    "stats/index/min", "stats/index/max", "stats/index/mean",
]
BASE_FIELDS = [
    "episode_index", "episode_id", "tasks", "length",
    "dataset_from_index", "dataset_to_index",
]
STAT_FIELDS = IDENTITY_FIELDS + ["stats/task_index/min"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def close(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b) <= np.maximum(1e-5, 1e-7 * np.maximum(np.abs(a), np.abs(b)))


def identity_expectations(episode, start, end):
    return {
        "stats/episode_index/min": episode,
        "stats/episode_index/max": episode,
        "stats/episode_index/mean": episode,
        "stats/episode_index/std": np.zeros(len(episode)),
        "stats/index/min": start,
        "stats/index/max": end - 1,
        "stats/index/mean": (start + end - 1) / 2,
    }


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/results/20260912-cosmos-metadata/verified_inputs.json")
    parser.add_argument("--reference", type=Path, default=ROOT / "experiments/results/20260912-cosmos-metadata/independent_referential_check.json")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory to preserve prior runs")
    args.output_dir.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    manifest = json.loads(args.manifest.read_text())
    reference = json.loads(args.reference.read_text())
    protocol = {
        "analysis_version": "source-prefix-identity-v1",
        "frozen_utc": started,
        "source_revision": REVISION,
        "script_sha256": sha256(Path(__file__)),
        "input_manifest_sha256": sha256(args.manifest),
        "reference_result_sha256": sha256(args.reference),
        "inputs": [{k: entry[k] for k in ("path", "remote_path", "sha256", "size_bytes")} for entry in manifest["files"]],
        "group_definition": "Substring of episode_id before the first slash; source prefix only, not a verified institution identifier.",
        "order_definition": "Sort complete split by episode_index; reset episode and cumulative frame counts within each source prefix.",
        "global_identity_fields": IDENTITY_FIELDS,
        "criterion": "abs(actual-expected) <= max(1e-5, 1e-7 * max(abs(actual),abs(expected))) for each scalar; discrepancy is any failed field.",
        "hypothesis_checks": [
            "All seven identity summary fields equal source-prefix-local predictions within the frozen scalar tolerance.",
            "Identity-coherent rows are exactly those whose local episode and frame starts equal global values.",
            "The source groups and existing global/task discrepancy counts equal the frozen independent audit.",
        ],
        "limits": "Metadata-only deterministic census; no source-institution attribution, new consumer replay, downstream harm, or upstream causal claim.",
    }
    save(args.output_dir / "PROTOCOL.json", protocol)
    verified = []
    for entry in protocol["inputs"]:
        path = args.data_root / entry["path"]
        actual = sha256(path)
        assert actual == entry["sha256"], entry["path"]
        assert path.stat().st_size == entry["size_bytes"], entry["path"]
        verified.append({"path": entry["path"], "sha256": actual, "size_bytes": path.stat().st_size})
        print("VERIFIED", entry["path"], actual, flush=True)
    save(args.output_dir / "VERIFIED_INPUTS.json", {"all_match": True, "files": verified})

    split_results, split_rows, per_row_tables = {}, [], []
    for split in ("success", "failure"):
        shards = sorted((args.data_root / split / "episodes").glob("*.parquet"))
        expected_shards = {entry["path"] for entry in protocol["inputs"] if entry["path"].startswith(split + "/episodes/")}
        assert {str(path.relative_to(args.data_root)) for path in shards} == expected_shards
        values = {field: [] for field in BASE_FIELDS + STAT_FIELDS}
        for shard in shards:
            table = pq.read_table(shard, columns=list(values)).to_pydict()
            for field in values:
                values[field].extend(table[field])
        frame = pd.DataFrame({
            key: ([float(value[0]) for value in column] if key in STAT_FIELDS else column)
            for key, column in values.items()
        }).sort_values("episode_index").reset_index(drop=True)
        assert np.array_equal(frame.episode_index, np.arange(len(frame)))
        frame["source_prefix"] = frame.episode_id.str.split("/", n=1).str[0]
        assert frame.source_prefix.notna().all() and (frame.source_prefix.str.len() > 0).all()
        groups = frame.groupby("source_prefix", sort=False)
        local_episode = groups.cumcount()
        local_start = groups.length.cumsum() - frame.length
        local_end = local_start + frame.length
        expected_global = identity_expectations(frame.episode_index, frame.dataset_from_index, frame.dataset_to_index)
        expected_local = identity_expectations(local_episode, local_start, local_end)
        global_matches = {field: close(frame[field], predicted) for field, predicted in expected_global.items()}
        local_matches = {field: close(frame[field], predicted) for field, predicted in expected_local.items()}
        frame["identity_bad"] = ~np.logical_and.reduce(list(global_matches.values()))
        frame["local_summary_matches"] = np.logical_and.reduce(list(local_matches.values()))
        frame["numbering_coincides"] = (frame.episode_index == local_episode) & (frame.dataset_from_index == local_start)
        frame["episode_offset"] = frame.episode_index - local_episode
        frame["frame_offset"] = frame.dataset_from_index - local_start

        tasks = pd.DataFrame(pq.read_table(args.data_root / split / "tasks.parquet", columns=["task_index", "task"]).to_pydict())
        assert tasks.task_index.is_unique
        task_lookup = tasks.set_index("task_index").task.to_dict()
        task_index = np.rint(frame["stats/task_index/min"]).astype(int)
        task_text = frame.tasks.apply(lambda row: str(row[0]) if len(row) == 1 else "")
        frame["task_bad"] = task_text != task_index.map(task_lookup)
        frame["identity_only"] = frame.identity_bad & ~frame.task_bad
        frame["task_only"] = ~frame.identity_bad & frame.task_bad
        frame["both_bad"] = frame.identity_bad & frame.task_bad
        frame["neither_bad"] = ~frame.identity_bad & ~frame.task_bad
        count_names = ["identity_bad", "task_bad", "identity_only", "task_only", "both_bad", "neither_bad"]
        counts = {key: int(frame[key].sum()) for key in count_names}
        assert len(frame) == reference["splits"][split]["episode_rows"]
        assert all(value == reference["splits"][split]["counts"][key] for key, value in counts.items())
        assert set(frame.source_prefix) == set(reference["splits"][split]["by_lab"])
        for prefix, group in frame.groupby("source_prefix", sort=True):
            count = len(group)
            identity_bad = int(group.identity_bad.sum())
            row = {
                "split": split, "source_prefix": prefix,
                "episode_rows": count, "identity_discrepant": identity_bad,
                "identity_discrepancy_percent": 100 * identity_bad / count,
                "task_discrepant": int(group.task_bad.sum()),
                "both_discrepant": int(group.both_bad.sum()),
                "identity_only": int(group.identity_only.sum()),
                "task_only": int(group.task_only.sum()),
                "neither_discrepant": int(group.neither_bad.sum()),
                "local_summary_matches": int(group.local_summary_matches.sum()),
                "local_global_coincident_rows": int(group.numbering_coincides.sum()),
                "episode_offset_min": int(group.episode_offset.min()),
                "episode_offset_max": int(group.episode_offset.max()),
                "frame_offset_min": int(group.frame_offset.min()),
                "frame_offset_max": int(group.frame_offset.max()),
            }
            prior = reference["splits"][split]["by_lab"][prefix]
            assert count == prior["episode_rows"]
            assert identity_bad == prior["identity_bad"]
            assert row["task_discrepant"] == prior["task_bad"]
            split_rows.append(row)
        exact_coincidence_relation = np.array_equal(~frame.identity_bad, frame.numbering_coincides)
        split_results[split] = {
            "episode_rows": len(frame), "source_prefix_count": int(frame.source_prefix.nunique()),
            "source_prefixes": sorted(frame.source_prefix.unique()), "counts": counts,
            "missing_prefix_rows": 0,
            "global_summary_field_violations": {field: int((~mask).sum()) for field, mask in global_matches.items()},
            "local_summary_field_violations": {field: int((~mask).sum()) for field, mask in local_matches.items()},
            "all_seven_local_summary_fields_match_rows": int(frame.local_summary_matches.sum()),
            "local_global_numbering_coincides_rows": int(frame.numbering_coincides.sum()),
            "identity_coherent_iff_numbering_coincides": bool(exact_coincidence_relation),
            "coherent_prefixes": sorted(frame.loc[~frame.identity_bad, "source_prefix"].unique()),
            "global_and_per_group_counts_match_frozen_audit": True,
        }
        per_row_tables.append(frame)
        print("SPLIT", split, json.dumps(split_results[split], sort_keys=True), flush=True)
    combined = pd.concat(per_row_tables, ignore_index=True)
    combined_rows = []
    for prefix, group in combined.groupby("source_prefix", sort=True):
        n = len(group)
        bad = int(group.identity_bad.sum())
        combined_rows.append({
            "source_prefix": prefix, "episode_rows": n, "identity_discrepant": bad,
            "identity_discrepancy_percent": 100 * bad / n,
            "task_discrepant": int(group.task_bad.sum()),
            "both_discrepant": int(group.both_bad.sum()),
            "identity_only": int(group.identity_only.sum()),
            "task_only": int(group.task_only.sum()),
            "neither_discrepant": int(group.neither_bad.sum()),
            "local_summary_matches": int(group.local_summary_matches.sum()),
            "local_global_coincident_rows": int(group.numbering_coincides.sum()),
        })
    combined_rows.sort(key=lambda row: (-row["identity_discrepancy_percent"], -row["episode_rows"], row["source_prefix"]))
    write_csv(args.output_dir / "by_split_source_prefix.csv", split_rows)
    write_csv(args.output_dir / "by_source_prefix.csv", combined_rows)
    results = {
        "analysis_version": protocol["analysis_version"],
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_revision": REVISION,
        "protocol_sha256": sha256(args.output_dir / "PROTOCOL.json"),
        "script_sha256": sha256(Path(__file__)), "splits": split_results,
        "combined": {
            "episode_rows": len(combined), "source_prefix_count": int(combined.source_prefix.nunique()),
            "identity_discrepant": int(combined.identity_bad.sum()),
            "identity_coherent": int((~combined.identity_bad).sum()),
            "local_summary_matches": int(combined.local_summary_matches.sum()),
            "local_global_coincident_rows": int(combined.numbering_coincides.sum()),
            "coherent_prefixes": sorted(combined.loc[~combined.identity_bad, "source_prefix"].unique()),
            "identity_coherent_iff_numbering_coincides": bool(np.array_equal(~combined.identity_bad, combined.numbering_coincides)),
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
        "interpretation": "Observed metadata values match source-prefix-local numbering. This supports a localized release-level numbering discrepancy; it does not identify the upstream implementation that introduced it or establish institution-level prevalence.",
    }
    save(args.output_dir / "results.json", results)
    save(args.output_dir / "MANIFEST.json", {
        "files": [{"path": path.name, "sha256": sha256(path), "size_bytes": path.stat().st_size} for path in sorted(args.output_dir.iterdir()) if path.is_file()],
    })
    print("COMPLETE", json.dumps(results["combined"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
