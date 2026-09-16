#!/usr/bin/env python3
"""Independent vector/table calculation of the Cosmos identity/task findings.

This checker imports no frozen audit code. It validates observed metadata
relationships, not an upstream causal diagnosis or raw trajectory validity.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def close(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    # The documented scalar rule uses max(relative tolerance, absolute tolerance).
    return np.abs(a - b) <= np.maximum(1e-5, 1e-7 * np.maximum(np.abs(a), np.abs(b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "experiments/work/cosmos3-droid")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/results/20260912-cosmos-metadata/verified_inputs.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/20260912-cosmos-metadata/independent_referential_check.json")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    started = datetime.now(timezone.utc).isoformat()
    files = json.loads(args.manifest.read_text())["files"]
    for entry in files:
        p = args.data_root / entry["path"]
        assert p.stat().st_size == entry["size_bytes"] and digest(p) == entry["sha256"], p
    base = ["episode_index", "episode_id", "tasks", "length", "dataset_from_index", "dataset_to_index"]
    stats = [f"stats/{feature}/{metric}" for feature, metrics in [("episode_index", ["min", "max", "mean", "std"]), ("index", ["min", "max", "mean"]), ("task_index", ["min", "max", "std"])] for metric in metrics]
    results = {}
    for split in ("success", "failure"):
        paths = sorted((args.data_root / split / "episodes").glob("*.parquet"))
        columns = {key: [] for key in base + stats}
        for path in paths:
            values = pq.read_table(path, columns=base + stats).to_pydict()
            for key in columns:
                columns[key].extend(values[key])
        frame = pd.DataFrame({key: ([float(x[0]) for x in value] if key in stats else value) for key, value in columns.items()}).sort_values("episode_index").reset_index(drop=True)
        frame["lab"] = frame["episode_id"].str.split("/", n=1).str[0]
        frame["task_text"] = frame["tasks"].apply(lambda x: str(x[0]) if len(x) == 1 else "")
        expected = {"stats/episode_index/min": frame.episode_index, "stats/episode_index/max": frame.episode_index, "stats/episode_index/mean": frame.episode_index, "stats/episode_index/std": 0., "stats/index/min": frame.dataset_from_index, "stats/index/max": frame.dataset_to_index - 1, "stats/index/mean": (frame.dataset_from_index + frame.dataset_to_index - 1) / 2}
        frame["identity_bad"] = np.logical_or.reduce([~close(frame[k], v) for k, v in expected.items()])
        # Preserve physical Parquet columns: to_pandas() restores a stored
        # pandas index and can hide the task text as an index level.
        tasks = pd.DataFrame(pq.read_table(args.data_root / split / "tasks.parquet", columns=["task_index", "task"]).to_pydict())
        assert tasks.task_index.is_unique
        frame["lookup_task_index"] = np.rint(frame["stats/task_index/min"]).astype(int)
        frame = frame.merge(tasks.rename(columns={"task_index": "lookup_task_index", "task": "companion_task_text"}), on="lookup_task_index", how="left", validate="many_to_one", sort=False)
        frame["task_bad"] = frame.task_text != frame.companion_task_text
        groups = frame.groupby("lab", sort=False)
        local_episode = groups.cumcount()
        local_frame = groups.length.cumsum() - frame.length
        local_task = groups.task_text.transform(lambda x: pd.factorize(x, sort=False)[0])
        frame["local_identity_matches"] = close(frame["stats/episode_index/min"], local_episode) & close(frame["stats/index/min"], local_frame)
        frame["local_task_matches"] = frame.lookup_task_index == local_task
        frame["identity_only"] = frame.identity_bad & ~frame.task_bad
        frame["task_only"] = ~frame.identity_bad & frame.task_bad
        frame["both_bad"] = frame.identity_bad & frame.task_bad
        frame["neither_bad"] = ~frame.identity_bad & ~frame.task_bad
        counts = {name: int(frame[name].sum()) for name in ["identity_bad", "task_bad", "identity_only", "task_only", "both_bad", "neither_bad", "local_task_matches"]}
        counts["identity_bad_with_local_pattern"] = int((frame.identity_bad & frame.local_identity_matches).sum())
        by_lab = frame.groupby("lab").agg(episode_rows=("episode_index", "size"), declared_frames=("length", "sum"), identity_bad=("identity_bad", "sum"), task_bad=("task_bad", "sum"), identity_only=("identity_only", "sum"), task_only=("task_only", "sum")).astype(int).to_dict(orient="index")
        results[split] = {"episode_rows": len(frame), "declared_frames": int(frame.length.sum()), "shards": len(paths), "counts": counts, "by_lab": by_lab, "global_checks": {"episode_ids_unique": bool(frame.episode_id.is_unique), "episode_index_equals_complete_range": bool(np.array_equal(frame.episode_index, np.arange(len(frame)))), "spans_equal_lengths": bool(np.array_equal(frame.dataset_to_index-frame.dataset_from_index, frame.length)), "dataset_intervals_contiguous": bool(np.array_equal(frame.dataset_from_index.iloc[1:], frame.dataset_to_index.iloc[:-1]))}}
    output = {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "input_manifest_sha256": digest(args.manifest), "checker_sha256": digest(Path(__file__)), "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.platform(), "splits": results, "scope": "Independent metadata relation calculation; no original audit implementation imported and no raw trajectory/video validation."}
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({s: {"episode_rows": x["episode_rows"], "counts": x["counts"]} for s, x in results.items()}, indent=2))


if __name__ == "__main__":
    main()
