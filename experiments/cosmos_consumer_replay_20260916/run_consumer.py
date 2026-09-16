#!/usr/bin/env python3
"""Scoped, exact LeRobot consumer execution with a summary-only intervention."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import ssl
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import certifi
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def cell(value):
    return float(np.asarray(value).reshape(-1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=HERE.parent / "work" / "cosmos_consumer_20260916")
    ap.add_argument("--metadata-cache", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    inputs = json.loads((HERE / "INPUTS.json").read_text())
    report = {"experiment": protocol["experiment"], "started_utc": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(HERE / "PROTOCOL.json"), "input_manifest_sha256": sha(HERE / "INPUTS.json"),
              "script_sha256": sha(__file__), "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "pyarrow", "huggingface_hub")},
              "scope": protocol["scope"], "metadata": [], "records": [], "frame_receipts": [],
              "fixed_offsets": protocol["fixed_offsets"], "status": "EXECUTION_ERROR"}
    try:
        source = args.work / "aggregate.py"
        url = ("https://raw.githubusercontent.com/huggingface/lerobot/" + protocol["consumer_commit"]
               + "/" + protocol["consumer_path"])
        if not source.exists():
            with urlopen(url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as incoming, source.open("wb") as f:
                shutil.copyfileobj(incoming, f)
        if sha(source) != protocol["consumer_full_file_sha256"]:
            raise ValueError("pinned consumer source digest mismatch")
        tree = ast.parse(source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == protocol["consumer_function"])
        # Execute the exact function text, with annotations deferred to avoid importing the full
        # robotics dependency graph. No statements inside the function are modified or replaced.
        exact_function = ast.get_source_segment(source.read_text(), node)
        namespace = {"np": np, "pd": pd}
        exec(compile("from __future__ import annotations\n" + exact_function, "pinned_update_meta_data", "exec"), namespace)
        consumer = namespace[protocol["consumer_function"]]
        report["consumer"] = {"commit": protocol["consumer_commit"], "source_url": url,
                              "full_file_sha256": sha(source), "function": node.name,
                              "function_text_sha256": hashlib.sha256(exact_function.encode()).hexdigest(),
                              "line_start": node.lineno, "line_end": node.end_lineno,
                              "extraction": "Exact ast source segment; future annotations prepended, pd/np dependencies supplied; function body unchanged."}
        metadata, tasks = {}, {}
        fixed_cols = ["episode_index", "length", "dataset_from_index", "dataset_to_index", "tasks",
                      "meta/episodes/chunk_index", "meta/episodes/file_index", "data/chunk_index", "data/file_index"]
        for item in inputs["files"]:
            src = ((args.metadata_cache / item["path"]) if args.metadata_cache else
                   args.work / "metadata" / item["path"])
            origin = "explicit_verified_cache" if args.metadata_cache else "verified_public_download"
            if not src.exists():
                if args.metadata_cache:
                    raise FileNotFoundError("metadata cache missing " + item["path"])
                src.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(item["url"], timeout=120,
                             context=ssl.create_default_context(cafile=certifi.where())) as incoming, src.open("wb") as f:
                    shutil.copyfileobj(incoming, f)
            if src.stat().st_size != item["size_bytes"] or sha(src) != item["sha256"]:
                raise ValueError("input integrity mismatch: " + item["path"])
            report["metadata"].append({"path": item["path"], "sha256": item["sha256"], "origin": origin})
            split = item["path"].split("/")[0]
            if item["path"].endswith("tasks.parquet"):
                tasks[split] = pq.read_table(src).to_pandas()
            else:
                schema = pq.ParquetFile(src).schema_arrow.names
                cols = fixed_cols + [c for c in schema if c.startswith("stats/episode_index/") or c.startswith("stats/task_index/")]
                metadata.setdefault(split, []).append(pq.read_table(src, columns=cols).to_pandas())
        metadata = {s: pd.concat(t, ignore_index=True).sort_values("episode_index") for s, t in metadata.items()}
        # Verify the registered first-affected/first-control selection rather than choosing outputs.
        for split, df in metadata.items():
            matches = np.ones(len(df), dtype=bool)
            for stat in ("min", "max", "mean"):
                matches &= np.array([cell(x) for x in df["stats/episode_index/" + stat]]) == df.episode_index.to_numpy()
            for kind, mask in (("affected", ~matches), ("control", matches)):
                expected = next(r for r in protocol["selection_records"] if r["split"] == split and r["kind"] == kind)
                if int(df.loc[mask].iloc[0].episode_index) != expected["episode_index"]:
                    raise ValueError("preselection identity mismatch")
        fs = HfFileSystem(token=False)
        for selection in protocol["selection_records"]:
            split, ep = selection["split"], selection["episode_index"]
            original = metadata[split].loc[metadata[split].episode_index == ep].copy(deep=True)
            rel = f"{split}/data/chunk-{selection['chunk_index']:03d}/file-{selection['file_index']:03d}.parquet"
            remote = f"datasets/{inputs['repository']}@{inputs['revision']}/{rel}"
            fetched = 0
            with fs.open(remote, "rb", cache_type="none") as handle:
                fetch = handle.cache.fetcher
                def counted(start, end):
                    nonlocal fetched
                    if fetched + end - start > 1 << 28:
                        raise RuntimeError("per-file 256 MiB range budget exceeded")
                    block = fetch(start, end)
                    fetched += len(block)
                    report["frame_receipts"].append({"path": rel, "start": start, "end_exclusive": end,
                                                      "returned_bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
                    return block
                handle.cache.fetcher = counted
                frame = pq.ParquetFile(handle).read(columns=["episode_index", "index", "frame_index"]).to_pandas()
            frame = frame.loc[frame.episode_index == ep]
            if len(frame) != int(original.iloc[0].length) or frame.empty:
                raise ValueError("selected frame coverage mismatch")
            h = hashlib.sha256()
            for col in ("episode_index", "index", "frame_index"):
                h.update(col.encode() + b"\0")
                h.update(frame[col].to_numpy(dtype="<i8").tobytes())
            corrected = original.copy(deep=True)
            for stat in protocol["frozen_summary_keys"]:
                col = "stats/episode_index/" + stat
                corrected[col] = [np.full_like(x, ep) for x in corrected[col]]
            offsets = protocol["fixed_offsets"]
            destination = SimpleNamespace(info=SimpleNamespace(total_frames=offsets["total_frames"],
                                                               total_episodes=offsets["total_episodes"]),
                                          tasks=tasks[split])
            expect = int(frame.episode_index.iloc[0]) + offsets["total_episodes"]
            record = {**selection, "frame_path": rel, "frame_rows": len(frame),
                      "selected_frame_digest": h.hexdigest(), "expected_output_episode_summary": expect,
                      "conditions": {}}
            for condition, df in (("original", original), ("summary_corrected", corrected)):
                out = consumer(df.copy(deep=True), destination,
                               {"chunk": offsets["meta_chunk"], "file": offsets["meta_file"]},
                               {"chunk": offsets["data_chunk"], "file": offsets["data_file"]}, {})
                values = {s: cell(out.iloc[0]["stats/episode_index/" + s]) for s in protocol["frozen_summary_keys"]}
                wrong = [s for s, value in values.items() if value != expect]
                record["conditions"][condition] = {"output_summary": values, "violating_fields": wrong,
                                                    "violating_field_count": len(wrong),
                                                    "output_episode_index": int(out.iloc[0].episode_index)}
            report["records"].append(record)
            print(json.dumps({"split": split, "kind": selection["kind"], "episode_index": ep,
                              "original_bad_fields": record["conditions"]["original"]["violating_field_count"],
                              "corrected_bad_fields": record["conditions"]["summary_corrected"]["violating_field_count"]}), flush=True)
        report["totals"] = {condition: {"records": len(report["records"]),
                                        "violating_records": sum(bool(r["conditions"][condition]["violating_fields"]) for r in report["records"]),
                                        "violating_fields": sum(r["conditions"][condition]["violating_field_count"] for r in report["records"])}
                            for condition in ("original", "summary_corrected")}
        report["status"] = "EXECUTED"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.out / "results.json", report)
    print(json.dumps({"status": report["status"], "totals": report.get("totals")}), flush=True)
    return 0 if report["status"] == "EXECUTED" else 2


if __name__ == "__main__":
    sys.exit(main())
