#!/usr/bin/env python3
"""Standalone exact row-wise C5; see PROTOCOL.md. No historical results imported."""
from __future__ import annotations

import argparse
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
from urllib.request import urlopen

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import certifi
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
COLS = ["episode_index", "index", "frame_index"]
META_COLS = ["episode_index", "length", "dataset_from_index", "dataset_to_index",
             "data/chunk_index", "data/file_index"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=HERE.parent / "work" / "cosmos_c5_20260916")
    ap.add_argument("--metadata-cache", type=Path, help="optional full-hash-verified metadata cache")
    ap.add_argument("--out", type=Path, required=True, help="new, nonexistent output directory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    args.work.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "INPUTS.json").read_text())
    report = {
        "experiment": "COSMOS-C5-20260916", "protocol_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": digest(__file__), "protocol_sha256": digest(HERE / "PROTOCOL.md"),
        "input_manifest_sha256": digest(HERE / "INPUTS.json"),
        "dataset": {k: manifest[k] for k in ("repository", "revision")},
        "versions": {x: importlib.metadata.version(x) for x in
                     ("numpy", "pandas", "pyarrow", "huggingface_hub")},
        "python": platform.python_version(), "platform": platform.system() + " " + platform.machine(),
        "check": "index[row] == dataset_from_index[episode_index[row]] + frame_index[row]",
        "expected": {"files": 74, "episodes": 71907, "rows": 22412712},
        "totals": {"files": 0, "episodes": 0, "rows": 0,
                   "violating_rows": 0, "violating_episodes": 0},
        "metadata": [], "files": [], "integrity_errors": [], "execution_errors": [],
        "range_response_bytes": 0, "range_requests": 0,
        "range_ceiling_bytes": 1 << 30,
        "integrity_scope": "Full SHA-256/size for episode metadata; pinned revision, fetched-range receipts and decoded-column digests for frame files, not full Parquet hashes.",
        "transfer_scope": "Range GET response bodies only; excludes metadata downloads, headers, redirects and path/size resolution traffic.",
        "scientific_scope": "Exact row-wise relation among the three columns and metadata interval start; no physical ordering, visual/timestamp/action correctness or model-harm inference.",
    }
    receipt = (args.out / "range_receipts.jsonl").open("w")
    try:
        splits = {}
        for item in manifest["files"]:
            destination = args.work / "metadata" / item["path"]
            source = (args.metadata_cache / item["path"]) if args.metadata_cache else destination
            origin = "explicit_verified_cache" if args.metadata_cache else "existing_verified_download"
            if not source.exists():
                if args.metadata_cache:
                    raise FileNotFoundError("specified metadata cache missing " + item["path"])
                source.parent.mkdir(parents=True, exist_ok=True)
                with urlopen(item["url"], timeout=120,
                             context=ssl.create_default_context(cafile=certifi.where())) as incoming, source.open("wb") as output:
                    shutil.copyfileobj(incoming, output)
                origin = "public_download"
            if source.stat().st_size != item["size_bytes"] or digest(source) != item["sha256"]:
                raise ValueError("metadata integrity failed: " + item["path"])
            report["metadata"].append({"path": item["path"], "origin": origin,
                                       "sha256": item["sha256"], "size_bytes": item["size_bytes"]})
            split = item["path"].split("/")[0]
            splits.setdefault(split, []).append(pq.read_table(source, columns=META_COLS).to_pandas())
        fs = HfFileSystem(token=False)
        for split in sorted(splits):
            metadata = pd.concat(splits[split], ignore_index=True)
            if metadata.episode_index.duplicated().any():
                raise ValueError("duplicate metadata episode_index in " + split)
            for (chunk, file), episodes in metadata.groupby(["data/chunk_index", "data/file_index"], sort=True):
                rel = f"{split}/data/chunk-{int(chunk):03d}/file-{int(file):03d}.parquet"
                remote = f"datasets/{manifest['repository']}@{manifest['revision']}/{rel}"
                detail = {"path": rel, "range_response_bytes": 0, "range_requests": 0}
                try:
                    with fs.open(remote, "rb", cache_type="none") as handle:
                        fetch = handle.cache.fetcher
                        def tracked(start, end):
                            if report["range_response_bytes"] + end - start > report["range_ceiling_bytes"]:
                                raise RuntimeError("frame range response-body ceiling reached")
                            block = fetch(start, end)
                            for owner in (report, detail):
                                owner["range_response_bytes"] += len(block)
                                owner["range_requests"] += 1
                            receipt.write(json.dumps({"path": rel, "start": start, "end_exclusive": end,
                                                      "returned_bytes": len(block),
                                                      "sha256": hashlib.sha256(block).hexdigest()}) + "\n")
                            receipt.flush()
                            return block
                        handle.cache.fetcher = tracked
                        table = pq.ParquetFile(handle).read(columns=COLS).to_pandas()
                    for name in COLS:
                        if table[name].isna().any() or not pd.api.types.is_integer_dtype(table[name]):
                            raise ValueError("non-integer/null identity column: " + name)
                    h = hashlib.sha256()
                    for name in COLS:
                        h.update(name.encode("ascii") + b"\0")
                        h.update(table[name].to_numpy(dtype="<i8").tobytes(order="C"))
                    detail["decoded_columns_sha256"] = h.hexdigest()
                    detail["digest_encoding"] = "ordered column name ASCII+NUL then all values little-endian signed int64 in physical row order; columns episode_index,index,frame_index"
                    counts = table.groupby("episode_index", sort=False).size()
                    wanted = episodes.set_index("episode_index")
                    unknown = sorted(set(counts.index) - set(wanted.index))
                    missing = sorted(set(wanted.index) - set(counts.index))
                    if unknown or missing:
                        report["integrity_errors"].append({"path": rel, "unknown_episode_ids": unknown, "missing_episode_ids": missing})
                    expected_starts = table.episode_index.map(wanted.dataset_from_index)
                    valid = expected_starts.notna()
                    mismatch = valid & (table["index"] != expected_starts.fillna(0).astype(np.int64) + table.frame_index)
                    bad_eps = sorted(map(int, table.loc[mismatch, "episode_index"].unique()))
                    lengths_bad = []
                    for ep in sorted(set(counts.index) & set(wanted.index)):
                        if int(counts.loc[ep]) != int(wanted.loc[ep, "length"]):
                            lengths_bad.append(int(ep))
                    if lengths_bad:
                        report["integrity_errors"].append({"path": rel, "length_mismatch_episode_ids": lengths_bad})
                    detail.update({"rows": len(table), "episodes": len(counts),
                                   "violating_rows": int(mismatch.sum()), "violating_episodes": len(bad_eps),
                                   "violating_episode_ids": bad_eps,
                                   "examples": [{"physical_row_offset": int(i), "episode_index": int(table.loc[i, "episode_index"]),
                                                 "index": int(table.loc[i, "index"]), "frame_index": int(table.loc[i, "frame_index"]),
                                                 "dataset_from_index": int(expected_starts.loc[i])}
                                                for i in table.index[mismatch][:10]]})
                    report["files"].append(detail)
                    report["totals"]["files"] += 1
                    for key in ("rows", "episodes", "violating_rows", "violating_episodes"):
                        report["totals"][key] += detail[key]
                    print(f"{rel}: rows={len(table)} episodes={len(counts)} C5_violations={int(mismatch.sum())} fetched={report['range_response_bytes']}", flush=True)
                    save(args.out / "progress.json", report)
                except Exception as exc:
                    report["execution_errors"].append({"path": rel, "type": type(exc).__name__, "message": str(exc)})
                    print(f"EXECUTION_ERROR {rel}: {type(exc).__name__}: {exc}", flush=True)
        full = all(report["totals"][k] == n for k, n in report["expected"].items())
        report["complete_coverage"] = bool(full and not report["execution_errors"] and not report["integrity_errors"])
        report["status"] = ("INCOMPLETE" if not report["complete_coverage"] else
                            "CHECK_FAILED" if report["totals"]["violating_rows"] else "PASS")
    except Exception as exc:
        report["status"] = "EXECUTION_ERROR"
        report["execution_errors"].append({"type": type(exc).__name__, "message": str(exc)})
        traceback.print_exc()
    finally:
        receipt.close()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["range_receipts_sha256"] = digest(args.out / "range_receipts.jsonl")
        save(args.out / "results.json", report)
        print(json.dumps({"status": report["status"], "totals": report["totals"]}), flush=True)
    return 0 if report["status"] in ("PASS", "CHECK_FAILED") else 2


if __name__ == "__main__":
    sys.exit(main())
