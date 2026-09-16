#!/usr/bin/env python3
"""Replay the frozen Cosmos3-DROID metadata audit with full verified inputs.

Run scripts/restore_original.py first. Downloads retain upstream licensing and
are ignored by Git; this script never changes the frozen original source.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "dabaaffe428d67cf93fd355b82658934ee59bfec"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "original")
    parser.add_argument("--data-root", type=Path, default=ROOT / "experiments/work/cosmos3-droid")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/20260912-cosmos-metadata")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True, exist_ok=True)
    historical_path = args.source_root / "results/p8-droid-full-metadata-audit.json"
    historical = json.loads(historical_path.read_text())
    manifest = []
    for split, result in sorted(historical["splits"].items()):
        for record in result["manifests"]["episode_metadata"]:
            manifest.append({"remote_path": record["remote_path"], "path": f"{split}/episodes/{Path(record['remote_path']).name}", "size_bytes": record["expected_size_bytes"], "sha256": record["expected_sha256"]})
        for record in result["manifests"]["companions"].values():
            manifest.append({"remote_path": record["remote_path"], "path": f"{split}/{Path(record['remote_path']).name}", "size_bytes": record["size_bytes"], "sha256": record["sha256"]})
    code = {str(p.relative_to(args.source_root)): sha(p) for p in [args.source_root / "scripts/audit_droid_full_metadata.py", args.source_root / "src/robocontract/droid_metadata_audit.py", args.source_root / "src/robocontract/droid_metadata_collection.py"]}
    protocol = {"frozen_before_download_utc": started, "revision": REVISION, "source_commit": "5d50746be318692bc46a417af65e781bd3946b88", "historical_result_sha256": sha(historical_path), "code_sha256": code, "files": manifest, "scope": "Complete metadata files and companion tables; no trajectory or video bytes. Rows are episode records, frames are declared totals."}
    protocol_path = args.out / "input_protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text())
        if any(existing[key] != value for key, value in protocol.items() if key != "frozen_before_download_utc"):
            raise ValueError("Protocol differs from its preserved freeze")
    else:
        save(protocol_path, protocol)
    if not args.download_only and (args.out / "rerun.json").exists():
        raise FileExistsError("Use a new --out directory to preserve completed runs")

    def recover(item):
        path = args.data_root / item["path"]
        url = f"https://huggingface.co/datasets/nvidia/Cosmos3-DROID/resolve/{REVISION}/{item['remote_path']}?download=true"
        if not path.exists():
            if args.offline:
                raise FileNotFoundError(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=90) as response, temp.open("wb") as target:
                for b in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(b)
            temp.replace(path)
        actual = sha(path)
        if actual != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Input identity mismatch: {item['path']}")
        print(f"VERIFIED {item['path']} {item['size_bytes']} {actual}", flush=True)
        return {**item, "verified_utc": datetime.now(timezone.utc).isoformat(), "url": url}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        verified = list(executor.map(recover, manifest))
    verified_path = args.out / ("verified_inputs.json" if args.download_only else "preaudit_verified_inputs.json")
    if verified_path.exists():
        raise FileExistsError(verified_path)
    save(verified_path, {"files": verified, "total_bytes": sum(x["size_bytes"] for x in verified), "all_full_file_sha256_verified": True})
    if args.download_only:
        return 0
    command = [sys.executable, str(args.source_root / "scripts/audit_droid_full_metadata.py"), "--data-root", str(args.data_root), "--output", str(args.out / "rerun.json")]
    start = time.perf_counter()
    print("RUN", json.dumps(command), flush=True)
    proc = subprocess.run(command, text=True, capture_output=True)
    (args.out / "audit.stdout.log").write_text(proc.stdout)
    (args.out / "audit.stderr.log").write_text(proc.stderr)
    print(proc.stdout, end="", flush=True)
    print(proc.stderr, end="", file=sys.stderr, flush=True)
    save(args.out / "run.json", {"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(), "command": command, "exit_code": proc.returncode, "elapsed_s": time.perf_counter() - start, "python": sys.version, "platform": platform.platform(), "runner_sha256": sha(Path(__file__)), "result_sha256": sha(args.out / "rerun.json") if (args.out / "rerun.json").exists() else None, "code_unchanged": all(sha(args.source_root / name) == h for name, h in code.items())})
    if proc.returncode:
        return proc.returncode
    fresh = json.loads((args.out / "rerun.json").read_text())
    checks = {"aggregate_summary_equal": fresh["summary"] == historical["summary"]}
    for split in ("success", "failure"):
        for key in ("declared_count_checks", "cross_shard", "cross_shard_video_lineage", "aggregate_single_shard_violations"):
            checks[f"{split}.{key}"] = fresh["splits"][split][key] == historical["splits"][split][key]
    save(args.out / "historical_comparison.json", {"checks": checks, "all_equal": all(checks.values()), "note": "Timing, absolute paths and fresh run metadata are excluded. Original results were never used as audit output."})
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
