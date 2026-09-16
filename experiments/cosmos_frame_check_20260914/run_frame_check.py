"""COSMOS-FRAME-20260914: frame-level check against the per-episode identity summaries.

Executes the protocol frozen in PROTOCOL.md. Reads four low-dimensional columns
(`episode_index`, `task_index`, `frame_index`, `index`) from the pinned release's frame-level Parquet
files over HTTP range requests, tallying every byte transferred against a 1 GiB ceiling.

Local metadata is read from the primary checkout's already-verified working copy, read-only; its
full-file digests are re-checked against the 2026-09-12 verified-input manifest before use.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
META = REPO / "experiments" / "work" / "cosmos3-droid"
VERIFIED = REPO / "experiments" / "results" / "20260912-cosmos-metadata" / "verified_inputs.json"
REPO_ID = "nvidia/Cosmos3-DROID"
REVISION = "dabaaffe428d67cf93fd355b82658934ee59bfec"
COLUMNS = ["episode_index", "task_index", "frame_index", "index"]
CEILING_BYTES = 1 << 30  # 1 GiB

# Network already spent on this question before the accounting was corrected, charged against the
# ceiling up front so the budget covers the whole investigation rather than only the accepted run.
# See PROTOCOL.md amendment B1 for how this figure is derived; it is a calibrated estimate, not a
# measurement, because the aborted attempt's accounting is exactly what was found to be wrong.
PRIOR_TRANSFER_BYTES = 190_000_000
SEED = 20260915
SPLITS = ("success", "failure")
EP_COLS = [
    "episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
    "dataset_from_index", "dataset_to_index",
    "stats/episode_index/min", "stats/episode_index/max", "stats/episode_index/mean",
    "stats/episode_index/count",
    "stats/task_index/min", "stats/task_index/max", "stats/task_index/mean",
]


class TransferBudget:
    """Tally bytes actually fetched over HTTP and refuse a fetch that would exceed the ceiling.

    Accounting is hooked at the fsspec cache's `fetcher`, which is `HfFileSystemFile._fetch_range`:
    that method issues one HTTP Range GET and returns `response.content`, so `len()` of its return
    value is the HTTP response body actually transferred for that request. Files are opened with
    `cache_type="none"`, which keeps the handle seekable (`BaseCache`) but performs no read-ahead, so
    fetched bytes equal the bytes PyArrow asked for rather than exceeding them.

    What this does and does not count: it counts HTTP response bodies for the range GETs that read
    file content. It excludes request/response headers, TLS framing, redirect responses and the
    small metadata calls fsspec makes to resolve a path and its size. It is therefore a close lower
    bound on wire traffic, not an exact wire measurement, and is reported as such.
    """

    def __init__(self, ceiling_bytes: int, already_spent_bytes: int = 0):
        self.ceiling = ceiling_bytes
        self.prior = already_spent_bytes
        self.fetched = 0
        self.requests = 0
        self.refused = None

    @property
    def total(self) -> int:
        return self.prior + self.fetched

    @property
    def remaining(self) -> int:
        return self.ceiling - self.total

    def attach(self, handle):
        """Wrap this handle's fetcher so every range GET is checked before it is issued."""
        original = handle.cache.fetcher
        per_file = {"fetched": 0, "requests": 0}

        def counting_fetch(start: int, end: int):
            requested = end - start
            if requested > self.remaining:
                self.refused = {"requested_bytes": int(requested),
                                "remaining_bytes": int(self.remaining)}
                raise BudgetExceeded(
                    f"refusing a {requested} byte range GET: only {self.remaining} bytes remain "
                    f"of the {self.ceiling} byte ceiling")
            block = original(start, end)
            self.fetched += len(block)
            self.requests += 1
            per_file["fetched"] += len(block)
            per_file["requests"] += 1
            return block

        handle.cache.fetcher = counting_fetch
        return per_file


class BudgetExceeded(RuntimeError):
    """Raised before a fetch that would cross the ceiling, so the ceiling is never crossed."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_local_metadata() -> dict:
    """Re-check the local working copy against the 2026-09-12 verified-input manifest."""
    manifest = json.loads(VERIFIED.read_bytes())
    expected = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    checked, mismatches = {}, []
    for rel, want in expected.items():
        path = META / rel
        if not path.exists():
            mismatches.append({"path": rel, "reason": "absent from the local working copy"})
            continue
        got = sha256_file(path)
        checked[rel] = got
        if got != want:
            mismatches.append({"path": rel, "expected": want, "observed": got})
    return {"verified": checked, "mismatches": mismatches,
            "manifest": str(VERIFIED.relative_to(REPO))}


# Slash-bearing column names cannot be namedtuple attributes, so rename on load.
SAFE_NAMES = {
    "data/chunk_index": "data_chunk_index", "data/file_index": "data_file_index",
    "stats/episode_index/min": "summary_ep_min", "stats/episode_index/max": "summary_ep_max",
    "stats/episode_index/mean": "summary_ep_mean", "stats/episode_index/count": "summary_ep_count",
    "stats/task_index/min": "summary_task_min", "stats/task_index/max": "summary_task_max",
    "stats/task_index/mean": "summary_task_mean",
}


def load_split_metadata(split: str):
    shards = sorted((META / split / "episodes").glob("*.parquet"))
    episodes = pd.concat(
        [pq.read_table(s, columns=EP_COLS).to_pandas() for s in shards], ignore_index=True)
    episodes = episodes.rename(columns=SAFE_NAMES)
    tasks = pq.read_table(META / split / "tasks.parquet").to_pandas()
    # tasks.parquet is indexed by task string with a task_index column.
    task_text_by_index = {int(v): str(k) for k, v in zip(tasks.index, tasks["task_index"])}
    info = json.loads((META / split / "info.json").read_bytes())
    return episodes, task_text_by_index, info


def scalar(value):
    """Episode-table statistic cells are length-one arrays; unwrap to a float."""
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else float("nan")


def episode_task_text(cell) -> str:
    """The episode table stores `tasks` as a list-like of task strings."""
    values = list(np.asarray(cell).reshape(-1))
    return str(values[0]) if len(values) == 1 else "|".join(str(v) for v in values)


def main() -> int:
    started = datetime.now(timezone.utc)
    metadata_check = verify_local_metadata()
    if metadata_check["mismatches"]:
        print(json.dumps(metadata_check["mismatches"], indent=2))
        raise SystemExit("local metadata digests do not match the verified manifest")

    splits = {}
    eligible = []
    for split in SPLITS:
        episodes, task_text, info = load_split_metadata(split)
        splits[split] = {"episodes": episodes, "task_text": task_text, "info": info}
        pairs = (episodes[["data_chunk_index", "data_file_index"]]
                 .drop_duplicates().sort_values(["data_chunk_index", "data_file_index"]))
        for chunk, file_index in pairs.itertuples(index=False):
            eligible.append({"split": split, "chunk_index": int(chunk),
                             "file_index": int(file_index)})

    # Primary plan: complete coverage. The pre-registered permutation fixes the order in which files
    # are attempted, so that if the ceiling is reached the processed subset is the frozen one.
    order = np.random.default_rng(SEED).permutation(len(eligible))
    ordered = [eligible[i] for i in order]

    fs = HfFileSystem()
    budget = TransferBudget(CEILING_BYTES, PRIOR_TRANSFER_BYTES)
    processed, skipped, per_episode_failures, summary_rows = [], [], [], []
    totals = {"files": 0, "frames": 0, "episodes": 0,
              "frame_consistent_episodes": 0,
              "c1_length": 0, "c2_index_interval": 0, "c3_frame_order": 0, "c4_task_identity": 0}
    summary_counters = {
        "summary_episode_index_matches_frames": 0,
        "summary_episode_index_differs_from_frames": 0,
        "summary_task_index_matches_frames": 0,
        "summary_task_index_differs_from_frames": 0,
        "summary_count_matches_frames": 0,
        "summary_count_differs_from_frames": 0,
    }

    for entry in ordered:
        split = entry["split"]
        remote = (f"datasets/{REPO_ID}@{REVISION}/{split}/data/"
                  f"chunk-{entry['chunk_index']:03d}/file-{entry['file_index']:03d}.parquet")
        try:
            # cache_type="none" keeps the handle seekable but disables read-ahead, so the bytes
            # fetched are the bytes PyArrow needs for the four selected columns.
            handle = fs.open(remote, "rb", cache_type="none")
            per_file = budget.attach(handle)
            table = pq.ParquetFile(handle).read(columns=COLUMNS).to_pandas()
        except BudgetExceeded as exc:
            skipped.append({**entry, "reason": str(exc), "cause": "ceiling"})
            break
        except Exception as exc:  # noqa: BLE001
            skipped.append({**entry, "reason": f"{type(exc).__name__}: {exc}", "cause": "error"})
            continue

        episodes = splits[split]["episodes"]
        task_text = splits[split]["task_text"]
        in_file = episodes[(episodes["data_chunk_index"] == entry["chunk_index"])
                           & (episodes["data_file_index"] == entry["file_index"])]
        grouped = {int(k): v for k, v in table.groupby("episode_index")}

        file_failures = 0
        for row in in_file.itertuples(index=False):
            ep = int(getattr(row, "episode_index"))
            frames = grouped.get(ep)
            declared_length = int(getattr(row, "length"))
            lo = int(getattr(row, "dataset_from_index"))
            hi = int(getattr(row, "dataset_to_index"))
            if frames is None:
                per_episode_failures.append({**entry, "episode_index": ep,
                                             "check": "C0_frames_present", "detail": "no frame rows"})
                file_failures += 1
                totals["episodes"] += 1
                continue

            observed_index = frames["index"].to_numpy()
            observed_frame_index = frames["frame_index"].to_numpy()
            observed_task = frames["task_index"].to_numpy()

            c1 = len(frames) == declared_length
            c2 = (len(observed_index) == hi - lo
                  and np.array_equal(np.sort(observed_index), np.arange(lo, hi)))
            c3 = np.array_equal(np.sort(observed_frame_index), np.arange(declared_length))
            constant_task = observed_task.min() == observed_task.max()
            declared_text = episode_task_text(getattr(row, "tasks"))
            resolved = task_text.get(int(observed_task[0])) if constant_task else None
            c4 = bool(constant_task and resolved is not None and resolved == declared_text)

            totals["episodes"] += 1
            totals["frames"] += len(frames)
            totals["c1_length"] += int(c1)
            totals["c2_index_interval"] += int(c2)
            totals["c3_frame_order"] += int(c3)
            totals["c4_task_identity"] += int(c4)
            if c1 and c2 and c3 and c4:
                totals["frame_consistent_episodes"] += 1
            else:
                file_failures += 1
                per_episode_failures.append({
                    **entry, "episode_index": ep,
                    "C1_length": bool(c1), "C2_index_interval": bool(c2),
                    "C3_frame_order": bool(c3), "C4_task_identity": bool(c4),
                    "declared_length": declared_length, "observed_frames": int(len(frames)),
                    "declared_interval": [lo, hi],
                    "observed_index_min": int(observed_index.min()),
                    "observed_index_max": int(observed_index.max()),
                    "observed_task_index_min": int(observed_task.min()),
                    "observed_task_index_max": int(observed_task.max()),
                    "declared_task_text_prefix": declared_text[:120],
                    "resolved_task_text_prefix": (resolved or "")[:120],
                })

            # C5: do the affected per-episode summaries describe these frames?
            summary_ep_min = scalar(row.summary_ep_min)
            summary_ep_max = scalar(row.summary_ep_max)
            summary_ep_mean = scalar(row.summary_ep_mean)
            summary_ep_count = scalar(row.summary_ep_count)
            summary_task_min = scalar(row.summary_task_min)
            summary_task_max = scalar(row.summary_task_max)
            summary_task_mean = scalar(row.summary_task_mean)

            ep_summary_matches = (summary_ep_min == ep and summary_ep_max == ep
                                  and summary_ep_mean == ep)
            task_summary_matches = (summary_task_min == observed_task.min()
                                    and summary_task_max == observed_task.max()
                                    and abs(summary_task_mean - float(observed_task.mean())) <= 1e-6)
            count_matches = summary_ep_count == len(frames)
            summary_counters["summary_episode_index_matches_frames" if ep_summary_matches
                             else "summary_episode_index_differs_from_frames"] += 1
            summary_counters["summary_task_index_matches_frames" if task_summary_matches
                             else "summary_task_index_differs_from_frames"] += 1
            summary_counters["summary_count_matches_frames" if count_matches
                             else "summary_count_differs_from_frames"] += 1
            if len(summary_rows) < 200 and not (ep_summary_matches and task_summary_matches):
                summary_rows.append({
                    "split": split, "episode_index": ep,
                    "summary_episode_index_min_max_mean": [summary_ep_min, summary_ep_max,
                                                           summary_ep_mean],
                    "observed_frame_episode_index": ep,
                    "summary_task_index_min_max_mean": [summary_task_min, summary_task_max,
                                                        summary_task_mean],
                    "observed_frame_task_index": int(observed_task[0]),
                })

        totals["files"] += 1
        processed.append({**entry, "rows": int(len(table)),
                          "episodes_in_file": int(len(in_file)),
                          "bytes_fetched": int(per_file["fetched"]),
                          "range_requests": int(per_file["requests"]),
                          "episodes_failing_c1_c4": file_failures})
        print(f"{split}/chunk-{entry['chunk_index']:03d}/file-{entry['file_index']:03d} "
              f"rows={len(table)} eps={len(in_file)} fail={file_failures} "
              f"fetched_MiB={budget.fetched/1048576:.1f} "
              f"charged_MiB={budget.total/1048576:.1f}", flush=True)

    remaining = [e for e in ordered if not any(
        p["split"] == e["split"] and p["chunk_index"] == e["chunk_index"]
        and p["file_index"] == e["file_index"] for p in processed)]

    report = {
        "experiment_id": "COSMOS-FRAME-20260914",
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python_version": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "dataset": {"repo_id": REPO_ID, "revision": REVISION,
                    "codebase_version": {s: splits[s]["info"]["codebase_version"] for s in SPLITS},
                    "declared_episodes": {s: splits[s]["info"]["total_episodes"] for s in SPLITS},
                    "declared_frames": {s: splits[s]["info"]["total_frames"] for s in SPLITS}},
        "local_metadata_verification": metadata_check,
        "columns_read": COLUMNS,
        "no_image_bytes_read": True,
        "image_features_are_video_dtype": True,
        "transfer_accounting": {
            "ceiling_bytes": CEILING_BYTES,
            "measurement_point": (
                "HfFileSystemFile._fetch_range return length, i.e. the HTTP response body of each "
                "range GET; excludes headers, TLS framing, redirects and path/size metadata calls, "
                "so it is a close lower bound on wire traffic rather than an exact wire measurement"
            ),
            "cache_type": "none (seekable BaseCache, no read-ahead)",
            "range_requests": budget.requests,
            "bytes_fetched_this_run": budget.fetched,
            "mebibytes_fetched_this_run": round(budget.fetched / 1048576, 2),
            "prior_transfer_charged_bytes": PRIOR_TRANSFER_BYTES,
            "prior_transfer_basis": "calibrated estimate for the aborted attempt and probes; see "
                                    "PROTOCOL.md amendment B1",
            "total_charged_bytes": budget.total,
            "mebibytes_charged_total": round(budget.total / 1048576, 2),
            "remaining_bytes": budget.remaining,
            "refused_fetch": budget.refused,
        },
        "ceiling_reached": bool(budget.refused),
        "eligible_files": len(eligible),
        "processed_files": len(processed),
        "unprocessed_files": remaining,
        "selection": {
            "plan": "complete coverage of every eligible data file",
            "order_rule": f"numpy.random.default_rng({SEED}).permutation over eligible files sorted "
                          "by (split, chunk_index, file_index)",
            "seed": SEED,
            "fallback_in_force": bool(remaining),
        },
        "totals": totals,
        "summary_versus_frames": summary_counters,
        "per_episode_failures": per_episode_failures[:500],
        "per_episode_failure_count": len(per_episode_failures),
        "summary_mismatch_examples": summary_rows,
        "processed_file_detail": processed,
        "skipped": skipped,
        "scope_statement": (
            "Frame-level conclusions cover exactly the four columns read from the files listed in "
            "processed_file_detail. No observed frame defect means no observed frame defect in that "
            "scope; it is not a claim that unread columns or other revisions are unaffected."
        ),
    }
    (HERE / "cosmos_frame_check_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + json.dumps({k: report[k] for k in (
        "eligible_files", "processed_files", "transfer_accounting", "totals",
        "summary_versus_frames", "per_episode_failure_count")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
