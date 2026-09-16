# Independent Cosmos row-wise alignment check C5

Python 3.12 or later. From the repository root, in a fresh standalone environment:

```sh
python -m pip install -r experiments/cosmos_c5_20260916/requirements.txt
python experiments/cosmos_c5_20260916/run_c5.py --out /tmp/cosmos-c5-new-run
```

The output directory must not exist. The default ignored work directory is
`experiments/work/cosmos_c5_20260916/`. The script recovers all six pinned episode
metadata shards (about 387 MB) using `INPUTS.json`, verifies their complete sizes
and SHA-256 digests, then reads three identity columns from all 74 frame files
using public HTTP range requests. No credentials, private source tree, historical
result cache, videos, actions or original full frame-Parquet downloads are needed.

An optional `--metadata-cache DIRECTORY` accepts existing metadata in the manifest's
relative layout, only after the same full-file verification. This changes input
recovery, not the predicate. `--work DIRECTORY` changes the input cache location.
Fetched-range SHA-256 receipts and canonical decoded-column digests identify the
frame-column read; they are not full-Parquet hashes.

The public package's shared environment (NumPy 1.26.4, pandas 2.1.4,
PyArrow 14.0.2, huggingface_hub 0.34.4) was also verified. If using that environment,
keep its root requirements instead of installing this folder's standalone pins.
`CURRENT_RESULTS.json` points to the accepted current-runner result,
`public_env_compat2/results.json`: fresh dependency environment plus freshly
public-recovered metadata, identical counts and all decoded-column digests.

The scientific check is exact integer equality, in the decoded physical row pairing:

```text
index[row] == dataset_from_index[episode_index[row]] + frame_index[row]
```

`PASS` means complete expected coverage, no integrity errors and no C5 violation.
`CHECK_FAILED` is a successfully executed scientific negative; it exits zero, with
the violations in `results.json`. `INCOMPLETE`/`EXECUTION_ERROR` exits nonzero and
cannot establish a pass. The runner preserves existing output directories rather
than overwriting them. Inspect `status`, `complete_coverage` and `totals`, not just
the process exit code.

Accepted first run: `run_20260916T033332Z/results.json`; raw stdout is the adjacent
`.log` file. All 74 files, 71,907 episodes and 22,412,712 rows were checked, with
zero violating rows or episodes. 128,800,948 bytes in 148 range response bodies
were fetched in 87.47 seconds. This first run reused explicitly verified metadata
but freshly read every frame column. `clean_recovery/` records the additional
run from newly recovered public inputs with no private metadata-cache argument.

Historical C1–C4 and their logs remain unchanged. C5 does not test physical row
ordering, timestamps, image content, action alignment, downstream training or
model harm. A historical code comment labeled the separate summary check “C5”;
the versioned `COSMOS-C5-20260916` protocol unambiguously denotes the new row-wise
predicate, not that old comment.
