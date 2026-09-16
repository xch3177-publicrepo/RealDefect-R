# Work record: independent C5

Purpose (UTC 2026-09-16): add the requested row-wise identity relation while
preserving the historical C1–C4 results. All changes in this experiment are new
files. No old source, tolerance or scientific output was edited.

The protocol and six-file manifest were frozen before the first execution. The
first run, 03:33:32–03:34:59 UTC, used full-digest-verified existing metadata and
fresh range reads. It covered all 74 files, 71,907 episodes, 22,412,712 rows, with
zero C5 violations and zero integrity/execution errors. `results.json` binds the
actual runner, protocol and input-manifest SHA-256; `range_receipts.jsonl` records
each fetched byte range. Complete original frame Parquet files were not hashed.

The `clean_recovery` invocation used an empty input work directory, no
`--metadata-cache`, and recovered the six metadata shards from their public URLs.
It reproduced all scientific counts and every decoded frame-column digest. This
used the same existing dependency environment, so it establishes fresh public
input recovery, not a clean dependency installation. See `RECOVERY_COMPARISON.json`.

The public-artifact environment was subsequently created separately with
NumPy 1.26.4, pandas 2.1.4, PyArrow 14.0.2 and huggingface_hub 0.34.4. Its first
C5 invocation (`public_env_compat`) used newly public-recovered metadata; the
parallel consumer attempt exposed missing urllib TLS roots before execution.
Both runners now explicitly use certifi roots, retaining TLS verification.
This transport-only amendment changes no predicate, input, selection or tolerance.
Executed pre-amendment scripts are preserved within their run directories.

The amended runner's `public_env_compat2` execution completed from both the fresh
public dependency environment and an empty input work directory. It recovered all
metadata publicly, checked the full population and returned PASS with zero
violations/errors. Every decoded-column digest and scientific count matches the
first run. `CURRENT_RESULTS.json` binds this current-runner acceptance.

Commands (portable interpreter spelling; paths are relative to the repository):

```sh
python experiments/cosmos_c5_20260916/run_c5.py --metadata-cache VERIFIED_METADATA --out NEW_FIRST_RUN
python experiments/cosmos_c5_20260916/run_c5.py --work experiments/work/cosmos_c5_clean_20260916 --out experiments/cosmos_c5_20260916/clean_recovery
python experiments/cosmos_c5_20260916/run_c5.py --work experiments/work/cosmos_c5_public_env_clean_20260916 --out experiments/cosmos_c5_20260916/public_env_compat2
```

Per-run JSON retains actual UTC times and package versions. Scientific
CHECK_FAILED and execution failure are separate statuses. No commit or push was
performed by the experiment worker; project integration records those later.
