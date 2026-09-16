# COSMOS-FRAME-20260914: frame-level check of the Cosmos3-DROID identity inconsistency

**Status: frozen before any frame-level outcome was inspected.** The eligible set, the selection
rule, the seed, the download ceiling, the comparisons and the reporting rules below were written
first; only a transfer-cost probe on a single file (`success/data/chunk-000/file-000.parquet`,
1.71 MiB for four columns) was run beforehand, to size the budget.

## 1. Question

The existing audit establishes that 61,502 of 71,907 **per-episode summary** records carry
lab-local rather than global identities. It does not establish what the **frame-level** columns
contain. The review asks, correctly, whether the inconsistency reaches the data a trainer actually
reads.

Four things are kept strictly separate throughout and are never substituted for one another:

1. **metadata mismatch** — a per-episode summary disagrees with the global identifiers;
2. **actual frame defect** — a frame-level column disagrees with the companion tables;
3. **consumer exposure** — some released code path reads the affected field;
4. **measured downstream harm** — a demonstrated degradation in a trained model or a decision.

This experiment addresses 1, 2 and 3. It does **not** address 4, and no claim about 4 is made.

## 2. Pinned inputs

- Dataset `nvidia/Cosmos3-DROID` at revision `dabaaffe428d67cf93fd355b82658934ee59bfec`.
- Local metadata already downloaded and full-file SHA-256 verified in the 2026-09-12 audit
  (`experiments/results/20260912-cosmos-metadata/verified_inputs.json`): six episode shards, two
  `info.json`, two `stats.json`, two `tasks.parquet`. This run re-verifies those digests before use.
- Format declared by both splits: LeRobot `codebase_version` **v3.0**,
  `data_path = data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet`,
  `data_files_size_in_mb = 100`, `fps = 15`.
- The three `observation.image.*` features are declared `dtype: video`, so the frame-level Parquet
  files carry **no image bytes**; they hold 17 low-dimensional columns.

## 3. Eligible set

Every distinct `(data/chunk_index, data/file_index)` pair referenced by the episode tables:

| Split | Episodes | Declared frames | Distinct data files |
|---|---:|---:|---:|
| success | 57,639 | 18,691,281 | 62 |
| failure | 14,268 | 3,721,431 | 12 |
| total | 71,907 | 22,412,712 | **74** |

The sampling unit is the **data file**, not the episode: Parquet column chunks are per file, so
selecting individual episodes would not reduce transfer.

## 4. Budget and selection rule

New-network-download ceiling: **1 GiB**, enforced by a counting wrapper that tallies every byte
returned by the remote file object; the run aborts if the tally would exceed it.

Only four columns are read: `episode_index`, `task_index`, `frame_index`, `index`. The measured cost
of one file is 1.71 MiB against a 77 MiB file, so complete coverage of all 74 files is projected at
roughly 127 MiB, inside the ceiling.

**Primary plan: complete coverage of all 74 eligible files.** If complete coverage is achievable
within the ceiling, no sampling inference is required for these columns and the result is a
population statement about them.

**Pre-registered fallback, in force only if the ceiling is reached:** process files in the
deterministic order produced by `numpy.random.default_rng(20260915).permutation` over the eligible
files sorted by `(split, chunk_index, file_index)` with `success` before `failure`, stop at the last
file that fits, and report every excluded file by name together with the episodes and frames they
would have covered. Blocked expansion is reported, never silently skipped.

## 5. Comparisons

For every episode whose frames appear in a processed file:

- **C1 frame count** — number of frame rows with that `episode_index` equals the episode table's
  declared `length`.
- **C2 global row interval** — the episode's frame `index` values equal exactly
  `range(dataset_from_index, dataset_to_index)`.
- **C3 within-episode frame order** — `frame_index` equals exactly `range(0, length)`.
- **C4 frame task identity** — every frame `task_index` of the episode is constant and resolves,
  through that split's companion `tasks.parquet`, to a task string equal to the episode table's
  stored `tasks` value.
- **C5 summary versus frames** — the affected per-episode summaries
  `stats/episode_index/{min,max,mean,count}` and `stats/task_index/{min,max,mean}` are compared with
  the values actually observed in that episode's frames.

C5 is the discriminating comparison. C1–C4 say whether the frame columns are globally correct; C5
says whether the per-episode summaries describe those frames.

An episode is **frame-consistent** when C1–C4 all hold.

## 6. Consumer exposure

Pinned consumer evidence is gathered from LeRobot at tag **v0.6.1**, commit
`7e241bd630a3719a56157a497ce5d08f244784f1`, whose `CODEBASE_VERSION` is `v3.0` and therefore reads
this release's format. Files downloaded and hashed into `lerobot_pinned/`.

Identified consumer of the affected per-episode summary fields:
`src/lerobot/datasets/aggregate.py`, in the episode-table merge path — it reads
`stats/episode_index/{min,max,mean,q01,q10,q50,q90,q99}` and `stats/index/*` and **adds a constant
offset** to them, on the stated assumption that they describe the pre-merge global identifiers; and
it **recomputes** `stats/task_index/*` from the episode's stored task strings through the unified
task table.

This is recorded as consumer exposure only. Whether any published model was trained through this
path, and with what effect, is not evidenced here.

## 7. Reporting rules (frozen)

1. Frame-level conclusions cover exactly the files processed. If coverage is complete for the four
   columns read, that is stated as complete **for those columns in those files**, and never
   generalised to columns that were not read or to other revisions.
2. A result of no observed frame defect is reported as no observed frame defect, never as "frame
   columns are unaffected".
3. Exact `n` is reported at every level: files, episodes, frames.
4. The four categories in section 1 stay separate in every sentence of the write-up.
5. Any mismatch found is reported in full, including mismatches that contradict the expectation that
   the frame columns are correct.
6. Measured bytes transferred are reported.

## 8. Upstream reporting

An upstream issue with evidence and a reproducer is drafted locally at
`research/20260914-claude-p0/UPSTREAM_ISSUE_DRAFT.md`. It is **not posted**. No maintainer, organiser
or forum is contacted. Posting requires a separate, explicit user decision after reviewing the draft.

## 9. Amendments

**B1 (2026-09-15, after peer implementation review; attempt 1 aborted).** Attempt 1's transfer
accounting was wrong and its ceiling enforcement was unproven. It wrapped the file object's `read()`
and counted bytes **returned to PyArrow**. Those are logical bytes. The handle was opened with
fsspec's default `cache_type="readahead"` and a 5 MiB block size, so the bytes actually fetched over
HTTP were larger and unmeasured, and no claim about the 1 GiB ceiling was supported.

Attempt 1 was stopped after 52 of 74 files with no frame-level failure observed. Its log and its
exact source are preserved under `aborted_attempts/`. Its partial outcomes are **not** used as
results; the corrected run recomputes everything from scratch over the same frozen file set.

The corrected implementation:

1. Accounting is hooked at the fsspec cache's `fetcher`, which is `HfFileSystemFile._fetch_range`.
   That method issues exactly one HTTP Range GET and returns `response.content`, so the length of its
   return value is the HTTP response body transferred for that request.
2. The ceiling is enforced **before** each fetch: a range GET larger than the remaining budget is
   refused and raises, so the ceiling cannot be crossed rather than merely detected afterwards.
3. Files are opened with `cache_type="none"`. Measured on
   `success/data/chunk-000/file-000.parquet`: default read-ahead fetches **3,368,374 bytes** in 3
   range requests, while `cache_type="none"` fetches **1,796,138 bytes** in 5 requests — the exact
   bytes the four columns need. `block_size=0` was deliberately **not** used: that makes
   `huggingface_hub` select a non-seekable streaming implementation, which Parquet footer reads
   require to be seekable. `cache_type="none"` keeps `AbstractBufferedFile` with `BaseCache`, and the
   handle was verified to report `seekable() == True` and to read correctly.
4. What the figure is: HTTP response bodies for content range GETs. What it excludes: request and
   response headers, TLS framing, redirect responses, and the small path/size metadata calls fsspec
   makes. It is reported as a close lower bound on wire traffic, not as an exact wire measurement.
   No claim of exact network transfer size is made.

**Prior transfer, charged to the budget.** Earlier network use on this question is debited from the
ceiling up front, so the 1 GiB covers the whole investigation and not just the accepted run:

| Item | Bytes | Basis |
|---|---:|---|
| sizing probe, read-ahead | ~3,368,374 | same file, measured |
| accounting calibration, two opens of one file | 5,164,512 | measured |
| aborted attempt 1, 52 files, read-ahead | ~171,700,000 | 87.3 MiB logical x the 1.876 read-ahead inflation factor measured on a representative file |
| **charged** | **190,000,000** | rounded up |

The aborted attempt's actual transfer is an estimate, not a measurement, precisely because its
accounting was the defect. A loose upper bound is also available: every file has one row group and
the same schema, read-ahead issued 3 range requests per file, and each request fetches at most its
own length plus the 5 MiB block, giving at most ~15 MiB per file and ~780 MiB for 52 files — still
inside the ceiling, though not a tight bound. The 190,000,000-byte charge is the calibrated estimate.

**Selection is unchanged.** Same eligible set, same seed, same permutation, same complete-coverage
plan, same comparisons. B1 corrects implementation and accounting only. No selection or comparison
was altered after seeing an outcome.

## 10. Scope limits on the Arm-analogous reasoning

Recorded here because the same caution applies to how this experiment's results are written up: a
negative frame-level finding is a statement about the four columns read from the files processed at
this revision. It is not a statement about columns that were not read, about other revisions, or
about the video streams, and it is never phrased as "frame-level columns are unaffected".
