# Cosmos3-DROID frame-level findings (P0-3)

Executed 2026-09-15 under `PROTOCOL.md` and its amendment B1. Raw log: `run2_accepted.log`.
Machine-readable result: `cosmos_frame_check_results.json`.

## 1. Coverage achieved

Complete, not sampled. Every eligible file was processed within the budget, so no sampling inference
is needed for the columns read.

| | |
|---|---:|
| eligible frame-level data files | 74 |
| processed | **74** |
| unprocessed / skipped | 0 |
| episodes covered | **71,907** |
| frames covered | **22,412,712** |
| columns read | `episode_index`, `task_index`, `frame_index`, `index` |
| image bytes read | none — the three `observation.image.*` features are `dtype: video` |
| HTTP range requests | 370 |
| bytes fetched this run | 129,323,418 (123.33 MiB) |
| prior transfer charged (probes + aborted attempt 1) | 190,000,000 |
| total charged against the 1 GiB ceiling | 319,323,418 (304.53 MiB) |
| fetch refused by the ceiling | none |

## 2. Frame-level result

Every one of the **71,907** episodes passes all four frame-level checks. Zero failures of any kind.

| Check | Passing |
|---|---:|
| C1 frame count equals the declared `length` | 71,907 / 71,907 |
| C2 frame `index` values equal exactly `[dataset_from_index, dataset_to_index)` | 71,907 / 71,907 |
| C3 `frame_index` equals exactly `range(0, length)` | 71,907 / 71,907 |
| C4 frame `task_index` constant and resolving through the companion table to the episode's stored task text | 71,907 / 71,907 |

**Scope.** This is a complete-population statement about the four columns listed above, in the 74
declared frame-level files of both splits, at revision `dabaaffe428d67cf93fd355b82658934ee59bfec`.
It is **no observed frame defect in that scope**. It is not a claim about columns that were not read,
about the video streams, or about any other revision, and it must not be written as "frame-level
columns are unaffected".

## 3. The summaries disagree with the frames they summarise

The discriminating comparison. For each episode, the per-episode summary fields were compared with
the values actually observed in that episode's frames:

| | Agrees with frames | Differs from frames |
|---|---:|---:|
| `stats/episode_index/{min,max,mean}` | 10,405 | **61,502** |
| `stats/task_index/{min,max,mean}` | 21,054 | **50,853** |
| `stats/episode_index/count` | **71,907** | 0 |

The 61,502 and 50,853 figures reproduce the counts the existing metadata audit reports, but they are
now established from a different direction — against the actual frame contents rather than against
metadata self-consistency. The `count` field is correct everywhere; only the identity summaries are
not.

This sharpens the characterisation of the defect: in this release the per-episode identity summaries
do not describe the episode's own frames, while the frames themselves carry globally correct
identifiers. The inconsistency is confined to the summary layer, for the columns examined.

## 4. Consumer exposure, kept separate from harm

Pinned consumer evidence: LeRobot **v0.6.1**, commit `7e241bd630a3719a56157a497ce5d08f244784f1`,
whose `CODEBASE_VERSION` is `v3.0` and which therefore reads this release's format. Sources
downloaded and hashed into `../work/cosmos_frame_sample_20260914/lerobot_pinned/`.

In `src/lerobot/datasets/aggregate.py`, the episode-table merge path reads
`stats/episode_index/{min,max,mean,q01,q10,q50,q90,q99}` and `stats/index/*` and **adds a constant
offset** to them, on the stated assumption that they describe the pre-merge global identifiers. Its
own comment says: *"Per-episode stats still describe the pre-merge values of the bookkeeping columns
reindexed above. index/episode_index shift by a constant; task_index is relabeled, so recompute it
from the episode's (stable) task strings via the unified tasks table."* For `stats/task_index/*` the
same path **recomputes** the values from the episode's stored task strings, so that field is
regenerated rather than shifted.

Consequence within this evidence: a released code path reads the affected `episode_index` summaries
and transforms them under an assumption this release violates. That is **consumer exposure**.

Four things stay distinct and are never substituted for one another:

1. **metadata mismatch** — established for 61,502 records, now also against the frames themselves;
2. **actual frame defect** — none observed, in the complete scope of section 2;
3. **consumer exposure** — evidenced above at a pinned revision and line range;
4. **measured downstream harm** — **not measured.** No claim is made that any published model, any
   training run, or any decision was affected. Nothing here demonstrates that.

## 5. Failed and superseded attempts

- **Attempt 1, aborted after 52 of 74 files.** Its byte accounting wrapped the file object's
  `read()` and therefore measured logical bytes returned to PyArrow rather than HTTP bytes fetched;
  the handle used fsspec's default 5 MiB read-ahead cache, so its ceiling enforcement was unproven.
  Stopped on peer review, preserved under `aborted_attempts/`, and its partial outcomes are not used.
  Amendment B1 documents the correction and charges its estimated transfer to the budget.
- **The accepted run raised `KeyError: 'mebibytes_transferred'` in its final `print`,** after
  `cosmos_frame_check_results.json` had already been written. The crash is in the console summary
  only; no measurement, comparison or file was affected. The stale key name was a leftover from the
  pre-B1 accounting. Fixed in the script; the results were not recomputed, because doing so would
  re-download 123 MiB to change nothing. The raw log in `run2_accepted.log` ends with that traceback.
