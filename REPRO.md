# Reproduction from a fixed public snapshot

Use a tagged release/commit, not a moving checkout. The complete snapshot includes newly added
GX, RD8-PD, Cosmos, C5 and scoped consumer-replay code and results; its identity is not merely the historical `5d50746`
source pin. Run `verify_public.py` first. Every manifest file must exist with its exact SHA-256.

## Fast native GX replay from the public tag

Download the fixed snapshot without a GitHub login:

```sh
curl -q -fL https://github.com/xch3177-publicrepo/RealDefect-R/archive/refs/tags/public-v1.1.0-20260919.tar.gz -o realdefect-r-public-v1.1.0-20260919.tar.gz
mkdir realdefect-snapshot
tar -xzf realdefect-r-public-v1.1.0-20260919.tar.gz --strip-components=1 -C realdefect-snapshot
cd realdefect-snapshot
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run_public.py --suite gx --download --work-dir /absolute/new-empty-gx-directory
```

The `gx` route verifies the snapshot and projection identities, recovers three pinned inputs
(32,561,298 bytes in total), reconstructs all seven paired fixtures and executes the unchanged native
GX suites: **7 pairs × 3 arms × 2 branches = 42 branch validations**. It checks all six GX comparisons
from the extended suite: tally, case identities, denominator and preservation of arms, cases and
suite-code digest. `GX_EXECUTION_COVERAGE.json` verifies the complete Cartesian coverage and binds
the source snapshot's `MANIFEST.json` digest. `SCIENTIFIC_COMPARISON.json` records the six outcomes.
These files and raw logs are written to the external work directory, never back into the downloaded
snapshot. The GX route does not execute all eight upstream case runners, RD8-PD or Cosmos.

## Installation and one-command execution

Tested interpreter: CPython 3.12. Install `requirements.txt` in a new virtual environment. System
requirements are curl and (only for rebuilding the paper PDF) a standard TeX distribution with
pdfLaTeX. The unmodified IEEEtran class is included under its original terms.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_public.py
.venv/bin/python run_public.py --suite all --download --work-dir /absolute/new-empty-directory
```

The runner verifies the snapshot, copies it into the new directory, creates fresh logs and isolated
cache directories, then executes calculations there. It never imports an author's checkout or
uses an external historical result cache. Shipped reference results remain untouched. Use a new
empty work directory for every run.

| Suite | What is actually executed | Network |
|---|---|---|
| `gx` | Integrity, canonical RD8 projection hashes, three recovered inputs, seven paired fixture constructions, native GX and six GX comparisons | Hash-verified HDF5/source inputs |
| `smoke` | Integrity, canonical RD8 input-array hashes, RD3/RD6 scoped replays, RD8 condition-matrix extraction, figure generation | None |
| `cases` | Smoke plus all eight scoped case runners, RD7/RD8 post-run audits and cross-case checks | Hash-verified source/HDF5/videos |
| `extended` | Smoke plus fixture construction, native GX, all-order RD8-PD, Cosmos metadata/independent checker, C1–C4, C5 and the scoped consumer replay | Pinned metadata and identity-column range reads |
| `all` | Cases and extended suites | Both |

Every command has a raw transcript under `logs/` and a record in `commands.json`.
`RUN_SUMMARY.json` distinguishes expected outcomes from `SCIENTIFIC_MISMATCH` and `INFRASTRUCTURE_OR_EXECUTION_FAILURE`. Existing historical counts are reference comparisons, never substituted for execution.

## Expected scientific outcomes

- RD1, RD2, RD3, RD5, RD6 and RD7 meet their scoped paired replay rules. RD4 meets its single-path
  coverage-audit rule; no repaired upstream counterpart is executed. These are not seven full
  upstream application replays.
- RD8 normally exits 1; the case wrapper accepts that exit **only after checking** the produced
  result has `status == NOT_REPRODUCED` and `correctness.reproduced == false`. Full coverage does
  not change the original `1e-5` criterion; 24/44 mean/std conditions remain outside tolerance.
- GX evaluates seven faulty/reference pairs (RD3 not applicable); declared separates 1, fitted 1,
  informed 7. RD4's reference is constructed, RD6's anchor predicate applies only 2 of 73,802 rows,
  and RD8 uses the primary natural-order batch 32 condition. Suites change predicates and references.
- Cosmos metadata: 71,907 episodes; 61,502 identity-summary mismatches. C1–C4 check 22,412,712 frame rows
  over 74 pinned files. The new C5 has its own result/protocol and must not be conflated with these
  earlier checks. The scoped consumer replay executes the pinned `update_meta_data` function on four
  preselected records: the original-summary condition has 16 violating fields across two affected
  records; the summary-corrected condition has none. Neither control violates the rule. This is
  function-level controlled execution, not a full merge or downstream model-impact experiment.

## Input bytes and runtime

The 16 September 2026 clean `cases` run verified 15 input objects totaling **43,486,336 bytes**.
Of these, 13 were newly downloaded: **43,199,368 bytes (43.20 MB)**, comprising two HDF5 files
(32,547,984 bytes), the converter source (13,314 bytes) and ten videos (10,638,070 bytes).
The remaining 286,968 bytes are the bundled PushT/LIBERO metadata. These are full-file payload
sizes, not a network wire-byte measurement; small additional upstream Python-source fetches,
redirects, headers and dependency installation are excluded.

The separate Cosmos audit recovers 12 full metadata/companion objects totaling **389,225,505 bytes**.
The accepted C1–C4 run fetched 129,323,418 bytes in identity-column HTTP response bodies; the C5
public-environment run fetched 128,800,948 bytes. These range-read amounts exclude headers and TLS
framing. The consumer replay has separate range receipts. The `all` runner currently re-fetches GX's
two HDF5 inputs and converter (32,561,298 bytes) after the case suite; do not add the case inventory
alone and call it a whole-workflow network total.

CPU time depends on decoding and the all-order RD8 calculations. Consult the packaged acceptance
transcripts and `commands.json` for actual durations and completed steps; an input download's
completion does not by itself establish that the corresponding scientific check completed.

## What is bundled and what is fetched

Bundled: 123 lossless low-dimensional ALOHA NPZ projections, 123 input-projection checkpoints and four
metadata files; a derived one-row ALOHA fixture; small PushT/LIBERO metadata; public pinned source
files; numerical results and tiny derived records needed to construct the GX fixtures. These are
**redistributed** and individually covered by [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The input checkpoints bind projection identity, not previously completed analysis state.

Recovered over pinned public URLs: IsaacLab/robomimic HDF5 inputs, pinned converter body, 10 Isaac-GR00T
videos and Cosmos metadata. Large assets retain upstream terms and are not in this snapshot.
The original 72.1 GB ALOHA image-bearing Parquets are not downloaded for RD8: the bundled projections
are verified against both file hashes and the historical canonical array inventory. Cosmos range
reads bind dataset revision/path and decoded column hashes, not full-file Parquet digests.

## Fresh GX fixture identities

Fixture construction can change container metadata or generation timestamps across environments.
The public runner preserves the historical freeze and records an explicitly named local rerun
amendment for the newly generated fixture manifest. It does not change suites, semantic rules,
case counts or thresholds. New result bytes are separate from shipped expected results.

## Paper reconstruction

```sh
python3 scripts/make_figures.py
python3 scripts/build_paper.py
```

Scientific values bind to the packaged result JSONs in `paper/FIGURE_BINDING.json`. The public
builder regenerates bindings after path redaction, so hashes identify public bytes. pdfLaTeX may
embed timestamps; compare figures/text/page layout rather than assuming byte-identical PDFs.

## Packaged acceptance records

When present, `acceptance/` contains only the selected, already sanitized execution transcripts,
environment lock, scientific comparisons and executed-source hashes listed in
`ACCEPTANCE_IMPORT.json`. It does not contain the test runtime, cache or downloaded working data.
The acceptance records identify the candidate and executed code they actually checked; the public
file manifest and source mapping identify this final snapshot.


Post-publication access tests are separate release evidence: their records bind the downloaded tag,
source/archive identity and manifest hash. They are not inserted retroactively into the tested fixed
snapshot. This keeps the snapshot identity stable and avoids a self-referential acceptance hash.

## Source-prefix regrouping and scientific figures (v1.1.0)

The additional Cosmos grouping analysis is a separately versioned re-analysis of the same 12 hash-verified metadata objects. It is not a new consumer experiment. Follow [the grouping protocol and commands](research/20260919-round6-evidence-charts/cosmos_group_analysis/README.md) to recover metadata and reproduce the 13-group table. This optional analysis is separate from the existing `all` suite.

`python scripts/make_figures.py` regenerates the pipeline map, the 44-condition RD8 CSV and vector figure, the Cosmos joint/source-group figure, and three tables from packaged evidence. It uses only the Python standard library. The original seven identity-summary predicates and eight consumer target fields remain distinct.
