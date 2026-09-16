# GX-BASE-20260914: native Great Expectations baseline on the RealDefect-R corpus

**Status: frozen before execution.** This document, `build_artifacts.py`, and `suites.py` are hashed
into `FROZEN_PLAN.json` before `run_gx_baseline.py` is run for the first time; the runner asserts
those hashes. Any later change is recorded as a dated amendment below, never as a silent edit.

## 1. Question

The manuscript claims the corpus provides "an inspectable basis for evaluating checks of robot-data
meaning". The review correctly objects that no external validator has ever been run on it. This
experiment runs one — Great Expectations — natively, and asks:

> For each corpus case, does a natively executed GX validation of the **faulty** data artifact fail,
> does the same suite **accept** the paired repaired/reference artifact, and how does the answer
> change with the information supplied to the validated batch?

The conclusion this experiment can support is about **what information a check needs**, not about
generic incapability of any framework. Great Expectations is deliberately also given an arm in which
the domain knowledge is expressed *inside* GX, to show the framework can carry it.

## 2. Executor and versions

All validations run natively through `great_expectations` — `ExpectationSuite`,
`ValidationDefinition`, the pandas execution engine, and (for the enriched arm) Expectation classes
registered into GX by subclassing `ColumnMapExpectation` / `BatchExpectation`. No custom Python
predicate is evaluated outside GX and then reported as a GX result.

- Great Expectations 1.6.3, pandas 2.1.4, NumPy 1.26.4, PyArrow 14.0.2, CPython 3.12.3, macOS arm64.
- Isolated, git-ignored virtual environment `.venv-gx/`.
- Artifact construction runs in a separate environment `.venv-build/` (NumPy 1.26.4, h5py 3.11.0,
  PyArrow 14.0.2, pandas 2.1.4) so that artifact building cannot perturb the validated environment.

## 3. Case eligibility and denominator

A case is **eligible** when its replay yields a columnar data artifact with more than one record that
a downstream consumer reads, and both a faulty and a repaired/reference version of that artifact can
be materialised from hash-verified inputs.

| Case | Eligible | Artifact | Records |
|---|---|---|---|
| RD1 | yes | windowed-sample selection table | 118 |
| RD2 | yes | episode metadata table (`meta/episodes` fields) | 5 |
| RD3 | **no** | a single 4-element configuration constant | 1 |
| RD4 | yes | stored quaternion instance table | 23,833 |
| RD5 | yes | world-frame displacement table | 9,666 |
| RD6 | yes (derived) | normalised gripper column, long format | 73,802 |
| RD7 | yes | request / returned-frame table | 2,238 |
| RD8 | yes | `norm_stats` per-component statistics table | 28 |

**Denominator is 7, not 8.** RD3 is excluded as **NA — not a multi-record data artifact**: its
replay object is one grasp-offset constant in a task configuration file, not a dataset. Its
population analogue, the same WXYZ/XYZW convention defect over 23,833 stored instances, is RD4,
which is included. RD3 is never counted as a detection miss.

**RD6 is a derived target and is labelled as such everywhere.** RD6's frozen replay unit is two
encoder anchors plus one illustration row, which is not a multi-record artifact. The artifact a
consumer actually reads is the normalised gripper column, so we apply the old and repaired
normalisations to the `right_gripper` and `left_gripper` components of all 36,900 hash-verified rows
of the same pinned dataset (`physical-intelligence/aloha_pen_uncap_diverse`, revision
`e82d8b40b8ac66c0b40273dd80a077dfc40b732e`, recovered as the RD8 projection), plus the two
calibration anchor rows. **This is not part of RD6's frozen replay scope and does not change RD6's
recorded result.**

Deviation from the review's suggested 8x3 table is therefore: 7 cases instead of 8 (RD3 excluded with
a stated reason rather than forced into a misleading denominator), and the three columns are three
*information* arms rather than three unrelated tools, because the question the manuscript must answer
is what information a check needs.

## 4. Batches

Every case materialises four tables:

- `faulty_public` / `reference_public` — only the columns a downstream consumer obtains from the
  artifact itself.
- `faulty_enriched` / `reference_enriched` — the same rows plus the case's reference-information
  columns (source convention tuples, episode bounds, base transform, calibration anchors, expected
  frame identity, independently computed reference statistics).

Arms A and B validate the `_public` tables. Arm C validates the `_enriched` tables. The arms
therefore differ in the **information available to the check**, with the framework, executor,
versions and scoring held constant.

Every table is written to Parquet and SHA-256 hashed into `artifact_manifest.json` before validation.

## 5. Arms

### Arm A — declared-schema structure and range

Expectations chosen only from what the artifact's own declared schema or published documentation
states: column set, dtype, non-null, declared value bounds (normalised action range, unit quaternion
norm, non-negative standard deviation, declared image shape and dtype, declared controller output
range), declared row count, and schema-declared relations between columns already present in the
batch. No parameter may be computed from the faulty batch. No knowledge of the defect is used.

### Arm B — disjoint-reference-fitted suite

Expectation *parameters* are fitted on a reference fixture disjoint from the validated batch, then
executed natively against the held-out faulty and reference records. The generation rule is fixed and
applied mechanically and identically to every case:

For each column of the fitting fixture emit
1. `ExpectTableColumnsToMatchSet` over the fixture's column set (once per suite);
2. `ExpectColumnValuesToNotBeNull` when the fixture column has no nulls;
3. `ExpectColumnValuesToBeOfType` with the fixture column's dtype;
4. for numeric columns, `ExpectColumnValuesToBeBetween(min, max)` at the fixture's observed extremes;
5. for boolean/string/object columns with at most 20 distinct fixture values,
   `ExpectColumnDistinctValuesToBeInSet` over that set.

No per-case tuning, no hand-picked thresholds. Fitting fixtures, declared in advance:

| Case | Fitting fixture | Validated records |
|---|---|---|
| RD1 | reference rows 0–58 | rows 59–117, both branches |
| RD2 | reference episodes 0–2 | episodes 3–4, both branches |
| RD4 | reference instances, first half by instance index | second half, both branches |
| RD5 | reference rows of the first half of demos | remaining demos, both branches |
| RD6 | reference data rows, first half by source index | second half plus anchor rows, both branches |
| RD7 | reference rows of videos 0–4 | videos 5–9, both branches |
| RD8 | the dataset's own published companion `meta/stats.json` (an independently published reference; the 28-record artifact has no exchangeable split) | all 28 records, both branches |

### Arm C — information-enriched suite with native GX custom expectations

The case's reference information is present in the enriched batch, and the semantic rule is evaluated
by an Expectation class registered into GX. Built-in Expectations are used where one suffices;
otherwise a `ColumnMapExpectation` or `BatchExpectation` subclass with a pandas metric provider is
registered. The registered type name of every expectation is recorded in the results.

## 6. Scoring rules (frozen)

Per (case, arm, branch):

- `EXECUTED` or `FAILED_TO_EXECUTE` with the captured error. A failure to execute is never reported
  as a detection or as a non-detection.
- Faulty batch: **DETECTED** if the suite's overall `success` is `False`, otherwise **NOT_DETECTED**.
- Reference batch: **ACCEPTED** if the suite's overall `success` is `True`, otherwise
  **FALSE_POSITIVE**.
- A case counts as *detected by an arm* only when the faulty batch is DETECTED **and** the paired
  reference batch is ACCEPTED. Faulty-DETECTED with reference-FALSE_POSITIVE is recorded as
  **ALWAYS_FAIL** and does **not** count as detection. This paired control is what distinguishes
  detection from a suite that rejects everything.
- Every individual expectation result — type, success, unexpected count, element count — is recorded.
- `mostly` is left at the GX default of 1.0 everywhere.

## 7. Pre-declared prohibitions

1. No Arm A or Arm B parameter may be derived from the batch being validated.
2. No profiling of an evaluation target followed by validation of that same target.
3. No expectation is added, removed, or retuned after an outcome is observed. If the run reveals a
   defect in the harness, the fix is recorded as a dated amendment with both outcomes preserved.
4. Custom Python checks are never described as native GX executions.
5. Outcomes that contradict the expectation that "generic validators miss these defects" are reported
   in full. Prior to execution we already expect at least RD5 to be detected by Arm A, because the
   robomimic controller configuration declares an output displacement range and the faulty world
   delta exceeds it; that possibility is recorded here, before running, so it cannot be presented as
   a post-hoc framing.

## 8. Amendments

Both superseded runs are preserved verbatim under `failed_runs/`. `FROZEN_PLAN_HISTORY.jsonl` holds
each superseded frozen plan. No case, arm, batch, split rule, Arm A or Arm B parameter, or scoring
rule was changed by any amendment.

**A1 (2026-09-15, after run 1 — total failure to execute).** All 21 validations failed before
producing any verdict. Two harness defects: GX binds an `Expectation` object to the first suite it is
added to, so a single object could not be reused across the faulty and reference validations, and
`ExpectWorldDeltaToEqualRotatedDisplacement` did not declare `base_rotation` / `base_translation` as
pydantic fields. Fixed by constructing a fresh, identically parameterised expectation set per
validation and declaring the fields. GX usage analytics were also disabled at this point
(`GX_ANALYTICS_ENABLED=false`, `DO_NOT_TRACK=true`) so that nothing about the private artifacts is
transmitted; run 1 had emitted PostHog traffic.

**A2 (2026-09-15, after run 2 — three defects, two of which produced spurious verdicts).**

1. The RD6 calibration-anchor metric returned a non-boolean pandas Series, which raised inside GX's
   map-condition machinery. Its two `ALWAYS_FAIL` verdicts in run 2 were therefore spurious. Fixed by
   returning a clean boolean Series.
2. The RD4 angular expectation used `max_error_deg=1e-6`. The trace/arccos angular formulation has a
   float64 conditioning floor near the identity rotation, measured at **2.958e-6 deg** on the
   correctly converted reference batch, so the reference was rejected for numerical, not semantic,
   reasons. Raised to `1e-3` deg — still five orders of magnitude below this case's defect signal,
   every missed instance of which exceeds 90 deg. After the change the expectation flags exactly
   15,141 of 23,833 instances on the faulty batch, the same count the frozen RD4 record reports, and
   0 on the reference batch.
3. The runner scored an expectation whose metric raised as `success=False`, i.e. as a **detection**.
   Raised exceptions are now surfaced and scored as `FAILED_TO_EXECUTE`. This defect is the reason
   defect 1 was visible as a verdict rather than as an error.

A2 also raised RD4's `max_error_deg` from 1e-6 to 1e-3 degrees. **That part of A2 was wrong and is
withdrawn by A3**: it moved a decision boundary after observing an outcome, which no argument about
the size of the defect signal makes acceptable.

**A3 (2026-09-15, after supervisor review of run 3).** RD4's Arm C decision boundary is restored to
its original **1e-6 degrees**. The real defect was the *evaluation*, not the boundary: the
trace/arccos rotation distance loses roughly half the available precision near the identity rotation.
It is replaced by the mathematically equivalent relative-quaternion geodesic distance
`2*atan2(||vec(q_src^-1 (x) q_tgt)||, |w(q_src^-1 (x) q_tgt)|)`, which is well conditioned at small
angles and invariant under the q / -q double cover. Verified in `verify_rotation_distance.py`
(`rotation_distance_verification.json`, status PASS, 20,000 samples per check, seed 20260915):

| Check | Result |
|---|---|
| identical orientations across the two conventions | max 1.46e-14 deg |
| bit-identical inputs in one convention | exactly 0 |
| q versus -q | max 1.46e-14 deg; exactly 0 for bit-identical inputs |
| rotations at 0.5x, 0.99x, 1.01x and 2.0x the unchanged 1e-6 deg boundary | all classified correctly; recovered angle accurate to 2e-8 relative |
| recovery of the applied angle over 1e-3 to 179 deg | 2.0e-11 max relative error |
| agreement with the superseded formulation where it is well conditioned (>= 1 deg) | 3.5e-12 max relative difference |
| superseded formulation on identical orientations | max **2.70e-6 deg**, above the 1e-6 boundary for **5.6%** of random orientations |

At the restored 1e-6-degree boundary the expectation flags **15,141 of 23,833** instances on the
faulty batch — the same count the frozen RD4 record reports — and **0 of 23,833** on the reference
batch. The verdict is unchanged from run 3, so the A2 threshold change had no effect on any reported
outcome, but the reported boundary is now the original one and is genuinely met.

Run 3 and its harness are preserved under `failed_runs/gx_baseline_results__run3_superseded_by_A3.*`
and `harness_snapshots/pre_A3_run3/`.

**Snapshot completeness, stated plainly.** Byte-level source snapshots exist only from
`harness_snapshots/pre_A3_run3/` onward. For runs 1 and 2 only the SHA-256 of each superseded
`PROTOCOL.md` / `build_artifacts.py` / `suites.py` (in `FROZEN_PLAN_HISTORY.jsonl`) and the full
result JSON survive; the superseded source bytes were not retained. Amendments A1 and A2 describe
the exact changes, but that is a description, not an artifact, and it is not claimed otherwise.

**Not all thresholds were frozen before all observations.** The A2 threshold change was made after
observing run 2 and was withdrawn by A3. The boundary now in force is the pre-execution one.

## 9. Outcome of the executed run

Executed 2026-09-15 under the plan above and amendments A1-A3: 7 cases x 3 arms x 2 branches = 42
native GX validations, all `EXECUTED`.

`ALWAYS_FAIL` below is the machine label for **both of this case's two paired batches rejected**. It
describes the two batches actually tested. It is not evidence that the suite would reject every
possible batch.

| Case | Arm A declared schema | Arm B disjoint-reference fitted | Arm C information-enriched |
|---|---|---|---|
| RD1 | MISSED | both paired batches rejected | DETECTED (118/118 faulty, 0/118 reference) |
| RD2 | MISSED | both paired batches rejected | DETECTED (1/5, 0/5) |
| RD4 | MISSED | both paired batches rejected | DETECTED (15,141/23,833, 0/23,833) |
| RD5 | **DETECTED** | both paired batches rejected | DETECTED (9,666/9,666, 0/9,666) |
| RD6 | both paired batches rejected | both paired batches rejected | DETECTED (2/73,802, 0/73,802) |
| RD7 | MISSED | both paired batches rejected | DETECTED (200/2,238, 0/2,238) |
| RD8 | MISSED | **DETECTED** | DETECTED (28/28, 0/28) |

Arm A: 1 detected, 5 missed, 1 with both paired batches rejected. Arm B: 1 detected, 6 with both
paired batches rejected. Arm C: 7 detected, 0 missed, 0 false positives on the paired reference
batches. No arm produced a failure to execute in this run.

Three observations that were not the expected story and are reported as found:

- **RD5 is detected by Arm A.** The robomimic controller configuration declares an output
  displacement range of +/-0.05 m per axis; the faulty world delta has norm up to 1.127 m and the
  reference up to 0.072 m. This was pre-declared in section 7 before the run.
- **RD8 is detected by Arm B**, because the dataset publishes its own companion `meta/stats.json`,
  and the old branch's 3.1%-coverage statistics fall outside the range fitted from it. Where a
  disjoint published reference exists, a mechanical range suite is sufficient.
- **On six of seven cases Arm B rejected both paired batches.** Minimum/maximum bounds fitted
  mechanically on one disjoint half did not cover the held-out half of the *reference* artifact
  either, so on these pairs the suite rejected correct data as readily as faulty data and its
  faulty-batch failures cannot be read as detections. This is a direct, negative answer to the
  "auto-profile then validate" suggestion for these seven artifacts; it is a statement about the
  batches tested, not a general claim about profiling.

**Reporting note.** GX serialises only an Expectation's domain kwargs into
`expectation_config.kwargs`, so Arm C success keys such as `max_error_deg` and `atol` appear as
`None` in `gx_baseline_results.json`. Their values live in `suites.py`, whose SHA-256 is recorded in
`FROZEN_PLAN.json` for the run, and in `harness_snapshots/`.

The Arm C result is the substantive one: every one of the seven semantic rules — clamp-and-pad row
identity, global interval chaining, quaternion convention, displacement-versus-position frame
handling, calibration anchors, requested-time frame correspondence, and reference-statistic agreement
— is expressible as a registered GX Expectation and detects its case exactly, with zero false
positives on the paired repaired artifacts. The limiting factor is the reference information, not the
validation framework.
