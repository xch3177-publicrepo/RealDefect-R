# RD8 findings (P0-2)

RD8's frozen record is unchanged: **NOT_REPRODUCED** — 24 of 44 conditions exceed the retained
absolute 1e-5 mean/std tolerance against the independent float64 full-array reference. Everything
below decomposes and diagnoses that outcome; none of it re-decides it.

## 1. Which conditions fail

Extracted from the archived result (`rd8_condition_matrix.md`), cell = max absolute fixed-branch
error over {state, actions} x {mean, std}:

| Order | batch 1 | batch 8 | batch 32 | batch 128 |
|---|---|---|---|---|
| natural | 2.099e-04 FAIL | 4.164e-06 PASS | 5.269e-06 PASS | 3.093e-06 PASS |
| episode_cluster_seed_0 | 1.995e-04 FAIL | 4.910e-05 FAIL | 1.267e-05 FAIL | 3.533e-06 PASS |
| episode_cluster_seed_1 | 4.692e-05 FAIL | 2.572e-05 FAIL | 1.213e-05 FAIL | 1.802e-06 PASS |
| episode_cluster_seed_2 | 6.884e-05 FAIL | 1.451e-05 FAIL | 6.689e-06 PASS | 5.266e-06 PASS |
| episode_cluster_seed_3 | 1.508e-04 FAIL | 2.221e-05 FAIL | 1.461e-05 FAIL | 6.913e-06 PASS |
| episode_cluster_seed_4 | 1.687e-04 FAIL | 2.120e-05 FAIL | 1.947e-05 FAIL | 4.628e-06 PASS |
| episode_cluster_seed_5 | 3.235e-04 FAIL | 2.230e-05 FAIL | 9.076e-06 PASS | 2.982e-06 PASS |
| episode_cluster_seed_6 | 1.565e-04 FAIL | 9.620e-06 PASS | 3.061e-06 PASS | 4.077e-06 PASS |
| episode_cluster_seed_7 | 1.753e-04 FAIL | 3.061e-06 PASS | 1.342e-05 FAIL | 5.731e-06 PASS |
| episode_cluster_seed_8 | 1.020e-04 FAIL | 2.157e-05 FAIL | 3.253e-06 PASS | 7.929e-06 PASS |
| episode_cluster_seed_9 | 1.798e-04 FAIL | 1.021e-05 FAIL | 6.275e-06 PASS | 5.731e-06 PASS |

- Failures by batch size: **11/11 at batch 1, 8/11 at batch 8, 5/11 at batch 32, 0/11 at batch 128.**
- All 24 failing conditions involve a standard-deviation vector above tolerance; **none** fails on the
  mean alone.
- Global maximum error **3.2345e-04**, on `state.std`, at order `episode_cluster_seed_5`, batch size 1.
- Projection dtype: `state` and `actions` are **float32**; `episode_index` and `frame_index` are
  int64, `timestamp` float32. 123/123 episodes re-verified against the frozen inventory, 0 mismatches.

## 2. The batching defect is not the cause

Batch size 1 is the negative control for the *batching* defect: with one row per batch the old
`batch[key][0]` path and the repaired `update(batch[key])` path are exactly equal, which the frozen
result records for every order. Batch size 1 is nevertheless the condition where **every** order
fails, and it holds the global maximum error. The residual grows as the batch gets smaller, i.e. as
the number of incremental reducer updates grows (36,900 updates at batch 1 versus 289 at batch 128).

Whatever produces the residual is therefore a property of the reducer's incremental arithmetic, not
of the coverage defect that RD8 replays.

## 3. RD8-PD: which intervention changes which error

Separately named diagnostic on identical inputs, all 11 frozen orders x 4 batch sizes = 44 conditions
per arm. Only D1 and D2 execute upstream code unmodified; D3-D5 are our controlled variants.

Control: **D1 reproduces the archived per-condition errors exactly in 44/44 conditions**, and the
independently recomputed float64 oracle differs from the frozen oracle by **exactly 0.0** for both
feature groups. The instrument reproduces the frozen replay before any variant is read.

| Arm | Change | Within 1e-5 | Max mean error | Max std error |
|---|---|---:|---:|---:|
| D1 | pinned repaired reducer, float32 projection | **20/44** | 2.804e-05 | 3.235e-04 |
| D2 | pinned reducer, same values cast to float64 | 44/44 | 1.22e-14 | 1.49e-13 |
| D3 | float64 accumulators, `E[x^2]-E[x]^2` retained | 44/44 | 1.22e-14 | 1.51e-08 |
| D4 | float32 accumulators, Chan/Welford M2 variance | **35/44** | 2.804e-05 | 4.689e-06 |
| D5 | both | 44/44 | 1.22e-14 | 1.82e-14 |

**Exactly what D3 changes.** The per-batch reductions and the running accumulators are evaluated in
float64 (`np.mean(..., dtype=np.float64)`, accumulators cast to float64). The stored input stays
float32, and — this matters for section 3's last bullet — the elementwise square `batch**2` is still
evaluated at the input dtype, float32, *before* the float64 reduction. D3 is therefore "float64
accumulation", not "every intermediate in float64". D2, which casts the values first, is the arm in
which the squaring is also float64.

Reading the arms against each other gives a clean separation:

- **The standard-deviation residual comes from the differencing variance.** D4 changes only the
  variance algorithm, at unchanged float32 precision, and the maximum std error falls from 3.235e-04
  to 4.689e-06 — a factor of 69 — bringing the std vector within tolerance in all 44 conditions. D4's
  mean error is bit-identical to D1's, so nothing else moved.
- **The mean residual comes from float32 accumulation.** It survives D4 untouched and is removed by
  D3, which changes only the accumulator dtype: the maximum mean error falls from 2.804e-05 to
  1.22e-14. The 9 conditions still failing under D4 are exactly the mean-driven ones.
- **Neither residual is attributable to the batching fix**, per section 2.
- D3 leaves a constant std floor of 1.514e-08, identical at every batch size. This floor is
  **rounding of the float32 elementwise square**, not float32 storage of the values. Storage cannot
  explain it: the oracle reads the *same* stored float32 values cast to float64, so any storage
  quantisation is common to both sides and cancels — which is why D2 and D5, working from those same
  stored values, reach 1e-13 to 1e-14. Confirmed directly
  (`../../logs/20260914-claude-p0/rd8_d3_floor_check.log`), with no batching at all, on the same
  36,900 x 14 array:

  | Arithmetic | max std error vs the float64 oracle |
  |---|---:|
  | square in float32, then reduce in float64 (D3's) — `actions` | **1.5144585e-08** |
  | square in float32, then reduce in float64 (D3's) — `state` | 1.8887e-09 |
  | square in float64 — `actions` | 2.81e-13 |
  | square in float64 — `state` | 1.58e-13 |

  The 1.5144585e-08 figure reproduces D3's reported 1.514e-08 floor to five significant figures, and
  the mean error in that check is exactly 0.0. The floor is constant across batch sizes precisely
  because per-element squaring rounding does not depend on accumulation order. It is four orders of
  magnitude below the tolerance either way.

So the review's hypothesis — single-precision accumulation of a sum-of-squares variance — is
**supported, and is more specific than stated**: the sum-of-squares differencing accounts for the
standard-deviation residual, while the single-precision accumulation separately accounts for the mean
residual. Before this diagnostic it was a hypothesis; the D3/D4 contrast is what makes it a
measurement.

## 4. What this does not establish

- RD8 stays negative. Coverage is fully restored by the upstream fix and the numerical criterion is
  still not met; the honest label is *coverage reproduced, numerical criterion not met*, never
  "successful".
- The diagnostic is run under NumPy 1.26.4 / CPython 3.12.3 on macOS arm64. The archived RD8 run used
  NumPy 2.2.6. The new run retains the archived 24/44 count and maximum error, which is evidence of
  **repeatability across that environment difference**. It is *not* evidence that environment cannot
  contribute to numerical outcomes in general, and no such claim is made.
- D3-D5 are our instruments. They show what *would* change if the reducer's arithmetic changed. They
  are not a proposed upstream patch, not an upstream behaviour, and not a defect claim against
  OpenPI: the reducer's precision is a design choice whose downstream significance this experiment
  does not measure.
- Whether a 1e-5 absolute deviation in normalisation statistics matters to any trained policy is not
  measured here. RD8's tolerance is a retained historical setting.
