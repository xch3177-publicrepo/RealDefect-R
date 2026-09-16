# RD8 decomposition and precision diagnostic (2026-09-14/15)

Answers review item P0-2. Two separate pieces of work, both of which leave RD8's frozen record
untouched: RD8 remains **NOT_REPRODUCED** under its original complete criterion (fixed-branch mean
and std within absolute 1e-5 of the independent float64 full-array oracle, in every one of the 44
conditions). Nothing here re-decides that.

## 1. Decomposition of the existing 44-condition result

`extract_rd8_condition_matrix.py` reads the archived `reproducibility/results/real-defect-openpi-570.json`
read-only and re-presents outcomes it already contains. It does not re-run the reducer.

Outputs: `rd8_condition_matrix.json`, `rd8_condition_matrix.md`.
Raw stdout: `../../logs/20260914-claude-p0/rd8_condition_matrix.log`.

## 2. Recovery and re-verification of the frozen projections

`recover_projections.py` extracts only `projections/` and `official-metadata/` from the hash-pinned
historical evidence archive into the git-ignored `experiments/work/rd8_diagnostic_20260914/`. No
historical analysis checkpoint is reused and no original artifact is written to. Every array is
re-hashed and compared with the frozen `projection.logical_inventory`.

Output: `rd8_projection_recovery.json` — 123/123 episodes verified, 0 mismatches, `state` and
`actions` confirmed **float32**.

## 3. RD8-PD: controlled precision/algorithm diagnostic

`rd8_precision_diagnostic.py` is a **separately named diagnostic**, not a re-run of RD8. Five arms on
identical inputs and identical batch orders:

| Arm | Change |
|---|---|
| D1 | pinned upstream repaired `RunningStats`, unmodified, on the float32 projection |
| D2 | same pinned reducer, unmodified, on the same values cast to float64 |
| D3 | float32 stored input; accumulation in float64; `E[x^2]-E[x]^2` retained |
| D4 | float32 accumulation retained; `E[x^2]-E[x]^2` replaced by a Chan/Welford M2 update |
| D5 | both changes |

Only D1 and D2 execute upstream code verbatim. D3-D5 are our own controlled variants and are
labelled as such in the output; they are diagnostic instruments, not claims about upstream.

D1 is the control: it must reproduce the archived per-condition errors before any variant is read.
`d1_vs_frozen_rd8` in the output records that comparison condition by condition.

### Development history, recorded rather than tidied away

The diagnostic was first run on the **natural order only**, writing `rd8_precision_diagnostic.json`.
That pilot file was deleted when the script was extended to sweep all 11 frozen orders, which
changed the output filename to `rd8_precision_diagnostic_<scope>.json`. The pilot's raw stdout
survives in full at `../../logs/20260914-claude-p0/rd8_precision_diagnostic.log` and contains every
number it produced. The extension added the order sweep, the per-arm rollup and the D1-versus-frozen
control; it did not change any arm definition, the oracle, the tolerance, or the input.

Usage: `python rd8_precision_diagnostic.py [natural|all_orders]` in `.venv-rd8`
(NumPy 1.26.4, pydantic 2.13.5, numpydantic 1.8.1, CPython 3.12.3).
Raw stdout of the accepted all-orders run: `../../logs/20260914-claude-p0/rd8_precision_diagnostic_all_orders.log`.
