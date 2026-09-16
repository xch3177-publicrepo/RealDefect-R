"""RD8-PD: a separately named precision/algorithm diagnostic for the RD8 residual.

This is NOT a re-run of RD8 and does NOT change RD8's frozen criterion, denominator, or negative
outcome. RD8 remains "coverage complete, numerical criterion not met" under its original rule.
RD8-PD asks a narrower, separate question:

    Given the *same* frozen projected input and the *same* pinned repaired reducer, which single
    controlled change to the reducer's arithmetic removes the residual error against the float64
    full-array reference?

Five arms, each run on identical inputs, identical batch order, identical batch sizes:

  D1 native            pinned upstream RunningStats, unmodified, on the float32 projection
                       (this is the arm that must reproduce the frozen RD8 fixed-branch numbers)
  D2 float64 input     pinned reducer, unmodified, on the same values cast to float64 before update
  D3 float64 accum     float32 input preserved; per-batch reductions and running accumulators in
                       float64. Note precisely: the elementwise square `batch**2` is still evaluated
                       at the input dtype, float32, before the float64 reduction, so D3 is "float64
                       accumulation", not "every intermediate in float64". D2, which casts the values
                       first, is the arm in which the squaring is also float64.
  D4 stable variance   float32 accumulation preserved; E[x^2]-E[x]^2 replaced by a streaming
                       Chan/Welford co-moment update (mathematically equivalent, differently
                       conditioned)
  D5 float64 + stable  both changes together

Only D1 uses upstream code verbatim. D3-D5 are our own controlled variants and are labelled as such
everywhere; they are diagnostic instruments, not claims about upstream behaviour.

Reference oracle: float64 two-pass full-array mean and population standard deviation, the same
definition the frozen RD8 run used. The script cross-checks its oracle against the oracle vectors
recorded in the archived RD8 result before reporting anything else.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PINNED_DIR = REPO / "reproducibility" / "light_inputs" / "upstream_pinned_sources"
PROJ = REPO / "experiments" / "work" / "rd8_diagnostic_20260914" / "projections"
RESULT = REPO / "reproducibility" / "results" / "real-defect-openpi-570.json"
TOL = 1e-5
BATCH_SIZES = (1, 8, 32, 128)
FEATURES = ("state", "actions")

sys.path.insert(0, str(PINNED_DIR))
import RD8_fixed_stats as pinned  # noqa: E402  pinned upstream repaired reducer, imported unchanged


# --------------------------------------------------------------------------------------
# Controlled variants. Each mirrors the pinned update rule exactly except for the one
# documented change; histogram/quantile machinery is dropped because quantiles are
# descriptive in RD8 and play no part in its mean/std criterion.
# --------------------------------------------------------------------------------------
class VariantStats:
    """Reimplementation of the pinned mean/std path with two switchable knobs.

    accum_dtype:     dtype of the running accumulators (np.float32 mirrors the pinned behaviour
                     when fed float32 batches; np.float64 is the controlled change).
    stable_variance: False reproduces the pinned `E[x^2] - E[x]^2` differencing;
                     True uses a streaming Chan/Welford M2 update instead.
    """

    def __init__(self, accum_dtype, stable_variance: bool):
        self.accum_dtype = accum_dtype
        self.stable_variance = stable_variance
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._m2 = None

    def update(self, batch: np.ndarray) -> None:
        batch = batch.reshape(-1, batch.shape[-1])
        n = batch.shape[0]
        dt = self.accum_dtype
        batch_mean = np.mean(batch, axis=0, dtype=dt).astype(dt)
        if self._count == 0:
            self._mean = batch_mean.copy()
            if self.stable_variance:
                self._m2 = np.sum(
                    (batch.astype(dt) - batch_mean) ** 2, axis=0, dtype=dt
                ).astype(dt)
            else:
                self._mean_of_squares = np.mean(batch**2, axis=0, dtype=dt).astype(dt)
            self._count = n
            return

        total = self._count + n
        if self.stable_variance:
            batch_m2 = np.sum(
                (batch.astype(dt) - batch_mean) ** 2, axis=0, dtype=dt
            ).astype(dt)
            delta = batch_mean - self._mean
            self._m2 = (
                self._m2 + batch_m2 + (delta**2) * (self._count * n / total)
            ).astype(self.accum_dtype)
            self._mean = (self._mean + delta * (n / total)).astype(self.accum_dtype)
        else:
            batch_mos = np.mean(batch**2, axis=0, dtype=dt).astype(dt)
            self._count = total
            self._mean = (self._mean + (batch_mean - self._mean) * (n / self._count)).astype(
                self.accum_dtype
            )
            self._mean_of_squares = (
                self._mean_of_squares + (batch_mos - self._mean_of_squares) * (n / self._count)
            ).astype(self.accum_dtype)
            return
        self._count = total

    def get_mean_std(self):
        if self.stable_variance:
            var = self._m2 / self._count
        else:
            var = self._mean_of_squares - self._mean**2
        return np.asarray(self._mean), np.sqrt(np.maximum(0, np.asarray(var)))


def run_pinned(rows: np.ndarray, batch_size: int):
    """Run the unmodified upstream repaired reducer, including its final partial batch."""
    rs = pinned.RunningStats()
    for start in range(0, rows.shape[0], batch_size):
        rs.update(rows[start : start + batch_size])
    st = rs.get_statistics()
    return np.asarray(st.mean), np.asarray(st.std)


def run_variant(rows: np.ndarray, batch_size: int, accum_dtype, stable: bool):
    vs = VariantStats(accum_dtype, stable)
    for start in range(0, rows.shape[0], batch_size):
        vs.update(rows[start : start + batch_size])
    return vs.get_mean_std()


def load_episode_arrays():
    """Load every episode projection, keyed by episode index, preserving in-episode frame order."""
    files = sorted(PROJ.glob("*.npz"), key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
    assert len(files) == 123, f"expected 123 projections, found {len(files)}"
    per_episode, src_dtypes = {}, set()
    for path in files:
        ep = int("".join(c for c in path.stem if c.isdigit()))
        with np.load(path) as z:
            per_episode[ep] = {}
            for f in FEATURES:
                arr = z[f]
                src_dtypes.add(str(arr.dtype))
                per_episode[ep][f] = np.array(arr)
    return per_episode, sorted(src_dtypes), [p.name for p in files]


def concat_order(per_episode, order):
    """Concatenate episode blocks in the given episode order (the frozen RD8 order definition)."""
    out = {f: np.concatenate([per_episode[ep][f] for ep in order], axis=0) for f in FEATURES}
    for f in FEATURES:
        assert out[f].shape == (36900, 14), f"{f} shape {out[f].shape}"
    return out


def main() -> int:
    scope = sys.argv[1] if len(sys.argv) > 1 else "natural"
    assert scope in {"natural", "all_orders"}, "scope must be 'natural' or 'all_orders'"

    frozen_doc = json.loads(RESULT.read_bytes())
    complete_orders = frozen_doc["frozen_inputs"]["complete_orders"]
    order_names = list(frozen_doc["frozen_inputs"]["design"]["orders"])
    if scope == "natural":
        order_names = ["natural"]

    per_episode, src_dtypes, file_names = load_episode_arrays()
    assert src_dtypes == ["float32"], f"unexpected projection dtypes {src_dtypes}"

    # float64 two-pass full-array reference (order-independent up to rounding), cross-checked
    # against the frozen RD8 oracle before anything else is reported.
    natural_rows = concat_order(per_episode, complete_orders["natural"])
    oracle = {}
    for f in FEATURES:
        x = natural_rows[f].astype(np.float64)
        oracle[f] = {"mean": x.mean(axis=0), "std": x.std(axis=0)}

    frozen_oracle = frozen_doc["oracle"]
    oracle_crosscheck = {}
    for f in FEATURES:
        fm = np.asarray(frozen_oracle[f]["mean"], dtype=np.float64)
        fs = np.asarray(frozen_oracle[f]["std"], dtype=np.float64)
        oracle_crosscheck[f] = {
            "max_abs_mean_difference": float(np.max(np.abs(oracle[f]["mean"] - fm))),
            "max_abs_std_difference": float(np.max(np.abs(oracle[f]["std"] - fs))),
            "frozen_count": frozen_oracle[f]["count"],
        }

    arms = [
        ("D1_native_pinned_float32", "pinned", None, None,
         "upstream repaired RunningStats, unmodified, float32 projection"),
        ("D2_pinned_float64_input", "pinned_f64in", None, None,
         "upstream repaired RunningStats, unmodified, identical values cast to float64"),
        ("D3_float64_accumulator", "variant", np.float64, False,
         "float32 input preserved; running accumulators in float64; E[x^2]-E[x]^2 retained"),
        ("D4_stable_variance_float32", "variant", np.float32, True,
         "float32 accumulators preserved; Chan/Welford M2 replaces E[x^2]-E[x]^2"),
        ("D5_float64_accum_stable_variance", "variant", np.float64, True,
         "float64 accumulators and Chan/Welford M2"),
    ]

    results = []
    for order_name in order_names:
        rows = concat_order(per_episode, complete_orders[order_name])
        for arm_name, kind, accum, stable, description in arms:
            for bs in BATCH_SIZES:
                per_feature = {}
                for f in FEATURES:
                    if kind == "pinned":
                        m, s = run_pinned(rows[f], bs)
                    elif kind == "pinned_f64in":
                        m, s = run_pinned(rows[f].astype(np.float64), bs)
                    else:
                        m, s = run_variant(rows[f], bs, accum, stable)
                    me = float(np.max(np.abs(np.asarray(m, dtype=np.float64) - oracle[f]["mean"])))
                    se = float(np.max(np.abs(np.asarray(s, dtype=np.float64) - oracle[f]["std"])))
                    per_feature[f] = {
                        "max_absolute_mean_error": me,
                        "max_absolute_std_error": se,
                        "within_1e_5": me <= TOL and se <= TOL,
                    }
                worst_mean = max(per_feature[f]["max_absolute_mean_error"] for f in FEATURES)
                worst_std = max(per_feature[f]["max_absolute_std_error"] for f in FEATURES)
                results.append({
                    "arm": arm_name,
                    "arm_description": description,
                    "upstream_code_unmodified": kind.startswith("pinned"),
                    "order_name": order_name,
                    "batch_size": bs,
                    "per_feature": per_feature,
                    "max_absolute_mean_error": worst_mean,
                    "max_absolute_std_error": worst_std,
                    "within_1e_5": worst_mean <= TOL and worst_std <= TOL,
                })
                print(f"{order_name:24s} {arm_name:34s} bs={bs:4d} mean={worst_mean:.3e} "
                      f"std={worst_std:.3e} "
                      f"{'PASS' if worst_mean <= TOL and worst_std <= TOL else 'FAIL'}", flush=True)

    # Per-arm rollup over every executed condition, plus an exact comparison of the unmodified
    # arm D1 against the frozen RD8 fixed-branch per-condition errors.
    by_arm = {}
    for arm_name, *_ in arms:
        sub = [r for r in results if r["arm"] == arm_name]
        by_arm[arm_name] = {
            "conditions": len(sub),
            "within_1e_5": sum(1 for r in sub if r["within_1e_5"]),
            "above_1e_5": sum(1 for r in sub if not r["within_1e_5"]),
            "max_absolute_mean_error": max(r["max_absolute_mean_error"] for r in sub),
            "max_absolute_std_error": max(r["max_absolute_std_error"] for r in sub),
        }

    frozen_cells = {}
    for cond in frozen_doc["conditions"]:
        fb = cond["branches"]["fixed"]
        frozen_cells[(cond["order_name"], cond["batch_size"])] = {
            f: {
                "max_absolute_mean_error": fb[f]["oracle_error"]["max_absolute_mean_error"],
                "max_absolute_std_error": fb[f]["oracle_error"]["max_absolute_std_error"],
            }
            for f in FEATURES
        }
    d1_checks = []
    for r in (r for r in results if r["arm"] == "D1_native_pinned_float32"):
        want = frozen_cells[(r["order_name"], r["batch_size"])]
        d1_checks.append({
            "order_name": r["order_name"],
            "batch_size": r["batch_size"],
            "exact_match_with_frozen_result": all(
                want[f]["max_absolute_mean_error"] == r["per_feature"][f]["max_absolute_mean_error"]
                and want[f]["max_absolute_std_error"] == r["per_feature"][f]["max_absolute_std_error"]
                for f in FEATURES
            ),
        })
    d1_exact = sum(1 for c in d1_checks if c["exact_match_with_frozen_result"])

    report = {
        "diagnostic_id": "RD8-PD",
        "relationship_to_rd8": (
            "separate diagnostic; RD8's frozen criterion, denominator and NOT_REPRODUCED outcome "
            "are unchanged and not re-decided by this run"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pydantic": md.version("pydantic"),
            "numpydantic": md.version("numpydantic"),
        },
        "pinned_reducer": {
            "path": str((PINNED_DIR / "RD8_fixed_stats.py").relative_to(REPO)),
            "sha256": hashlib.sha256((PINNED_DIR / "RD8_fixed_stats.py").read_bytes()).hexdigest(),
            "variance_expression": "self._mean_of_squares - self._mean ** 2",
            "imported_unmodified": True,
        },
        "input": {
            "projection_dtype": {f: "float32" for f in FEATURES},
            "index_dtypes": {"episode_index": "int64", "frame_index": "int64", "timestamp": "float32"},
            "rows": 36900,
            "components": 14,
            "orders_executed": order_names,
            "order_definition": "frozen RD8 complete_orders; frame order preserved inside each episode",
            "episode_files": len(file_names),
            "recovery_report": "experiments/rd8_diagnostic_20260914/rd8_projection_recovery.json",
        },
        "oracle": {
            "definition": "float64 two-pass full-array mean and population std",
            "crosscheck_against_frozen_rd8_oracle": oracle_crosscheck,
        },
        "tolerance": TOL,
        "batch_sizes": list(BATCH_SIZES),
        "scope": scope,
        "arm_rollup": by_arm,
        "d1_vs_frozen_rd8": {
            "compared_conditions": len(d1_checks),
            "exact_matches": d1_exact,
            "note": (
                "D1 re-executes the unmodified pinned reducer on the same verified projections; "
                "exact agreement with the archived per-condition errors is the control that the "
                "diagnostic instrument reproduces the frozen replay before any variant is read"
            ),
            "per_condition": d1_checks,
        },
        "arms": results,
    }
    out = HERE / f"rd8_precision_diagnostic_{scope}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("\noracle crosscheck:", json.dumps(oracle_crosscheck))
    print("arm rollup:", json.dumps(by_arm, indent=2))
    print(f"D1 exact agreement with frozen RD8: {d1_exact}/{len(d1_checks)} conditions")
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
