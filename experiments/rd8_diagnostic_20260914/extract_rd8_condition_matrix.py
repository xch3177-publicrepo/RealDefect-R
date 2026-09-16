"""Decompose the frozen RD8 result into a batch-size x order x statistic failure matrix.

Read-only with respect to every historical artifact. This script does not re-run the reducer and
does not change RD8's frozen decision rule; it only re-presents the outcomes that the archived
result JSON already records, so that the manuscript can report *which* conditions fail rather than
only how many.

Frozen criterion carried over unchanged from the original run:
    a condition passes iff, for both the `state` and `actions` feature groups, the maximum absolute
    error of the fixed-branch mean vector and of the fixed-branch population-std vector against the
    independently computed float64 full-array oracle is <= 1e-5.
Quantiles remain descriptive (approximate histogram) and are excluded from the criterion.

Usage:  python3 extract_rd8_condition_matrix.py
Output: rd8_condition_matrix.json, rd8_condition_matrix.md
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULT = REPO / "reproducibility" / "results" / "real-defect-openpi-570.json"
TOL = 1e-5
FEATURES = ("state", "actions")
STATS = ("mean", "std")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    raw = RESULT.read_bytes()
    doc = json.loads(raw)
    design = doc["frozen_inputs"]["design"]
    assert design["mean_std_absolute_tolerance"] == TOL, "frozen tolerance changed"

    orders = list(design["orders"])
    batch_sizes = list(design["batch_sizes"])

    cells = {}
    for cond in doc["conditions"]:
        key = (cond["order_name"], cond["batch_size"])
        entry = {
            "order_name": cond["order_name"],
            "batch_size": cond["batch_size"],
            "batch_one_negative_control_exact": cond["batch_one_negative_control_exact"],
            "branches": {},
        }
        for branch in ("old", "fixed"):
            b = cond["branches"][branch]
            per_feature = {}
            for feat in FEATURES:
                err = b[feat]["oracle_error"]
                per_feature[feat] = {
                    "max_absolute_mean_error": err["max_absolute_mean_error"],
                    "max_absolute_std_error": err["max_absolute_std_error"],
                    "mean_within_tol": err["max_absolute_mean_error"] <= TOL,
                    "std_within_tol": err["max_absolute_std_error"] <= TOL,
                    "mean_std_within_frozen_tolerance": err["mean_std_within_frozen_tolerance"],
                }
            entry["branches"][branch] = {
                "per_feature": per_feature,
                "max_absolute_mean_error": max(
                    per_feature[f]["max_absolute_mean_error"] for f in FEATURES
                ),
                "max_absolute_std_error": max(
                    per_feature[f]["max_absolute_std_error"] for f in FEATURES
                ),
                "passes_frozen_criterion": all(
                    per_feature[f]["mean_std_within_frozen_tolerance"] for f in FEATURES
                ),
            }
        cells[key] = entry

    assert len(cells) == len(orders) * len(batch_sizes) == 44, f"expected 44 cells, got {len(cells)}"

    # Which statistic, feature and condition drive the failures?
    failing = [c for c in cells.values() if not c["branches"]["fixed"]["passes_frozen_criterion"]]
    passing = [c for c in cells.values() if c["branches"]["fixed"]["passes_frozen_criterion"]]

    def _driver(cell):
        """Name every (feature, statistic) pair of the fixed branch that exceeds the tolerance."""
        out = []
        for feat in FEATURES:
            pf = cell["branches"]["fixed"]["per_feature"][feat]
            if not pf["mean_within_tol"]:
                out.append(f"{feat}.mean")
            if not pf["std_within_tol"]:
                out.append(f"{feat}.std")
        return out

    driver_counts: dict[str, int] = {}
    for cell in failing:
        for d in _driver(cell):
            driver_counts[d] = driver_counts.get(d, 0) + 1

    std_involved = [c for c in failing if any(d.endswith(".std") for d in _driver(c))]
    mean_involved = [c for c in failing if any(d.endswith(".mean") for d in _driver(c))]
    mean_only = [c for c in failing if all(d.endswith(".mean") for d in _driver(c))]
    driver_breakdown = {
        "conditions_failing_with_std_involved": len(std_involved),
        "conditions_failing_with_mean_involved": len(mean_involved),
        "conditions_failing_on_mean_only": len(mean_only),
        "mean_only_conditions": [(c["order_name"], c["batch_size"]) for c in mean_only],
    }

    # Global maximum fixed-branch error and where it occurs.
    max_cell = max(
        cells.values(),
        key=lambda c: max(
            c["branches"]["fixed"]["max_absolute_mean_error"],
            c["branches"]["fixed"]["max_absolute_std_error"],
        ),
    )
    max_value = max(
        max_cell["branches"]["fixed"]["max_absolute_mean_error"],
        max_cell["branches"]["fixed"]["max_absolute_std_error"],
    )
    max_stat = (
        "std"
        if max_cell["branches"]["fixed"]["max_absolute_std_error"] >= max_cell["branches"]["fixed"]["max_absolute_mean_error"]
        else "mean"
    )
    max_feature = max(
        FEATURES,
        key=lambda f: max(
            max_cell["branches"]["fixed"]["per_feature"][f]["max_absolute_mean_error"],
            max_cell["branches"]["fixed"]["per_feature"][f]["max_absolute_std_error"],
        ),
    )

    # Batch-size-one rows: the negative control for the *batching* defect.
    batch_one = [c for c in cells.values() if c["batch_size"] == 1]
    batch_one_old_equals_fixed = all(c["batch_one_negative_control_exact"] for c in batch_one)
    batch_one_fixed_pass = [c["branches"]["fixed"]["passes_frozen_criterion"] for c in batch_one]

    by_batch = {}
    for bs in batch_sizes:
        rows = [c for c in cells.values() if c["batch_size"] == bs]
        by_batch[str(bs)] = {
            "conditions": len(rows),
            "fixed_pass": sum(1 for c in rows if c["branches"]["fixed"]["passes_frozen_criterion"]),
            "fixed_fail": sum(1 for c in rows if not c["branches"]["fixed"]["passes_frozen_criterion"]),
            "max_fixed_mean_error": max(c["branches"]["fixed"]["max_absolute_mean_error"] for c in rows),
            "max_fixed_std_error": max(c["branches"]["fixed"]["max_absolute_std_error"] for c in rows),
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "source_result": str(RESULT.relative_to(REPO)),
        "source_result_sha256": sha256_file(RESULT),
        "frozen_status": doc["status"],
        "frozen_tolerance": TOL,
        "orders": orders,
        "batch_sizes": batch_sizes,
        "condition_count": len(cells),
        "fixed_fail_count": len(failing),
        "fixed_pass_count": len(passing),
        "failure_drivers": driver_counts,
        "failure_driver_breakdown": driver_breakdown,
        "max_fixed_error": {
            "value": max_value,
            "statistic": max_stat,
            "feature_group": max_feature,
            "order_name": max_cell["order_name"],
            "batch_size": max_cell["batch_size"],
        },
        "batch_size_one": {
            "condition_count": len(batch_one),
            "old_equals_fixed_exactly_for_every_order": batch_one_old_equals_fixed,
            "fixed_passes_frozen_criterion_count": sum(batch_one_fixed_pass),
            "fixed_fails_frozen_criterion_count": len(batch_one_fixed_pass) - sum(batch_one_fixed_pass),
            "max_fixed_mean_error": max(c["branches"]["fixed"]["max_absolute_mean_error"] for c in batch_one),
            "max_fixed_std_error": max(c["branches"]["fixed"]["max_absolute_std_error"] for c in batch_one),
        },
        "by_batch_size": by_batch,
        "cells": [cells[(o, b)] for o in orders for b in batch_sizes],
    }

    out_json = HERE / "rd8_condition_matrix.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")

    # Human-readable matrix: rows = order, cols = batch size, cell = max |Δ| of fixed branch.
    lines = ["# RD8 fixed-branch condition matrix (extracted, not re-run)", ""]
    lines.append(f"Source: `{summary['source_result']}` sha256 `{summary['source_result_sha256']}`")
    lines.append(f"Frozen status preserved: **{summary['frozen_status']}**; tolerance {TOL:g}.")
    lines.append("")
    lines.append("Cell = max absolute error of the fixed branch against the float64 oracle over")
    lines.append("{state, actions} x {mean, std}; `FAIL` marks a cell above the frozen tolerance.")
    lines.append("")
    lines.append("| Order | " + " | ".join(f"batch {b}" for b in batch_sizes) + " |")
    lines.append("|---|" + "---|" * len(batch_sizes))
    for o in orders:
        row = [o]
        for b in batch_sizes:
            c = cells[(o, b)]["branches"]["fixed"]
            v = max(c["max_absolute_mean_error"], c["max_absolute_std_error"])
            row.append(f"{v:.3e} {'PASS' if c['passes_frozen_criterion'] else 'FAIL'}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"Fixed-branch failures: {summary['fixed_fail_count']}/{summary['condition_count']}.")
    lines.append(f"Failure drivers (condition counts): {json.dumps(driver_counts)}")
    lines.append(
        f"Of the {summary['fixed_fail_count']} failing conditions, "
        f"{driver_breakdown['conditions_failing_with_std_involved']} involve a std vector above "
        f"tolerance and {driver_breakdown['conditions_failing_on_mean_only']} fail on the mean alone."
    )
    lines.append(
        "Maximum fixed error {value:.12g} on {feature_group}.{statistic} at order "
        "{order_name}, batch {batch_size}.".format(**summary["max_fixed_error"])
    )
    lines.append("")
    b1 = summary["batch_size_one"]
    lines.append(
        f"Batch size 1 ({b1['condition_count']} conditions): old branch equals fixed branch exactly for "
        f"every order = {b1['old_equals_fixed_exactly_for_every_order']}; fixed branch passes the frozen "
        f"criterion in {b1['fixed_passes_frozen_criterion_count']}/{b1['condition_count']} of them "
        f"(max mean error {b1['max_fixed_mean_error']:.3e}, max std error {b1['max_fixed_std_error']:.3e})."
    )
    (HERE / "rd8_condition_matrix.md").write_text("\n".join(lines) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
