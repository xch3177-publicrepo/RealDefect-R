#!/usr/bin/env python3
"""Bind frozen RD8 and RD8-PD results to a 44-row table and compact TikZ plot.

No reducer is run and no scientific result is overwritten. ``generate(root)`` writes
paper/rd8_conditions.csv, fig_rd8_residuals.tikz, and RD8_CHART_BINDING.json and returns
its binding dict for scripts/make_figures.py to include. Only the Python stdlib is used.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "original": ("reproducibility/results/real-defect-openpi-570.json", "f6193ca2c4a7111fbe25a59c908927bd63a838fb3cab982037b88fed0f03b6b7"),
    "matrix": ("experiments/rd8_diagnostic_20260914/rd8_condition_matrix.json", "8ae1688c02db87bd24a78fef409a89b51ae70dd60444872a1792e137b8b1919a"),
    "diagnostic": ("experiments/rd8_diagnostic_20260914/rd8_precision_diagnostic_all_orders.json", "c57c0180e39815f5bdc0475640993dc9aceb000d25d67290c4cd4d53733165d4"),
}
ARMS = {
    "D1": "D1_native_pinned_float32",
    "D2": "D2_pinned_float64_input",
    "D3": "D3_float64_accumulator",
    "D4": "D4_stable_variance_float32",
    "D5": "D5_float64_accum_stable_variance",
}
METRICS = ("mean", "std")
TOL = 1e-5


def _node(x, y, text, options=""):
    return rf"\node[{options}] at ({x:.5f},{y:.5f}) {{{text}}};" + "\n"


def _marker(x, y, arm, size=0.043):
    # Shapes preserve arm identity without color. The restrained Okabe--Ito
    # palette adds a redundant arm cue without changing positions or sizes.
    # Explicit coordinates avoid a pgfplots or plotmarks dependency.
    if arm == "D1":
        return rf"\draw[draw=rdPinned,fill=white,line width=0.45pt] ({x:.5f},{y:.5f}) circle[radius={size:.5f}cm];" + "\n"
    if arm == "D4":
        return (rf"\path[draw=rdStable,fill=rdStable,line width=0.45pt] ({x:.5f},{y+size:.5f}) -- "
                rf"({x-size:.5f},{y-size*.75:.5f}) -- ({x+size:.5f},{y-size*.75:.5f}) -- cycle;" + "\n")
    return (rf"\path[draw=rdAccum,fill=rdAccum,line width=0.45pt] ({x:.5f},{y+size:.5f}) -- "
            rf"({x+size:.5f},{y:.5f}) -- ({x:.5f},{y-size:.5f}) -- ({x-size:.5f},{y:.5f}) -- cycle;" + "\n")


def draw_tikz(rows):
    """Two metric panels at 8.52 cm width. Native 9 pt type; no downscaling."""
    out = [r"\begin{figure}[t]", r"\centering",
           r"\begin{tikzpicture}[x=1cm,y=1cm,font=\rmfamily\fontsize{9}{10.5}\selectfont,inner sep=0pt,text=black]",
           r"\definecolor{rdPinned}{HTML}{0072B2}",
           r"\definecolor{rdStable}{HTML}{D55E00}",
           r"\definecolor{rdAccum}{HTML}{009E73}",
           r"\path[use as bounding box] (0,-1.32) rectangle (8.52,3.65);"]
    body = "\n".join(out) + "\n"
    # Both panels use log batch positions. Metric-specific y ranges are explicitly ticked.
    for metric, left, bottom_exp, top_exp, ticks in [
        ("mean", 1.30, -16, -3, [-15, -10, -5]),
        ("std", 5.38, -9, -3, [-8, -6, -4]),
    ]:
        right, height = left + 2.95, 3.45
        def xx(batch):
            return left + 0.23 + (2.95 - 0.46) * math.log2(batch) / 7
        def yy(error):
            if not math.isfinite(error) or error <= 0:
                raise ValueError("Log axes require measured positive values; no floor substitution")
            return height * (math.log10(error) - bottom_exp) / (top_exp - bottom_exp)
        body += rf"\draw[line width=0.5pt] ({left:.5f},0) -- ({right:.5f},0);" + "\n"
        body += rf"\draw[line width=0.5pt] ({left:.5f},0) -- ({left:.5f},{height:.5f});" + "\n"
        for exponent in ticks:
            y = yy(10.0 ** exponent)
            body += rf"\draw[line width=0.5pt] ({left-.055:.5f},{y:.5f}) -- ({left:.5f},{y:.5f});" + "\n"
            body += _node(left-.095, y, rf"$10^{{{exponent}}}$", "anchor=east")
        for batch in (1, 8, 32, 128):
            x = xx(batch)
            body += rf"\draw[line width=0.5pt] ({x:.5f},0) -- ({x:.5f},-0.055);" + "\n"
            body += _node(x, -.14, str(batch), "anchor=north")
        y_tol = yy(TOL)
        body += rf"\draw[black,line width=0.5pt,dash pattern=on 2.4pt off 1.8pt] ({left:.5f},{y_tol:.5f}) -- ({right:.5f},{y_tol:.5f});" + "\n"
        # Small deterministic horizontal offsets separate the 11 orders and 3 arms.
        # They change display positions only; CSV retains the actual batch size.
        for arm, offset in [("D1", -.055), ("D4", .055), ("D3", 0)]:
            for row in rows:
                x = xx(row["batch"]) + (row["order_index"] - 5) * .023 + offset
                body += _marker(x, yy(row[f"{metric}_err_{arm}"]), arm)
        label = "(a) Mean" if metric == "mean" else "(b) Standard deviation"
        body += _node((left+right)/2, -.42, label, "anchor=north")
    body += _node(.10, 1.7, "Maximum absolute error", "rotate=90")
    body += _node(4.59, -.79, "Batch size", "anchor=north")
    # One unobtrusive shared line, with no title banner or colored text.
    for arm, x, label in [("D1", .31, "Pinned repaired"), ("D4", 3.04, "Stable variance"), ("D3", 5.79, "Float64 accum.")]:
        body += _marker(x, -1.23, arm, size=.055)
        body += _node(x+.15, -1.23, label, "anchor=west")
    body += r"\end{tikzpicture}" + "\n"
    body += (r"\caption{RD8 residuals over 44 conditions (four batch sizes, 11 orders); each point is the maximum error over the state and action components. "
             r"D1 is the pinned repaired reducer; D4 uses stable variance with float32 accumulation; D3 uses float64 accumulation with float32 input. "
             r"The ordinate ranges differ; small horizontal offsets separate orders and arms. Dashed lines mark the retained $10^{-5}$ tolerance. "
             r"D1 fails 24/44 conditions; D4 retains nine mean failures; D3 passes all 44 with nonzero std residuals.}" + "\n")
    body += r"\label{fig:rd8}" + "\n" + r"\end{figure}" + "\n"
    return body



def _input_identity(root, relative, expected, observed):
    """Accept exact archival bytes or one explicitly bound sanitized public derivative.

    Never replace the archival digest. SOURCE_MAP's relevant entry, not its whole-file
    digest, is retained because public finalization also updates unrelated output entries.
    """
    identity = {"path": relative, "sha256": observed, "archival_sha256": expected,
                "identity_mode": "exact_archival_bytes"}
    if observed == expected:
        return identity
    map_path = root / "SOURCE_MAP.json"
    if not map_path.is_file():
        raise ValueError(f"changed frozen input has no public source map: {relative}")
    mapping = json.loads(map_path.read_bytes())
    if not isinstance(mapping, list) or not all(isinstance(item, dict) for item in mapping):
        raise ValueError("SOURCE_MAP.json must contain an array of source records")
    candidates = [item for item in mapping if item.get("public_path") == relative]
    if len(candidates) != 1:
        raise ValueError(f"public source identity missing or ambiguous: {relative}")
    entry = candidates[0]
    if entry.get("source_sha256") != expected or entry.get("public_sha256") != observed:
        raise ValueError(f"public source digest relationship does not match: {relative}")
    identity.update({
        "identity_mode": "source_mapped_public_derivative",
        "public_derivative_sha256": observed,
        "public_source_mapping": {
            "path": "SOURCE_MAP.json",
            "public_path": relative,
            "source_sha256": entry["source_sha256"],
            "public_sha256": entry["public_sha256"],
        },
    })
    return identity


def generate(root=ROOT):
    root = Path(root)
    docs, sources = {}, {}
    for key, (relative, expected) in SOURCES.items():
        raw = (root / relative).read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        sources[key] = _input_identity(root, relative, expected, observed)
        docs[key] = json.loads(raw)
    original, matrix, diag = (docs[x] for x in ("original", "matrix", "diagnostic"))
    assert matrix["frozen_status"] == "NOT_REPRODUCED"
    assert matrix["source_result_sha256"] == sources["original"]["archival_sha256"]
    assert matrix["frozen_tolerance"] == diag["tolerance"] == TOL
    assert matrix["condition_count"] == 44
    assert matrix["orders"] == diag["input"]["orders_executed"]
    assert matrix["batch_sizes"] == diag["batch_sizes"] == [1, 8, 32, 128]
    assert diag["scope"] == "all_orders"
    assert diag["d1_vs_frozen_rd8"]["exact_matches"] == 44
    assert all(c["exact_match_with_frozen_result"] for c in diag["d1_vs_frozen_rd8"]["per_condition"])
    for check in diag["oracle"]["crosscheck_against_frozen_rd8_oracle"].values():
        assert check["max_abs_mean_difference"] == check["max_abs_std_difference"] == 0
    orig_index = {(c["order_name"], c["batch_size"]): c for c in original["conditions"]}
    matrix_index = {(c["order_name"], c["batch_size"]): (i, c) for i, c in enumerate(matrix["cells"])}
    diag_index = {(c["order_name"], c["batch_size"], c["arm"]): (i, c) for i, c in enumerate(diag["arms"])}
    assert len(orig_index) == len(matrix_index) == 44 and len(diag_index) == 220
    rows, mappings = [], []
    for order_index, order in enumerate(matrix["orders"]):
        for batch in matrix["batch_sizes"]:
            matrix_i, cell = matrix_index[(order, batch)]
            row = {"batch": batch, "order_index": order_index, "order": order}
            mapping = {"order": order, "batch": batch, "matrix_cell": matrix_i, "diagnostic_arms": {}}
            for short, arm in ARMS.items():
                arm_i, record = diag_index[(order, batch, arm)]
                mapping["diagnostic_arms"][short] = arm_i
                for metric in METRICS:
                    field = f"max_absolute_{metric}_error"
                    value = record[field]
                    assert value == max(record["per_feature"][f][field] for f in ("state", "actions"))
                    assert math.isfinite(value) and value > 0
                    if short == "D1":
                        assert value == cell["branches"]["fixed"][field]
                        for feature in ("state", "actions"):
                            assert record["per_feature"][feature][field] == orig_index[(order, batch)]["branches"]["fixed"][feature]["oracle_error"][field]
                    row[f"{metric}_err_{short}"] = value
                row[f"passes_{short}"] = row[f"mean_err_{short}"] <= TOL and row[f"std_err_{short}"] <= TOL
                assert row[f"passes_{short}"] == record["within_1e_5"]
            row["tolerance"] = TOL
            rows.append(row)
            mappings.append(mapping)
    rollup = {}
    for short, arm in ARMS.items():
        rollup[short] = {"conditions": len(rows), "failed": sum(not r[f"passes_{short}"] for r in rows)}
        assert rollup[short]["failed"] == diag["arm_rollup"][arm]["above_1e_5"]
        for metric in METRICS:
            values = [r[f"{metric}_err_{short}"] for r in rows]
            rollup[short][metric] = {"min": min(values), "max": max(values), "above_tolerance": sum(v > TOL for v in values)}
    assert rollup["D1"]["failed"] == matrix["fixed_fail_count"] == 24
    assert rollup["D4"]["failed"] == 9 and rollup["D3"]["failed"] == 0
    by_batch = {str(b): sum(not r["passes_D1"] for r in rows if r["batch"] == b) for b in matrix["batch_sizes"]}
    assert list(by_batch.values()) == [11, 8, 5, 0]
    paper = root / "paper"
    paper.mkdir(exist_ok=True)
    csv_path = paper / "rd8_conditions.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tikz_path = paper / "fig_rd8_residuals.tikz"
    tikz_path.write_text(draw_tikz(rows))
    binding = {
        "generator": "scripts/rd8_chart_data.py", "new_experiment": False,
        "frozen_rd8_status": matrix["frozen_status"], "sources": sources,
        "data_definition": "Per condition and arm, maximum absolute mean or population-std error over 14 state and 14 action components against the same float64 two-pass full-array reference.",
        "csv_column_mapping": {"batch": "diagnostic.arms[i].batch_size", "order": "diagnostic.arms[i].order_name", "order_index": "zero-based position in matrix.orders", "{metric}_err_{D1..D5}": "diagnostic.arms[i].max_absolute_{metric}_error", "passes_{D1..D5}": "both exported mean and std errors <= tolerance; cross-checked against diagnostic.arms[i].within_1e_5"},
        "condition_source_indexes": mappings,
        "arm_names": ARMS, "rollup": rollup, "D1_failed_by_batch": by_batch,
        "figure": {"width_cm": 8.52, "height_cm_before_caption": 4.97, "font_pt": 9, "selected_arms": ["D1", "D4", "D3"], "panel_metrics": ["mean", "std"], "plotted_points": 264, "batch_scale": "log2 with display-only arm/order offsets", "error_scale": "log10; mean [1e-16,1e-3], std [1e-9,1e-3]", "display_floors_or_zero_substitutions": False, "tolerance": TOL},
        "scope": ["RD8 remains NOT_REPRODUCED under the frozen 24/44 criterion.", "D3 and D4 are controlled diagnostic variants, not upstream fixes.", "D3 retains float32 elementwise squaring before float64 reductions and accumulators.", "D4 std improvement does not remove its nine mean failures.", "Points show selected finite conditions, not a monotonic law or a general accuracy guarantee.", "D2 and D5 remain in the CSV and full artifact; omission from the compact figure is explicit."],
        "outputs": {p.name: {"path": str(p.relative_to(root)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (csv_path, tikz_path)},
    }
    (paper / "RD8_CHART_BINDING.json").write_text(json.dumps(binding, indent=2) + "\n")
    return binding


if __name__ == "__main__":
    binding = generate()
    print(json.dumps({"outputs": binding["outputs"], "rollup": binding["rollup"], "figure": binding["figure"]}, indent=2))
