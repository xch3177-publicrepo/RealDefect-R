"""GX-BASE-20260914 step 3: execute the frozen suites natively in Great Expectations.

Freeze discipline: on first run this writes `FROZEN_PLAN.json` recording the SHA-256 of PROTOCOL.md,
build_artifacts.py, suites.py and every artifact table. On every later run it *asserts* those hashes
and refuses to proceed if the protocol, the suites or the data changed without a recorded amendment.
Pass `--amend "<reason>"` to deliberately re-freeze; the previous plan is preserved in
`FROZEN_PLAN_HISTORY.jsonl` and the reason is recorded.

Scoring follows PROTOCOL.md section 6 exactly: a case counts as detected by an arm only when the
faulty batch fails and the paired reference batch passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Great Expectations ships usage analytics that post to an external endpoint. This experiment must
# not transmit anything about private artifacts, so analytics are disabled before GX is imported.
os.environ["GX_ANALYTICS_ENABLED"] = "false"
os.environ["DO_NOT_TRACK"] = "true"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

import great_expectations as gx  # noqa: E402
import suites as S  # noqa: E402

MANIFEST_PATH = HERE / "artifact_manifest.json"
PLAN_PATH = HERE / "FROZEN_PLAN.json"
HISTORY_PATH = HERE / "FROZEN_PLAN_HISTORY.jsonl"
CASES = ["RD1", "RD2", "RD4", "RD5", "RD6", "RD7", "RD8"]
ARMS = ["A_declared_schema", "B_disjoint_reference_fitted", "C_information_enriched"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def current_plan(manifest: dict) -> dict:
    return {
        "protocol_sha256": sha256_file(HERE / "PROTOCOL.md"),
        "build_artifacts_sha256": sha256_file(HERE / "build_artifacts.py"),
        "suites_sha256": sha256_file(HERE / "suites.py"),
        "artifact_manifest_sha256": sha256_file(MANIFEST_PATH),
        "artifact_table_sha256": {
            f"{case}/{kind}": t["sha256"]
            for case, entry in manifest["cases"].items()
            for kind, t in entry["tables"].items()
        },
        "cases": CASES,
        "arms": ARMS,
    }


def enforce_freeze(plan: dict, amend: str | None) -> dict:
    if not PLAN_PATH.exists():
        record = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "amendment": None, **plan}
        PLAN_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print("froze plan (first run)")
        return record
    stored = json.loads(PLAN_PATH.read_bytes())
    drift = {k: (stored.get(k), plan[k]) for k in plan if stored.get(k) != plan[k]}
    if drift and not amend:
        raise SystemExit(
            "frozen plan mismatch; re-run with --amend '<reason>' to record a deliberate change:\n"
            + json.dumps({k: {"frozen": a, "now": b} for k, (a, b) in drift.items()}, indent=2)[:4000]
        )
    if drift:
        with HISTORY_PATH.open("a") as fh:
            fh.write(json.dumps(stored, sort_keys=True) + "\n")
        record = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "amendment": amend, **plan}
        PLAN_PATH.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"re-froze plan with amendment: {amend}")
        return record
    print("frozen plan verified unchanged")
    return stored


def read_table(manifest: dict, case: str, kind: str) -> pd.DataFrame:
    entry = manifest["cases"][case]["tables"][kind]
    path = REPO / entry["path"]
    assert sha256_file(path) == entry["sha256"], f"{case}/{kind} hash drift"
    return pd.read_parquet(path)


def validate(make_expectations, df: pd.DataFrame, label: str) -> dict:
    """Run one suite natively through Great Expectations against one pandas batch.

    `make_expectations` is a zero-argument factory: GX binds an Expectation object to the suite it is
    added to, so each validation needs its own freshly constructed, identically parameterised set.
    """
    expectations = make_expectations()
    context = gx.get_context(mode="ephemeral")
    asset = context.data_sources.add_pandas(f"src_{label}").add_dataframe_asset(f"asset_{label}")
    batch_definition = asset.add_batch_definition_whole_dataframe(f"bd_{label}")
    suite = context.suites.add(gx.ExpectationSuite(name=f"suite_{label}"))
    for expectation in expectations:
        suite.add_expectation(expectation)
    definition = context.validation_definitions.add(
        gx.ValidationDefinition(name=f"vd_{label}", data=batch_definition, suite=suite))
    result = definition.run(batch_parameters={"dataframe": df})
    per_expectation, raised = [], []
    for r in result.results:
        cfg = r.expectation_config
        # An Expectation whose metric raised also reports success=False. That is a failure to
        # execute, not a detection, and must never be scored as one.
        info = r.exception_info or {}
        exceptions = [v for v in info.values() if isinstance(v, dict) and v.get("raised_exception")]
        if exceptions:
            raised.append({"type": cfg.type,
                           "message": str(exceptions[0].get("exception_message"))[:500]})
        per_expectation.append({
            "type": cfg.type,
            "kind": "custom" if cfg.type in S.CUSTOM_TYPES else "builtin",
            "kwargs": {k: v for k, v in cfg.kwargs.items() if k != "batch_id"},
            "success": bool(r.success),
            "raised_exception": bool(exceptions),
            "unexpected_count": r.result.get("unexpected_count"),
            "element_count": r.result.get("element_count"),
            "observed_value": r.result.get("observed_value"),
        })
    if raised:
        raise RuntimeError(f"expectation raised during validation: {json.dumps(raised)[:800]}")
    return {
        "success": bool(result.success),
        "expectation_count": len(per_expectation),
        "failed_expectations": [e["type"] for e in per_expectation if not e["success"]],
        "rows_validated": int(len(df)),
        "expectations": per_expectation,
    }


def jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amend", default=None)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_bytes())
    plan = enforce_freeze(current_plan(manifest), args.amend)

    outcomes = []
    for case in CASES:
        notes = manifest["cases"][case]["notes"]
        public = {b: read_table(manifest, case, f"{b}_public") for b in ("faulty", "reference")}
        enriched = {b: read_table(manifest, case, f"{b}_enriched") for b in ("faulty", "reference")}

        for arm in ARMS:
            record = {"case": case, "arm": arm, "branches": {}}
            try:
                if arm == "A_declared_schema":
                    make = lambda c=case, n=notes: S.arm_a(c, n)  # noqa: E731
                    batches = {b: public[b] for b in public}
                    record["information"] = "declared schema and documented bounds only"
                elif arm == "B_disjoint_reference_fitted":
                    fixture_mask, validate_mask = S.arm_b_split(case, public["reference"])
                    if fixture_mask is None:
                        fixture = read_table(manifest, case, "armb_fixture")
                        record["fixture"] = "dataset-published companion statistics table"
                    else:
                        fixture = public["reference"][fixture_mask].reset_index(drop=True)
                        record["fixture"] = "disjoint split of the reference artifact"
                    make = lambda f=fixture: S.arm_b(f)  # noqa: E731
                    batches = {b: public[b][validate_mask].reset_index(drop=True) for b in public}
                    record["fixture_rows"] = int(len(fixture))
                    record["information"] = "parameters fitted on a disjoint reference fixture"
                else:
                    make = lambda c=case, n=notes: S.arm_c(c, n)  # noqa: E731
                    batches = {b: enriched[b] for b in enriched}
                    record["information"] = "case reference information in batch and/or parameters"

                for branch, df in batches.items():
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        record["branches"][branch] = validate(make, df, f"{case}_{arm}_{branch}")
                record["status"] = "EXECUTED"
            except Exception as exc:  # noqa: BLE001  failure to execute is a distinct outcome
                record["status"] = "FAILED_TO_EXECUTE"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()[-2000:]

            if record["status"] == "EXECUTED":
                faulty_ok = record["branches"]["faulty"]["success"]
                reference_ok = record["branches"]["reference"]["success"]
                record["faulty_outcome"] = "NOT_DETECTED" if faulty_ok else "DETECTED"
                record["reference_outcome"] = "ACCEPTED" if reference_ok else "FALSE_POSITIVE"
                if not faulty_ok and reference_ok:
                    record["verdict"] = "DETECTED"
                elif not faulty_ok and not reference_ok:
                    record["verdict"] = "ALWAYS_FAIL"
                elif faulty_ok and reference_ok:
                    record["verdict"] = "MISSED"
                else:
                    record["verdict"] = "REFERENCE_FALSE_POSITIVE_ONLY"
            else:
                record["verdict"] = "NOT_EXECUTED"
            outcomes.append(record)
            print(f"{case:5s} {arm:30s} {record['verdict']}"
                  + (f"  faulty={record.get('faulty_outcome')} reference={record.get('reference_outcome')}"
                     if record["status"] == "EXECUTED" else f"  {record.get('error','')}"), flush=True)

    tally = {arm: {v: sum(1 for o in outcomes if o["arm"] == arm and o["verdict"] == v)
                   for v in ("DETECTED", "MISSED", "ALWAYS_FAIL",
                             "REFERENCE_FALSE_POSITIVE_ONLY", "NOT_EXECUTED")}
             for arm in ARMS}

    report = {
        "experiment_id": "GX-BASE-20260914",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "great_expectations": md.version("great_expectations"),
            "pandas": pd.__version__, "numpy": np.__version__,
            "pyarrow": md.version("pyarrow"),
            "python_version": platform.python_version(), "platform": platform.platform(),
        },
        "frozen_plan": plan,
        "cases_evaluated": CASES,
        "excluded_cases": manifest["excluded_cases"],
        "denominator": len(CASES),
        "scoring": (
            "a case counts as detected by an arm only when the faulty batch fails and the paired "
            "reference batch passes; faulty-fail with reference-fail is ALWAYS_FAIL"
        ),
        "tally": tally,
        "outcomes": jsonable(outcomes),
    }
    (HERE / "gx_baseline_results.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = ["# GX-BASE-20260914 results", "",
             f"Great Expectations {report['environment']['great_expectations']}, "
             f"pandas {pd.__version__}, NumPy {np.__version__}, CPython {platform.python_version()}.", "",
             "| Case | Arm A declared schema | Arm B disjoint-reference fitted | Arm C information-enriched |",
             "|---|---|---|---|"]
    for case in CASES:
        row = [case]
        for arm in ARMS:
            o = next(x for x in outcomes if x["case"] == case and x["arm"] == arm)
            row.append(o["verdict"])
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Tally:", "```", json.dumps(tally, indent=2), "```",
              "", "RD3 excluded: " + manifest["excluded_cases"]["RD3"]]
    (HERE / "gx_baseline_results.md").write_text("\n".join(lines) + "\n")

    print("\n" + json.dumps(tally, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
