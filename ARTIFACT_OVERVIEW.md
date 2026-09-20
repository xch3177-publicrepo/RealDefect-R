# Artifact overview: evidence, commands and verdicts

Target snapshot: `public-v1.1.3-20260919`. Begin with the fixed snapshot and fresh
CPython 3.12 environment described in [REPRO.md](REPRO.md). Run commands from
that snapshot's root. Every work directory below is a **new sibling directory**;
do not substitute a development checkout, old execution directory or historical
cache. Keep the downloaded snapshot and its shipped expected results unchanged.

This document specifies how to execute and assess the evidence. It is not itself
an execution receipt. The release's access and replay receipts identify which
commands were actually completed on which snapshot. Earlier acceptance records
keep their original dates and scope; they do not certify new analyses by implication.

## Evidence correspondence

`W` below denotes the selected external work directory for a `run_public.py` command.
Use a different `W` for every invocation. All suite invocations produce `W/commands.json`,
`W/logs/` and `W/RUN_SUMMARY.json`; suites with scientific comparisons also produce
`W/SCIENTIFIC_COMPARISON.json`.

| Paper evidence | Fresh execution | Fresh result, relative to `W` | Expected scientific outcome and boundary |
|---|---|---|---|
| Table I: RD1–RD8 | `--suite cases --download` | `cases-all/run_summary.json`; per-case JSONs in `cases-all/code/results/`; `cross-case.json` | Six paired scoped replays and RD4's single-path audit meet their rules. RD8 retains `NOT_REPRODUCED`, complete repaired coverage and 24/44 numerical failures. Process success is not equivalent to a positive scientific verdict. |
| Table II: native GX | `--suite gx --download` | `GX_EXECUTION_COVERAGE.json`; `runtime/experiments/gx_baseline_20260914/gx_baseline_results.json`; six entries in `SCIENTIFIC_COMPARISON.json` | Seven pairs × three suites × two branches = 42 validations. Separated pairs: declared 1, fitted 1, informed 7. Inspect per-case outcomes, not only totals. This tests encoded rule portability. |
| Fig. 2: RD8 diagnostic | `--suite extended --download`, or the narrower route below | `runtime/experiments/rd8_diagnostic_20260914/rd8_precision_diagnostic_all_orders.json` | Five arms each execute 44 conditions. D1 reproduces all 44 frozen conditions, with 24 failures. D4 has nine mean failures, zero std failures. D3 meets the criterion in all 44 conditions, while retaining a nonzero std residual. |
| Fig. 3(a): metadata relations | `--suite extended --download` | `cosmos-metadata/rerun.json`; `cosmos-independent.json` | 71,907 episodes; 61,502 identity-summary discrepancies. Success/failure are dataset split names, not audit verdicts. |
| Fig. 3(b): source-prefix regrouping | Standalone recovery and regrouping below; **not in `all`** | Separate `analysis/results.json`, `by_source_prefix.csv`, `by_split_source_prefix.csv` | 13 prefixes; 10,405 globally coherent records in AUTOLab and 61,502 discrepant records in the other 12 prefixes. All seven checked summary fields agree with prefix-local predictions for all 71,907 records. The 26 prefix/split offsets plotted in Fig. 3(b) are constant within each group; only AUTOLab has zero offsets. Groups are not independently verified institutions. |
| Cosmos C1–C4 | `--suite extended --download` | `runtime/experiments/cosmos_frame_check_20260914/cosmos_frame_check_results.json` | Zero violations of the four specified constraints across 22,412,712 rows in 74 files. Set coverage does not establish every row-wise relation. |
| Cosmos C5 | `--suite extended --download` | `cosmos-c5/results.json` | Separate check of `index = dataset_from_index + frame_index` on every row; zero violations and complete coverage. This does not verify physical row order, images, timestamps or actions. |
| Table III: consumer | `--suite extended --download`, or the direct route below | `cosmos-consumer/results.json` | Four preselected records; eight `stats/episode_index/*` target fields per record. Affected records: 16/16 → 0/16; controls: 0/16 → 0/16. Exact scoped function execution, not full merge, prevalence estimation or model-harm measurement. |

## Supported suite commands

After installation, choose the route needed for the claim being checked:

```sh
.venv/bin/python run_public.py --suite gx --download --work-dir ../realdefect-gx
.venv/bin/python run_public.py --suite cases --download --work-dir ../realdefect-cases
.venv/bin/python run_public.py --suite extended --download --work-dir ../realdefect-extended
```

`extended` includes the smoke checks, GX, all-order RD8-PD, metadata audit,
C1–C4, C5 and consumer. It does not execute all eight case runners or the
source-prefix regrouping. `all` combines `cases` and `extended`; it still requires
the separate regrouping command for Fig. 3(b). A fresh `gx` result certifies only
the GX route and its stated prerequisite steps.

## Narrower RD8-PD route: no network after installation

The diagnostic writes next to its script. Therefore execute it in the isolated
runtime copy created by the smoke runner, never directly in the fixed snapshot:

```sh
.venv/bin/python run_public.py --suite smoke --work-dir ../realdefect-rd8pd
.venv/bin/python -B ../realdefect-rd8pd/runtime/experiments/rd8_diagnostic_20260914/rd8_precision_diagnostic.py all_orders > ../realdefect-rd8pd/logs/RD8_precision_diagnostic.log 2>&1
```

This uses 123 bundled, hash-verified low-dimensional projections, not the original
image-bearing Parquets. The smoke receipt covers only smoke; the subsequent
diagnostic has its own raw log and result file. Check its fresh numerical verdicts:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('../realdefect-rd8pd/runtime/experiments/rd8_diagnostic_20260914/rd8_precision_diagnostic_all_orders.json')
x = json.loads(p.read_text())
assert x['scope'] == 'all_orders'
assert x['d1_vs_frozen_rd8']['exact_matches'] == 44
assert len(x['arms']) == 5 * 44
expected = {
    'D1_native_pinned_float32': 24,
    'D2_pinned_float64_input': 0,
    'D3_float64_accumulator': 0,
    'D4_stable_variance_float32': 9,
    'D5_float64_accum_stable_variance': 0,
}
for arm, failures in expected.items():
    assert x['arm_rollup'][arm]['conditions'] == 44
    assert x['arm_rollup'][arm]['above_1e_5'] == failures
d4 = [r for r in x['arms'] if r['arm'] == 'D4_stable_variance_float32']
assert sum(r['max_absolute_mean_error'] > x['tolerance'] for r in d4) == 9
assert sum(r['max_absolute_std_error'] > x['tolerance'] for r in d4) == 0
print('PASS: fresh RD8-PD verdicts; original RD8 negative unchanged.')
PY
```

Rebuilding `paper/fig_rd8_residuals.tikz` alone reads shipped numerical evidence;
it is not a substitute for this diagnostic execution.

## Standalone source-prefix regrouping

These commands recover all 12 pinned metadata/companion objects into a new
directory, verify every full-file SHA-256, then compute the source groups:

```sh
.venv/bin/python verify_public.py
SSL_CERT_FILE="$(.venv/bin/python -m certifi)" \
  .venv/bin/python experiments/replay_cosmos_metadata.py \
  --source-root original \
  --data-root ../realdefect-groups/metadata \
  --out ../realdefect-groups/recovery \
  --download-only
.venv/bin/python scripts/cosmos_group_chart_data.py \
  --data-root ../realdefect-groups/metadata \
  --manifest ../realdefect-groups/recovery/verified_inputs.json \
  --reference experiments/results/20260912-cosmos-metadata/independent_referential_check.json \
  --output-dir ../realdefect-groups/analysis
```

Compare the fresh scientific fields and CSV rows, preserving independent
timestamps, environment and recovery provenance:

```sh
.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path
fresh = Path('../realdefect-groups/analysis')
frozen = Path('research/20260919-round6-evidence-charts/cosmos_group_analysis/run1')
a = json.loads((fresh / 'results.json').read_text())
b = json.loads((frozen / 'results.json').read_text())
for key in ('source_revision', 'combined', 'splits'):
    assert a[key] == b[key], key
for name in ('by_source_prefix.csv', 'by_split_source_prefix.csv'):
    with (fresh / name).open() as x, (frozen / name).open() as y:
        assert list(csv.DictReader(x)) == list(csv.DictReader(y)), name
print('PASS: fresh 13-group census and chart data agree with the shipped reference.')
PY
```

The [grouping protocol](research/20260919-round6-evidence-charts/cosmos_group_analysis/README.md)
defines all seven predicates and how to reuse inputs **from a completed public
snapshot run**, with full rehashing. This route does not read frame columns,
execute the consumer or identify the upstream converter that caused the pattern.

## Direct scoped consumer route

The consumer accepts explicit external input and output directories. The command
below downloads its own pinned inputs; it does not depend on a metadata cache:

```sh
mkdir ../realdefect-consumer
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1 \
  HF_HOME=../realdefect-consumer/hf-cache XDG_CACHE_HOME=../realdefect-consumer/cache \
  .venv/bin/python experiments/cosmos_consumer_replay_20260916/run_consumer.py \
  --work ../realdefect-consumer/work \
  --out ../realdefect-consumer/results \
  > ../realdefect-consumer/execution.log 2>&1
```

Fresh output is `../realdefect-consumer/results/results.json`; it records pinned
function/source identities, selected records, frame-range receipts and every
targeted field. Assess the two affected and two control records separately:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
x = json.loads(Path('../realdefect-consumer/results/results.json').read_text())
assert x['status'] == 'EXECUTED'
assert len(x['records']) == 4
for kind, original_fields in [('affected', 16), ('control', 0)]:
    rows = [r for r in x['records'] if r['kind'] == kind]
    assert len(rows) == 2
    for condition, expected in [('original', original_fields), ('summary_corrected', 0)]:
        assert all(len(r['conditions'][condition]['output_summary']) == 8 for r in rows)
        assert sum(r['conditions'][condition]['violating_field_count'] for r in rows) == expected
print('PASS: affected 16/16 -> 0/16; controls 0/16 -> 0/16 on the targeted fields.')
PY
```

## Resource cost and execution records

Install the pinned `requirements.txt` under CPython 3.12. GX additionally needs
`curl`; numerical/metadata routes use the same NumPy, pandas, PyArrow and
Hugging Face dependencies. A TeX installation is needed only to rebuild the PDF.

| Route | Input recovery beyond the public snapshot | Cost boundary |
|---|---|---|
| GX | Two HDF5 files and converter source: 32,561,298 bytes | Fixture construction plus 42 native GX validations; installation time is additional |
| Source groups | 12 full metadata/companion files: 389,225,505 bytes | Metadata-only census and hashing; no frame/video decoding |
| RD8-PD | No network input recovery | CPU numerical reductions over five arms × 44 conditions; slower than rendering the chart |
| Direct consumer | Eight full metadata/task files: 389,147,601 bytes, plus pinned Python source and frame-column HTTP ranges | Four preselected records; range reads may cover more bytes than the selected 1,095 rows; no video decoding |
| Extended/all | GX, Cosmos metadata, frame-column reads for C1–C5 and consumer; `all` additionally runs the eight cases | Independent readers can refetch inputs; do not treat the metadata size as total network traffic |

These are payload inventories, not promised download times or whole-workflow wire
totals. Current bandwidth, server availability and CPU affect elapsed time. The
suite runner stores per-command seconds in `commands.json`; standalone scripts
record start/end times in their result JSONs. Preserve installation logs, `pip
freeze`, interpreter/platform versions, the downloaded archive and manifest
digests, commands, exit codes, raw output and fresh result paths in an external
receipt. Record failures and unexecuted routes explicitly.

An external operator should identify their role and what they actually checked.
Running from a fresh directory is a clean execution, not evidence of independent
human review. A download or figure rebuild is not an experiment rerun. RD8's
expected scientific negative must never be relabeled as an installation failure
or silently replaced with a positive result.
