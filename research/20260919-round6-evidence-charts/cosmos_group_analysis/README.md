# Cosmos source-prefix chart analysis

This is an additional metadata derivation on the pinned release, not a new
consumer replay or a replacement for the frozen metadata/C1--C5 results.

## Inputs and execution

- Dataset revision: `dabaaffe428d67cf93fd355b82658934ee59bfec`.
- Whole-file identities: all 12 files in the existing
  `experiments/results/20260912-cosmos-metadata/verified_inputs.json` were checked
  again, totaling 389,225,505 bytes. Files were read from the existing local cache;
  no network download was needed. No cached input or old result was modified.
- Analysis: `scripts/cosmos_group_chart_data.py`, SHA-256 recorded in the run
  protocol and result. The script imports no historical audit implementation.
- `run1/PROTOCOL.json` was written before input verification or calculation.
- `run1/VERIFIED_INPUTS.json`, `results.json`, both CSVs, and their
  `MANIFEST.json` preserve the result. `run1.log` is the raw console transcript.
- Runtime: Python 3.12.2, NumPy 1.26.4, pandas 2.2.2, PyArrow 14.0.2. PyArrow
  printed non-fatal sandbox CPU-cache probing warnings; the run completed
  successfully. Every reported split/global/group count matched the frozen
  independent audit, and the new output manifest was rechecked.

## Replay from the public snapshot

Run the following from the root of a fixed public snapshot that includes this
analysis. First follow its `REPRO.md` to create `.venv` and install
`requirements.txt`. Use a new sibling directory named `cosmos-group-replay`;
the shipped evidence remains read-only. The commands below recover metadata
only: they do not rerun C1--C5, read frame/video files, or execute the consumer.

```sh
.venv/bin/python verify_public.py
SSL_CERT_FILE="$(.venv/bin/python -m certifi)" \
  .venv/bin/python experiments/replay_cosmos_metadata.py \
  --source-root original \
  --data-root ../cosmos-group-replay/metadata \
  --out ../cosmos-group-replay/recovery \
  --download-only
.venv/bin/python scripts/cosmos_group_chart_data.py \
  --data-root ../cosmos-group-replay/metadata \
  --manifest ../cosmos-group-replay/recovery/verified_inputs.json \
  --reference experiments/results/20260912-cosmos-metadata/independent_referential_check.json \
  --output-dir ../cosmos-group-replay/analysis
```

The first recovery step verifies the same 12 whole-file SHA-256 identities,
totaling 389,225,505 bytes. Its input list comes from the public snapshot's
`original/results/p8-droid-full-metadata-audit.json`; the needed historical
audit script and two modules are already bundled under `original/`. No private
source archive is required. `--download-only` writes a fresh recovery manifest;
it does not require or generate `rerun.json`. The new analysis reads only the
recovered metadata, its manifest, and the bundled independent count reference.

If a public `run_public.py --suite extended` or `--suite all` run already
completed metadata recovery, its existing cache is at
`WORK_DIRECTORY/runtime/experiments/work/cosmos3-droid`. Supply that path as
`--data-root` and `WORK_DIRECTORY/cosmos-metadata/preaudit_verified_inputs.json`
as `--manifest`, with a new analysis output directory. This skips downloading
the same inputs again while still rechecking every full-file hash. The cache
and manifests must originate from the fixed public snapshot, not an author's
private checkout.

Compare the newly computed scientific fields and chart CSVs against the shipped
record (timestamps, runtime versions, and recovery-manifest hashes may differ):

```sh
.venv/bin/python - <<'PY'
import csv
import json
from pathlib import Path

fresh = Path('../cosmos-group-replay/analysis')
frozen = Path('research/20260919-round6-evidence-charts/cosmos_group_analysis/run1')
a = json.loads((fresh / 'results.json').read_text())
b = json.loads((frozen / 'results.json').read_text())
for key in ('source_revision', 'combined', 'splits'):
    assert a[key] == b[key], key
for name in ('by_source_prefix.csv', 'by_split_source_prefix.csv'):
    with (fresh / name).open() as x, (frozen / name).open() as y:
        assert list(csv.DictReader(x)) == list(csv.DictReader(y)), name
print('PASS: 13 groups, 71,907 records, 61,502 identity discrepancies; chart data match.')
PY
```

An exception, failed input hash, or failed comparison is a failed replay; do not
replace its results with the shipped CSVs. Retain the output directory and use
a new directory for any corrected attempt.

The plotting path should read the frozen CSV/JSON with the Python standard
library. It does not need to import or execute this metadata analysis.

## Group definition and scope

The grouping key is the substring of `episode_id` before its first `/`.
There is no separate lab or institution column in the inspected metadata
schema. Exactly 13 nonempty prefixes occur in **each** split, with the same set
in both splits and zero missing-prefix rows. Thus "18 labs" is unsupported.
The chart must label these as **source groups** or **episode-ID source prefixes**;
it must not count them as independently verified institutions.

Within each split, records are sorted by global `episode_index`. A local episode
ordinal and cumulative frame start are derived separately for every prefix.
The seven existing identity-summary fields are evaluated against both the global
and local predictions, using the frozen scalar tolerance:

`abs(actual - expected) <= max(1e-5, 1e-7 * max(abs(actual), abs(expected)))`.

The seven fields are `stats/episode_index/{min,max,mean,std}` and
`stats/index/{min,max,mean}`. These seven audit fields are **not** the eight
episode-index fields evaluated by the separate scoped consumer experiment.

## Results

All seven identity-summary fields match the source-prefix-local predictions
on all 71,907 records. Identity-summary coherence with the global references is
equivalent, record by record, to local/global episode and frame numbering
coinciding. All 26 split/source-group combinations have constant episode and
frame offsets; both offsets are zero only for `AUTOLab`.

| Source prefix | Episodes, both splits | Identity-discrepant episodes | Rate |
|---|---:|---:|---:|
| TRI | 19,825 | 19,825 | 100% |
| RAIL | 8,100 | 8,100 | 100% |
| IPRL | 5,713 | 5,713 | 100% |
| CLVR | 5,090 | 5,090 | 100% |
| REAL | 5,008 | 5,008 | 100% |
| IRIS | 3,362 | 3,362 | 100% |
| ILIAD | 3,240 | 3,240 | 100% |
| WEIRD | 2,989 | 2,989 | 100% |
| PennPAL | 2,894 | 2,894 | 100% |
| RPL | 2,517 | 2,517 | 100% |
| RAD | 1,467 | 1,467 | 100% |
| GuptaLab | 1,297 | 1,297 | 100% |
| AUTOLab | 10,405 | 0 | 0% |
| **Total** | **71,907** | **61,502** | **85.53%** |

The 10,405 coherent records consist of 7,016 success-split and 3,389
failure-split records, all from the `AUTOLab` prefix. The other 12 prefixes have
100% identity-summary disagreement in **both** splits. "Success" and "failure"
are the dataset's episode-outcome split names, not audit verdicts.

## Safe manuscript statement

> Grouping by the source prefix in `episode_id` yields 13 groups. All 10,405
> globally coherent identity summaries belong to the AUTOLab prefix, where
> local and global episode/frame numbering coincide; every record in the other
> 12 groups violates the identity-summary relation. Across all 71,907 records,
> the seven checked summaries agree with source-prefix-local numbering.

This establishes an observed numerical relationship in the pinned metadata.
It does not identify the upstream code path that introduced it, attribute fault
to a laboratory, estimate institution-level prevalence, or establish downstream
model harm. The source-group panel must specifically display **identity-summary
disagreement**; task-summary disagreement has a different pattern, retained in
the CSV and in the proposed joint-distribution panel.
