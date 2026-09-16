# Scoped LeRobot consumer replay

This controlled execution tests the exact `update_meta_data` function at LeRobot
commit `7e241bd630a3719a56157a497ce5d08f244784f1` (v0.6.1). It is not a full
dataset merge. No videos are decoded, no output frame files are created, and no
production usage or model-performance consequence is inferred.

Python 3.12 or later is needed to parse the pinned source. From the repository root:

```sh
python -m pip install -r experiments/cosmos_c5_20260916/requirements.txt
python experiments/cosmos_consumer_replay_20260916/run_consumer.py --out /tmp/cosmos-consumer-new-run
```

The standalone requirements above and the public package's shared NumPy 1.26.4,
pandas 2.1.4, PyArrow 14.0.2, huggingface_hub 0.34.4 environment both reproduced
the exact scientific record. Use the public package's root requirements when
working in that shared environment. `CURRENT_RESULTS.json` identifies the
accepted current-runner result in `public_env_compat2/results.json`; it includes
the verified-TLS-root transport correction described in `FAILURE_SUMMARY.json`.

The output directory must not exist. Public input recovery is the default: the
runner downloads the pinned source, six episode metadata shards and two task
tables, verifying every full-file digest in `INPUTS.json`, then range-reads three
frame identity columns for the four selected frame files. `--work DIRECTORY`
selects an ignored working cache. Optional `--metadata-cache DIRECTORY` accepts
only full-hash-verified metadata and task tables. No private path is required.

The exact function text is extracted with `ast.get_source_segment`. Its body is
unchanged; annotations are deferred and its NumPy/pandas dependencies supplied,
so the full robotics dependency graph is unnecessary. The selected metadata
columns include episode, task and file identifiers and summary statistics. The
video map is empty. Execution therefore covers this metadata update function on
the stated columns and branches, not the whole merge pipeline.

`PROTOCOL.json` was frozen before consumer execution. In each split, it selects
the lowest episode ID with inconsistent min/max/mean episode summaries, and the
lowest agreeing control. IDs are failure 3389 and 0; success 7016 and 0. The runner
recomputes this selection from the verified metadata and refuses a mismatch. In
both conditions the source records, frame identities and offsets (100,000 episodes,
30,000,000 frames) are identical. The intervention replaces only the eight
`stats/episode_index/{min,max,mean,q01,q10,q50,q90,q99}` fields with their expected
constant frame episode ID. It is a constructed control on actual release records,
not a new historical defect.

In `run_20260916T033508Z/results.json`, both affected records retain errors in all
eight output episode-summary fields: 16 violating fields across two records. All
16 pass after the summary-only correction; both unaffected controls pass in both
conditions. The four selections contain 1,095 actual frame rows. This demonstrates
an executable metadata consequence under fixed conditions, not defect prevalence.
The result contains source/function digests, decoded selected-frame digests, range
receipts, versions, inputs and outputs. `EXECUTED` denotes completed execution;
its condition-level violations are scientific results, not runtime errors.
