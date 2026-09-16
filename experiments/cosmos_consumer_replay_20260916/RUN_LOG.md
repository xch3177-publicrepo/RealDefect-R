# Work record: scoped consumer replay

Purpose (UTC 2026-09-16): upgrade the earlier source-inspected exposure to a
controlled executable consequence, without claiming a full merge, production
use, a new historical defect or model harm.

`PROTOCOL.json` fixed the selection, source commit/hash, eight summary fields,
intervention and offsets before consumer output was observed. The first run
(03:35:08–03:35:14 UTC) executed the exact pinned `update_meta_data` function text.
Both selected affected records retained eight incorrect output summary values;
the summary-only correction removed all sixteen. Both unaffected controls passed
both conditions. Actual selected frame rows total 1,095; this is a designed
selection, not a prevalence estimate.

`clean_recovery` reused only metadata newly fetched from public URLs by the C5
clean-input invocation and fetched its additional task tables, source and frame
columns. It took no private-cache argument and reproduced the first records
exactly. `RECOVERY_COMPARISON.json` records the equality checks.

Fresh public dependency environment: NumPy 1.26.4, pandas 2.1.4, PyArrow 14.0.2,
huggingface_hub 0.34.4. The first invocation (`public_env_compat`) failed before
consumer execution because urllib could not find a usable TLS root store. The
source and traceback are retained privately; `FAILURE_SUMMARY.json` is the
sanitized public account. Both runners were amended to use certifi TLS roots,
without disabling verification or changing any scientific rule.

`public_env_compat2` then executed successfully in that fresh environment, with
every selected-frame digest and output value identical to the earlier run. Its
result binds the current runner's source hash. Earlier run directories preserve
the exact pre-amendment executed script, so no historical evidence is overwritten.

Commands use a portable interpreter spelling and repository-relative paths:

```sh
python experiments/cosmos_consumer_replay_20260916/run_consumer.py --metadata-cache VERIFIED_METADATA --out NEW_FIRST_RUN
python experiments/cosmos_consumer_replay_20260916/run_consumer.py --work experiments/work/cosmos_c5_clean_20260916 --out experiments/cosmos_consumer_replay_20260916/clean_recovery
python experiments/cosmos_consumer_replay_20260916/run_consumer.py --metadata-cache experiments/work/cosmos_c5_clean_20260916/metadata --work experiments/work/cosmos_compat_20260916 --out experiments/cosmos_consumer_replay_20260916/public_env_compat2
```

Per-run JSON includes UTC times, versions, source/function/input hashes, frame
range receipts and condition outputs. No commit or push was performed by the
experiment worker; project integration records those later.
