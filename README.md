# RealDefect-R — minimal public research snapshot

**Replaying well-formed but wrong data in shared robot-learning pipelines.** This standalone,
sanitized snapshot supplies the eight-case registry, scoped replay code, case rules and expected
outcomes, native Great Expectations suites, the RD8 arithmetic diagnostic, Cosmos metadata and
frame-identity checks, and the paper's figure/build sources. It carries **no private git history**.

Start with [REPRO.md](REPRO.md), [REGISTRY.md](REGISTRY.md), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The complete file manifest is `MANIFEST.json`; `SOURCE_MAP.json` binds the selected historical
source and every current experiment/script/result. The fixed public version is recorded in
`ARTIFACT_INFO.json`. This package version is `public-v1.1.2-20260919`; the previous fixed tag
remains available. Input recovery uses public pinned revisions and explicit digest verification.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_public.py
.venv/bin/python run_public.py --suite smoke --work-dir /tmp/realdefect-smoke-NEW
```

For a short end-to-end experiment, run native GX on all seven paired fixtures:

```sh
.venv/bin/python run_public.py --suite gx --download --work-dir /tmp/realdefect-gx-NEW
```

This executes 42 branch validations and six frozen-result/plan comparisons. It requires about
32.6 MB of public input payloads, excluding dependency installation. [REPRO.md](REPRO.md) includes
the fixed-tag download command, which needs no GitHub login.

Verified execution evidence: the [19 September 2026 public-v1.0.1 receipt](https://github.com/xch3177-publicrepo/RealDefect-R/releases/download/public-v1.0.1-20260919/PUBLIC_ACCESS_VERIFICATION.json) records anonymous retrieval followed by 42 completed native-GX validation branches and six matching comparisons, with fresh input/run directories and a preinstalled package-only environment. Version v1.1.0 adds a separately versioned source-prefix regrouping of the pinned Cosmos metadata and evidence-bound scientific figures. The original case, GX, C1-C5 and scoped consumer calculations are unchanged. The runner now regenerates frozen-evidence figures before extracting a matrix with fresh runtime provenance; the new grouping analysis has its own protocol and recovery instructions. Expected faulty-artifact rejections are successful scientific outcomes, not failed installation or execution.

The smoke and GX routes are **not** all eight replays. A full run is:

```sh
.venv/bin/python run_public.py --suite all --download --work-dir /tmp/realdefect-all-NEW
```

`RD8: NOT_REPRODUCED` is an expected scientific negative after successful execution: full repaired
coverage with 24 of 44 numerical conditions outside the frozen tolerance. Infrastructure failures
are reported separately and do not count as this expected negative. No model training or model
harm experiment is part of this resource.

The public mirror preserves public upstream source/dataset attribution. Sanitization removes
private working paths, account links, personal contact details, unrelated research and internal
review/agent material. It is a minimized single-blind submission artifact, **not a claim of full
anonymity**. Licenses in `LICENSES/` and the itemized notices govern third-party material.
