# Eight-case registry and executable scope

Pins, public URLs, input digests and thresholds are retained in
`original/contracts/real-defect-corpus-manifest.json`, its freeze amendment,
`reproducibility/light_inputs/downloads.json`, `upstream_pins_verification.json` and each result's
`frozen_inputs` field. `SOURCE_MAP.json` binds the scoped historical code and current adapters.

| Case | Public upstream record | Scope and reference | Expected retained result |
|---|---|---|---|
| RD1 | [LeRobot#2610/#2612](https://github.com/huggingface/lerobot/issues/2610) | E3 absolute-index clamp and padding;118queries | faulty118/118 contract decisions, reference0 |
| RD2 | [LeRobot#2057](https://github.com/huggingface/lerobot/pull/2057) | E2 accumulation-loop replay;5episodes;global continuity | episode4 begins285 versus expected1128 |
| RD3 | [IsaacLab#4559](https://github.com/isaac-sim/IsaacLab/pull/4559) | E3 exact quaternion constants/convention | faulty120degrees, reference0 |
| RD4 | [IsaacLab pinned converter](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py) | Single upstream-body coverage audit;field inventory and coefficient convention;not E1 |15141/23833 unconverted;no repaired counterpart executed |
| RD5 | [robosuite#626](https://github.com/ARISE-Initiative/robosuite/pull/626) | E2 displacement-versus-position transform on9666rows | faulty translation leakage;reference0;raw normalized actions remain in[-1,1] |
| RD6 | [OpenPI#557](https://github.com/Physical-Intelligence/openpi/pull/557) | E2 exact gripper branches;2encoder anchors+1observed row | old calibration error while internal roundtrip passes |
| RD7 | [Isaac-GR00T#172/#373](https://github.com/NVIDIA/Isaac-GR00T/issues/172) | E2 compatibility replay on10videos;PTS/RGB correspondence | old200/200 noncontrol mismatches;reference0 |
| RD8 | [OpenPI#570/#619/#623](https://github.com/Physical-Intelligence/openpi/issues/570) | E2 raw reducer on123episodes/36900rows;44orders×batchsize conditions | coverage restored;24/44 numerical failures;NOT_REPRODUCED |

Each case has a dedicated runner under `original/scripts/`; filenames are mapped in
`reproducibility/reproduce.py`. E2 is a historical branch/loop replay, not the whole application.
RD4 is an audit, and E3 is exact constants/minimal arithmetic. The eight purposive retrospective
histories do not estimate real-world prevalence. The frozen contract and each runner contain the complete executable rule and pinned input identities.
