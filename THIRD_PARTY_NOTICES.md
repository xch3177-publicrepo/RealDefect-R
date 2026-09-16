# Licenses, attribution and input disposition

Author-owned replay adapters, analyses and build glue use the license selected in `LICENSE`;
author-owned manuscript/documentation use `LICENSE-DOCUMENTATION`. These grants do not relicense
third-party code, dataset bytes, derived fixtures or IEEEtran. Pinned original notices and dataset
cards are under `LICENSES/`; `LICENSES/SOURCES.json` gives exact retrieval URLs and SHA-256.
Public source provenance is intentionally retained after privacy minimization.

| Component | Included / recovery / transformations | Governing source and terms |
|---|---|---|
| `inputs/rd8/projections/*.npz` (123files) | Bundled lossless projection of `observation.state`, `action`, `episode_index`, `frame_index`, `timestamp`; image columns removed; values/dtypes retained; container changed to NPZ | PhysicalIntelligence's `aloha_pen_uncap_diverse@e82d8b40b8ac66c0b40273dd80a077dfc40b732e` card declares Apache2.0. Attribution:BiPlay, S.Dasari, O.Mees, S.Zhao, M.K.Srirama, S.Levine, “The Ingredients for Robotic Diffusion Transformers,” ICRA2025. See `LICENSES/aloha_dataset_card.md` and `Apache-2.0.txt`. |
| `inputs/rd8/official-metadata/*` and `reproducibility/light_inputs/fixtures/openpi-aloha-pen-uncap-row0-lowdim.json` | Four upstreammetadata files and selected low-dimensional first-row fixture bundled; fixture is a subset derivative | Same pinned dataset/Apache2.0 attribution; fixture transformation recorded in its source. |
| `inputs/rd8/projection-checkpoints/*.json` | Bundled author-generated identity/projection records, not analysis completion cache | Underlying dataset provenance unchanged;author-generated metadata license follows `LICENSE`. |
| `reproducibility/light_inputs/data/raw/lerobot/pusht/**` | One full publicepisode-metadata Parquet bundled; no video/image assets | `lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19`,MIT per pinned card. LeRobot conversion of PushT; attribution: ChengChi and the DiffusionPolicy contributors, see `LICENSES/PushT-DiffusionPolicy-MIT.txt` and the card. |
| `reproducibility/light_inputs/data/raw/lerobot/libero-v21/**` | One upstream episode-metadata JSONL bundled | `HuggingFaceVLA/libero@affa19c0de0f6bce2a7edd26dddef8a532e7e6f6`,CC-BY4.0 per pinned card. Attribution:LIBERO dataset and HuggingFaceVLA conversion;source card retained. |
| GX paired fixtures | Generated locally, not bundled as opaque Parquet caches; includes author-constructed RD4 counterfactual and ALOHA-derived RD6 rows, both explicitly labeled | Underlying inputs retain the terms above;IsaacLab and robomimic inputs are fetched from public pinned URLs, not relicensed by this package. Result aggregates and code do not grant rights over rawdata. |
| RD5 representative result record | A tiny transformed/selected numerical example is retained in `reproducibility/results/real-defect-robosuite-626.json`; the9666row raw HDF5 is recovered rather than bundled | `robomimic/robomimic_datasets@33d08702ad61296b55a74c4f810ab54462f319e9` declares MIT;see `LICENSES/robomimic_dataset_card.md`. Attribution:robomimic dataset authors/contributors;transform semantics derived from the pinned robosuite source. |
| LeRobot/OpenPI/Isaac-GR00T pinned Python files and embedded historical bodies | Source extracts and branch-compatible replays;original headers preserved;each source pin/hash recorded | LeRobot, PhysicalIntelligence/OpenPI, NVIDIA/Isaac-GR00T licenses under `LICENSES/`;Apache2.0 and any additional upstream notices apply. |
| IsaacLab pinned source / branch bodies | Source files/derived replay logic;complete HDF5 input recovered, not bundled | BSD3-Clause copyright IsaacLab ProjectDevelopers, see `LICENSES/IsaacLab-BSD-3-Clause.txt`;downloaded dataset retains its own distribution terms. |
| robosuite source / replay branch bodies | Pinned source and scoped rigid-transform calculation;robomimic HDF5 fetched, not bundled | MIT, copyright StanfordVisionandLearningLab and UTRobotPerceptionandLearningLab;see `LICENSES/robosuite-MIT.txt` including the additional MuJoCo notice. |
| Cosmos3-DROID | No raw frame files or full metadata tables bundled;pin/hash-verified metadata recovery and range-read identity checks;analytical summaries included | `nvidia/Cosmos3-DROID@dabaaffe428d67cf93fd355b82658934ee59bfec`;OpenMDW1.1 per pinned card;all upstream attribution and terms retained in `LICENSES/Cosmos_dataset_card.md`. |
| Isaac-GR00T10MP4 inputs | Input recovery only;full-file digests pinned in manifest;decoded RGB data not redistributed | Pinned NVIDIA dataset card declares CC-BY4.0;see `LICENSES/GR00T_sim_dataset_card.md`. The result JSON retains timestamp/digest/shape records,not decodedRGBframes. |
| `paper/IEEEtran.cls` | Unmodified IEEE class bundled | Original header/license retained;LaTeXProjectPublicLicense1.3. |

`SOURCE_MAP.json` and `MANIFEST.json` enumerate the exact included files. No blanket statement that
all third-party data is re-fetched applies: the NPZ projections and small metadata/fixtures above
are deliberately redistributed with source identity and applicable attribution. This is a release
inventory, not a claim to expand any upstream permission.
