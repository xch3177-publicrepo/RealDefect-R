# RD8 fixed-branch condition matrix (extracted, not re-run)

Source: `reproducibility/results/real-defect-openpi-570.json` sha256 `f6193ca2c4a7111fbe25a59c908927bd63a838fb3cab982037b88fed0f03b6b7`
Frozen status preserved: **NOT_REPRODUCED**; tolerance 1e-05.

Cell = max absolute error of the fixed branch against the float64 oracle over
{state, actions} x {mean, std}; `FAIL` marks a cell above the frozen tolerance.

| Order | batch 1 | batch 8 | batch 32 | batch 128 |
|---|---|---|---|---|
| natural | 2.099e-04 FAIL | 4.164e-06 PASS | 5.269e-06 PASS | 3.093e-06 PASS |
| episode_cluster_seed_0 | 1.995e-04 FAIL | 4.910e-05 FAIL | 1.267e-05 FAIL | 3.533e-06 PASS |
| episode_cluster_seed_1 | 4.692e-05 FAIL | 2.572e-05 FAIL | 1.213e-05 FAIL | 1.802e-06 PASS |
| episode_cluster_seed_2 | 6.884e-05 FAIL | 1.451e-05 FAIL | 6.689e-06 PASS | 5.266e-06 PASS |
| episode_cluster_seed_3 | 1.508e-04 FAIL | 2.221e-05 FAIL | 1.461e-05 FAIL | 6.913e-06 PASS |
| episode_cluster_seed_4 | 1.687e-04 FAIL | 2.120e-05 FAIL | 1.947e-05 FAIL | 4.628e-06 PASS |
| episode_cluster_seed_5 | 3.235e-04 FAIL | 2.230e-05 FAIL | 9.076e-06 PASS | 2.982e-06 PASS |
| episode_cluster_seed_6 | 1.565e-04 FAIL | 9.620e-06 PASS | 3.061e-06 PASS | 4.077e-06 PASS |
| episode_cluster_seed_7 | 1.753e-04 FAIL | 3.061e-06 PASS | 1.342e-05 FAIL | 5.731e-06 PASS |
| episode_cluster_seed_8 | 1.020e-04 FAIL | 2.157e-05 FAIL | 3.253e-06 PASS | 7.929e-06 PASS |
| episode_cluster_seed_9 | 1.798e-04 FAIL | 1.021e-05 FAIL | 6.275e-06 PASS | 5.731e-06 PASS |

Fixed-branch failures: 24/44.
Failure drivers (condition counts): {"state.std": 19, "actions.std": 19, "state.mean": 7, "actions.mean": 6}
Of the 24 failing conditions, 24 involve a std vector above tolerance and 0 fail on the mean alone.
Maximum fixed error 0.00032345332925 on state.std at order episode_cluster_seed_5, batch 1.

Batch size 1 (11 conditions): old branch equals fixed branch exactly for every order = True; fixed branch passes the frozen criterion in 0/11 of them (max mean error 2.804e-05, max std error 3.235e-04).
