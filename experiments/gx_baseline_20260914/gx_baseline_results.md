# GX-BASE-20260914 results

Great Expectations 1.6.3, pandas 2.1.4, NumPy 1.26.4, CPython 3.12.3.

| Case | Arm A declared schema | Arm B disjoint-reference fitted | Arm C information-enriched |
|---|---|---|---|
| RD1 | MISSED | ALWAYS_FAIL | DETECTED |
| RD2 | MISSED | ALWAYS_FAIL | DETECTED |
| RD4 | MISSED | ALWAYS_FAIL | DETECTED |
| RD5 | DETECTED | ALWAYS_FAIL | DETECTED |
| RD6 | ALWAYS_FAIL | ALWAYS_FAIL | DETECTED |
| RD7 | MISSED | ALWAYS_FAIL | DETECTED |
| RD8 | MISSED | DETECTED | DETECTED |

Tally:
```
{
  "A_declared_schema": {
    "DETECTED": 1,
    "MISSED": 5,
    "ALWAYS_FAIL": 1,
    "REFERENCE_FALSE_POSITIVE_ONLY": 0,
    "NOT_EXECUTED": 0
  },
  "B_disjoint_reference_fitted": {
    "DETECTED": 1,
    "MISSED": 0,
    "ALWAYS_FAIL": 6,
    "REFERENCE_FALSE_POSITIVE_ONLY": 0,
    "NOT_EXECUTED": 0
  },
  "C_information_enriched": {
    "DETECTED": 7,
    "MISSED": 0,
    "ALWAYS_FAIL": 0,
    "REFERENCE_FALSE_POSITIVE_ONLY": 0,
    "NOT_EXECUTED": 0
  }
}
```

RD3 excluded: NA - single 4-element configuration constant, not a multi-record data artifact; its population analogue is RD4
