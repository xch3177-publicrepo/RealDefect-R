# GX-BASE-20260914 results

Great Expectations 1.6.3, pandas 2.1.4, NumPy 1.26.4, CPython 3.12.3.

| Case | Arm A declared schema | Arm B disjoint-reference fitted | Arm C information-enriched |
|---|---|---|---|
| RD1 | NOT_EXECUTED | NOT_EXECUTED | NOT_EXECUTED |
| RD2 | NOT_EXECUTED | NOT_EXECUTED | NOT_EXECUTED |
| RD4 | NOT_EXECUTED | NOT_EXECUTED | NOT_EXECUTED |
| RD5 | NOT_EXECUTED | NOT_EXECUTED | NOT_EXECUTED |
| RD6 | NOT_EXECUTED | NOT_EXECUTED | NOT_EXECUTED |
| RD7 | NOT_EXECUTED | NOT_EXECUTED | NOT_EXECUTED |
| RD8 | NOT_EXECUTED | NOT_EXECUTED | NOT_EXECUTED |

Tally:
```
{
  "A_declared_schema": {
    "DETECTED": 0,
    "MISSED": 0,
    "ALWAYS_FAIL": 0,
    "REFERENCE_FALSE_POSITIVE_ONLY": 0,
    "NOT_EXECUTED": 7
  },
  "B_disjoint_reference_fitted": {
    "DETECTED": 0,
    "MISSED": 0,
    "ALWAYS_FAIL": 0,
    "REFERENCE_FALSE_POSITIVE_ONLY": 0,
    "NOT_EXECUTED": 7
  },
  "C_information_enriched": {
    "DETECTED": 0,
    "MISSED": 0,
    "ALWAYS_FAIL": 0,
    "REFERENCE_FALSE_POSITIVE_ONLY": 0,
    "NOT_EXECUTED": 7
  }
}
```

RD3 excluded: NA - single 4-element configuration constant, not a multi-record data artifact; its population analogue is RD4
