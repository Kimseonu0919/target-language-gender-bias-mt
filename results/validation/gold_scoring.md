# Gold-standard scoring (800 stratified rows, blind human labels)

- labeled rows used: 800 / 800

| judge | scope | n | fine acc % | M/F/N acc % | Cohen's kappa |
|---|---|---:|---:|---:|---:|
| rule | overall | 800 | 100.00 | 100.00 | 1.0000 |
| rule | en | 200 | 100.00 | 100.00 | 1.0000 |
| rule | ja | 200 | 100.00 | 100.00 | 1.0000 |
| rule | kr | 200 | 100.00 | 100.00 | 1.0000 |
| rule | zh | 200 | 100.00 | 100.00 | 1.0000 |
| ensemble | overall | 800 | 100.00 | 100.00 | 1.0000 |
| ensemble | en | 200 | 100.00 | 100.00 | 1.0000 |
| ensemble | ja | 200 | 100.00 | 100.00 | 1.0000 |
| ensemble | kr | 200 | 100.00 | 100.00 | 1.0000 |
| ensemble | zh | 200 | 100.00 | 100.00 | 1.0000 |

Human label distribution: male=466, female=327, omitted=7

Mismatch rows (human vs rule or ensemble): 0 -> gold_mismatches.csv
