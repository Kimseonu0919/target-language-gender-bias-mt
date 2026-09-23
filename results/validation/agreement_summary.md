# Ensemble validation summary

- providers with results: anthropic, openai, gemini
- units: 11915, fully voted: 11915

## Row-weighted agreement (rule label vs ensemble majority)

| system | condition | language | rows | decided % | fine agree % | M/F/N agree % |
|---|---|---|---:|---:|---:|---:|
| deepl | clothing_color | en | 33600 | 100.00 | 100.00 | 100.00 |
| deepl | clothing_color | ja | 33600 | 100.00 | 100.00 | 100.00 |
| deepl | clothing_color | kr | 33600 | 100.00 | 100.00 | 100.00 |
| deepl | clothing_color | zh | 33600 | 100.00 | 100.00 | 100.00 |
| deepl | clothing_only | en | 1600 | 100.00 | 100.00 | 100.00 |
| deepl | clothing_only | ja | 1600 | 100.00 | 100.00 | 100.00 |
| deepl | clothing_only | kr | 1600 | 100.00 | 100.00 | 100.00 |
| deepl | clothing_only | zh | 1600 | 100.00 | 100.00 | 100.00 |
| google | clothing_color | en | 33600 | 100.00 | 99.95 | 100.00 |
| google | clothing_color | ja | 33600 | 100.00 | 100.00 | 100.00 |
| google | clothing_color | kr | 33600 | 100.00 | 100.00 | 100.00 |
| google | clothing_color | zh | 33600 | 100.00 | 99.98 | 100.00 |
| google | clothing_only | en | 1600 | 100.00 | 100.00 | 100.00 |
| google | clothing_only | ja | 1600 | 100.00 | 100.00 | 100.00 |
| google | clothing_only | kr | 1600 | 100.00 | 100.00 | 100.00 |
| google | clothing_only | zh | 1600 | 100.00 | 100.00 | 100.00 |

## Overall (row-weighted)

- rule vs majority, fine labels: 99.991% (kappa = 0.9998)
- rule vs majority, M/F/N: 100.000%
- inter-model Fleiss' kappa: 0.9996

## Pairwise model agreement (row-weighted)

| pair | agree % | kappa |
|---|---:|---:|
| anthropic-gemini | 99.998 | 1.0000 |
| anthropic-openai | 99.972 | 0.9995 |
| gemini-openai | 99.973 | 0.9995 |
