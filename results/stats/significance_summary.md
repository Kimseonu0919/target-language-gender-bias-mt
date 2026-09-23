# Statistical significance summary

Bootstrap: B=10,000, multinomial resampling, percentile 95% CI, seed=20260829.
Chi-square on M/F/N counts (scipy chi2_contingency; categories empty in every
group are dropped, and the Yates correction applies to 2x2 tables); Cramer's V
effect size; Holm correction per family. Sample sizes reach 134,400, so
p-values are close to zero for most tests and Cramer's V is the quantity to
interpret.

## NGB / GPS / GAR with 95% CI

| system | condition | lang | n | NGB [95% CI] | GPS [95% CI] | GAR [95% CI] |
|---|---|---|---:|---|---|---|
| deepl | clothing_only | kr | 1,600 | -0.258 [-0.305, -0.210] | -0.258 [-0.305, -0.210] | 1.000 [1.000, 1.000] |
| deepl | clothing_only | en | 1,600 | -0.092 [-0.141, -0.044] | -0.092 [-0.141, -0.044] | 1.000 [1.000, 1.000] |
| deepl | clothing_only | ja | 1,600 | -0.093 [-0.137, -0.049] | -0.113 [-0.167, -0.060] | 0.821 [0.802, 0.839] |
| deepl | clothing_only | zh | 1,600 | -0.256 [-0.305, -0.209] | -0.256 [-0.305, -0.209] | 1.000 [1.000, 1.000] |
| deepl | clothing_color | kr | 33,600 | +0.048 [+0.038, +0.059] | +0.048 [+0.038, +0.059] | 1.000 [1.000, 1.000] |
| deepl | clothing_color | en | 33,600 | +0.270 [+0.260, +0.280] | +0.270 [+0.260, +0.280] | 1.000 [1.000, 1.000] |
| deepl | clothing_color | ja | 33,600 | +0.151 [+0.141, +0.161] | +0.174 [+0.163, +0.186] | 0.865 [0.862, 0.869] |
| deepl | clothing_color | zh | 33,600 | +0.048 [+0.037, +0.058] | +0.048 [+0.037, +0.058] | 1.000 [1.000, 1.000] |
| google | clothing_only | kr | 1,600 | -0.315 [-0.361, -0.269] | -0.315 [-0.361, -0.269] | 1.000 [1.000, 1.000] |
| google | clothing_only | en | 1,600 | -0.316 [-0.364, -0.270] | -0.316 [-0.364, -0.270] | 1.000 [1.000, 1.000] |
| google | clothing_only | ja | 1,600 | -0.318 [-0.362, -0.271] | -0.318 [-0.362, -0.271] | 1.000 [1.000, 1.000] |
| google | clothing_only | zh | 1,600 | -0.314 [-0.361, -0.268] | -0.314 [-0.361, -0.268] | 1.000 [1.000, 1.000] |
| google | clothing_color | kr | 33,600 | -0.121 [-0.132, -0.111] | -0.121 [-0.132, -0.111] | 1.000 [1.000, 1.000] |
| google | clothing_color | en | 33,600 | -0.122 [-0.133, -0.111] | -0.122 [-0.133, -0.111] | 0.999 [0.999, 1.000] |
| google | clothing_color | ja | 33,600 | -0.121 [-0.132, -0.111] | -0.121 [-0.132, -0.111] | 1.000 [1.000, 1.000] |
| google | clothing_color | zh | 33,600 | -0.122 [-0.133, -0.111] | -0.122 [-0.133, -0.111] | 1.000 [0.999, 1.000] |

## Chi-square tests

| family | test | n | dof | chi2 | p (Holm) | Cramer's V |
|---|---|---:|---:|---:|---|---:|
| A_language | deepl/clothing_only: MFN x language | 6,400 | 6 | 940.18 | <0.001 | 0.2710 |
| A_language | deepl/clothing_color: MFN x language | 134,400 | 6 | 15,251.37 | <0.001 | 0.2382 |
| A_language | google/clothing_only: MFN x language | 6,400 | 3 | 0.01 | 1.000 | 0.0015 |
| A_language | google/clothing_color: MFN x language | 134,400 | 6 | 27.74 | <0.001 | 0.0102 |
| B_condition | deepl/kr: MFN x condition | 35,200 | 2 | 143.32 | <0.001 | 0.0638 |
| B_condition | deepl/en: MFN x condition | 35,200 | 1 | 213.63 | <0.001 | 0.0779 |
| B_condition | deepl/ja: MFN x condition | 35,200 | 2 | 127.69 | <0.001 | 0.0602 |
| B_condition | deepl/zh: MFN x condition | 35,200 | 1 | 140.86 | <0.001 | 0.0633 |
| B_condition | google/kr: MFN x condition | 35,200 | 2 | 58.35 | <0.001 | 0.0407 |
| B_condition | google/en: MFN x condition | 35,200 | 2 | 59.43 | <0.001 | 0.0411 |
| B_condition | google/ja: MFN x condition | 35,200 | 1 | 59.47 | <0.001 | 0.0411 |
| B_condition | google/zh: MFN x condition | 35,200 | 2 | 57.59 | <0.001 | 0.0404 |
| C_system | clothing_only/kr: MFN x system | 3,200 | 1 | 2.76 | 0.194 | 0.0294 |
| C_system | clothing_only/en: MFN x system | 3,200 | 1 | 41.33 | <0.001 | 0.1136 |
| C_system | clothing_only/ja: MFN x system | 3,200 | 2 | 346.60 | <0.001 | 0.3291 |
| C_system | clothing_only/zh: MFN x system | 3,200 | 1 | 2.76 | 0.194 | 0.0293 |
| C_system | clothing_color/kr: MFN x system | 67,200 | 2 | 485.82 | <0.001 | 0.0850 |
| C_system | clothing_color/en: MFN x system | 67,200 | 2 | 2,612.06 | <0.001 | 0.1972 |
| C_system | clothing_color/ja: MFN x system | 67,200 | 2 | 6,206.31 | <0.001 | 0.3039 |
| C_system | clothing_color/zh: MFN x system | 67,200 | 2 | 495.98 | <0.001 | 0.0859 |
