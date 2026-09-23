# Target-Language Effects on Gender Bias in Machine Translation

Code and data for the paper *"Target-Language Effects on Gender Bias in
Machine Translation: A Cross-Lingual Study of Gender-Neutral Source
Languages"* by Seung Yeon Kim and Kong Joo Lee (Chungnam National
University), The Transactions of the Korea Information Processing Society,
2026 (in press).

Turkish source sentences about a person wearing a piece of clothing are
translated into Korean, English, Japanese and Chinese with DeepL and Google
Translate. A rule-based detector labels each output male, female or neutral,
and the effect of the target language on gender bias is measured with GPS,
GAR, NGB (= GPS x GAR), UCA and JSD, with bootstrap confidence intervals and
chi-square tests (Cramer's V). All 281,600 labels were checked against an
ensemble of three commercial LLMs, and 800 of them against a blind human
gold set.

## Repository layout

| path | content |
|---|---|
| `data/fashion_data.xlsx` | stimulus workbook: clothing, color and context lists |
| `translations/` | 16 files `{deepl,google}_{default,color}_tr_{kr,en,ja,zh}.json`, 281,600 sentences |
| `src/detector/` | rule-based gender detector (lexicon and matching rules) |
| `src/analysis/` | metrics, statistics, paper tables and figures |
| `src/ensemble/` | LLM ensemble validation and gold-set scoring |
| `src/preprocessing/`, `src/translate/` | source sentence generation and translation collection |
| `results/data_summary.json` | per-file label counts and metrics; Tables 2 to 4 and the figures are derived from it |
| `results/stats/` | bootstrap CIs, chi-square tests with Cramer's V (Holm-corrected) |
| `results/tables/` | Tables 2 to 6 of the paper as Markdown and CSV |
| `results/validation/` | ensemble votes (3 models x 11,915 units), agreement statistics, adjudicated disagreements, human gold set and scoring |
| `figs/` | Figs. 2 to 4 of the paper |

## Data

Two conditions share 100 contexts (25 actions x 4 subject pairs):

- Clothing Only: 16 clothing items x 100 contexts = 1,600 sentences
- Clothing + Color: 16 items x 21 color slots x 100 contexts = 33,600 sentences

Each condition was translated by two systems into four languages, so the
corpus has (1,600 + 33,600) x 8 = 281,600 translations. Every translation
file is a JSON list of `{"sentence": ..., "language": ...}` in a fixed order,
and the analysis recovers the design cell of row `i` from its position:

- Clothing Only: context `i // 16`, item `i % 16`
- Clothing + Color: context `i // 336`, item `(i % 336) // 21`, color `i % 21`

The item and color-slot orders are `CLOTHING_ITEMS` and `COLOR_NAMES` in
`src/analysis/calculate_metrics.py`. The scripts refuse a file whose row
count differs from the design. Korean uses the language code `kr`.

### Stimulus workbook

`data/fashion_data.xlsx` is the stimulus set of Park, Cho and Lee (2026),
the earlier study of the same group (see "Licenses"). This study uses the
Turkish columns only: `CONTEXT-터키어` columns C and D (contexts),
`DATA-입고있다` column C (16 clothing sentences) and `DATA-색깔-입고있다`
column D (336 clothing + color sentences). The workbook also holds the
Estonian, Finnish and Hungarian columns of the earlier study and a working
sheet (`색상-12-23`) with the color synonym groups behind the 21 slots.

### Notes on the stimuli

These properties come with the shared stimulus set and were kept for
comparability with the earlier study.

- The 21 "colors" are slots. The Turkish color term of a slot can differ
  between clothing items (slot 4, `mint`, appears as *lime rengi*, *açık
  yeşil*, *nane yeşili*, *limon yeşili* or *mint rengi* depending on the
  item). Table 4 and the color axis of the heatmaps group by slot.
- Five items use a different Turkish noun in the two conditions: knitwear
  (*triko* / *kazak*), coat (*palto* / *kaban*), windbreaker (*mont* /
  *ceket*), sweater (*kazak* / *süveter*) and hoodie (*kapüşonlu* /
  *kapüşonlu tişört*). For these items the NGB shift in Table 3 reflects the
  noun change as well as the added color.
- Windbreaker and jacket both use *ceket* in the color condition, and 15 of
  the 21 windbreaker color sentences are identical to the jacket sentences
  (1,500 of the 33,600 color rows repeat another row).
- Clothing Only sentences have a comma after the pronoun (*O, kot pantolon
  giyiyor.*); color sentences do not (*O altın rengi kot pantolon giyiyor.*).

### Labels

The detector returns `male`, `female`, `neutral` (paired forms such as
he/she, 그/그녀, 彼/彼女, 他/她, and plural or non-human pronouns),
`none_one` (one-person phrases such as "one of them", 한 명, 一人は, 一个人),
`omitted` (no marker, subject omission) and `exception` (input that does not
split into two sentences; it does not occur in this corpus). The metrics
pool everything except `male` and `female` into N. The ensemble prompt uses
the label `one_of_them` for `none_one`.

## Detector and label validation (paper Sec. 3.3)

The detector reads only the second sentence of a translation. One-person
phrases are checked first, then paired forms, then the leftmost marker in
the sentence wins, gendered or neutral. Single-character markers such as
그, 彼, 他 and 她 use context patterns so they do not match inside 그녀, 彼女
or 他们. The full lexicon is in `src/detector/pronoun_gender.py`.

1. Ensemble check of every label: the 11,915 unique (language, second
   sentence) pairs, which cover all 281,600 rows, were classified by Claude
   Haiku 4.5, GPT-5 mini and Gemini 3.7 Flash (accessed August 2026).
   Majority vote vs. rule labels: 100.000% agreement on M/F/N and 99.991% on
   the fine-grained labels (Cohen's kappa 0.9998); inter-model Fleiss' kappa
   0.9996. Raw votes: `results/validation/raw/`; model IDs and request
   settings: `results/validation/PROVENANCE.md`.
2. Adjudication: the 22 disagreeing units (24 rows, 0.009% of all rows) were
   adjudicated by hand; all are boundary cases between the neutral
   subcategories (`results/validation/adjudication.csv`).
3. Human gold set: 800 sentences (200 per language, stratified over system
   and condition) were labeled with the machine labels hidden. Detector
   accuracy 100% (800/800, Cohen's kappa 1.0). The labels are in
   `results/validation/labeled_human_labels.csv`, the scoring in
   `results/validation/gold_scoring.md`; `python -m src.ensemble.sample_gold`
   regenerates the sampling sheet and the answer key.

## Reproduce

Python 3.11 or later is required. The committed outputs were produced with
Python 3.13.11 and the package versions pinned in `requirements.txt`
(numpy 2.3.5, scipy 1.17.1, pandas 3.0.5, openpyxl 3.1.5, matplotlib 3.11.1).
The bootstrap draws from `numpy.random.default_rng(20260829)`, so other numpy
versions may give slightly different confidence-interval bounds. Run the
commands from the repository root.

```bash
pip install -r requirements.txt

python -m src.analysis.compute_summary        # 1. results/data_summary.json (rule-based labels)
python -m src.ensemble.build_units            # 2. unique units and sanity check
python -m src.analysis.significance           # 3. bootstrap CIs, chi-square, Cramer's V
python -m src.analysis.make_integrated_table  # 4. Table 2
python -m src.analysis.make_paper_tables      # 5. Tables 3 to 6
python -m src.analysis.make_figures           # 6. Figs. 2 to 4
python -m src.ensemble.aggregate              # 7. ensemble votes vs. rule labels
python -m src.ensemble.score_gold             # 8. human gold set scoring
```

Every output is committed, so `git status` shows no change after these
steps. `python -m src.analysis.calculate_metrics --all translations` prints
the per-file metrics and the JSD pairs to the console.

### Re-running the LLM ensemble (optional)

The shipped vote logs mark every unit as done, so a new run has to start
from an empty `results/validation/raw/`:

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY
mv results/validation/raw results/validation/raw_paper
python -m src.ensemble.preflight
python -m src.ensemble.run_ensemble        # about USD 2-3 for the three providers
python -m src.ensemble.aggregate
```

Model outputs change over time, so a new run will not reproduce the vote
logs exactly.

### Collecting the translations again (optional)

```bash
python -m src.preprocessing.preprocessing          # data/generated/{default,color}_tr.json
python -m src.translate.translator data/generated/default_tr.json
python -m src.translate.translator data/generated/color_tr.json
```

The translator needs `DEEPL_API_KEY` and `GOOGLE_API_KEY` in `.env` and
writes to `data/generated/translations/`; copy the files into
`translations/` to analyze them. The shipped translations will not be
reproduced exactly by the current services, which change over time.

## Figures

| file | paper |
|---|---|
| `figs/fig2_jsd_avg.png` | Fig. 2, average JSD per (system x language) |
| `figs/fig3_ngb_heatmap_deepl_kr.png` | Fig. 3, NGB heatmap, DeepL-KR |
| `figs/fig4_ngb_heatmap_google_kr.png` | Fig. 4, NGB heatmap, Google-KR |

## Licenses

- Code: MIT (`LICENSE`).
- Translations, results and figures (`translations/`, `results/`, `figs/`):
  CC BY 4.0 (`DATA_LICENSE`).
- The stimulus workbook `data/fashion_data.xlsx` is reproduced from the
  earlier study of the same group, Y.-H. Park, M.-S. Cho, and K. J. Lee,
  "Exploring Gender Bias in Fashion Descriptions in Machine Translation to
  Korean," The Transactions of the Korea Information Processing Society,
  Vol.15, No.2, pp.102-112, 2026, https://doi.org/10.3745/TKIPS.2026.15.2.102.
  Its copyright remains with its authors; cite that work when you reuse the
  stimuli.

## Citation

S. Y. Kim and K. J. Lee, "Target-Language Effects on Gender Bias in Machine
Translation: A Cross-Lingual Study of Gender-Neutral Source Languages,"
The Transactions of the Korea Information Processing Society, 2026 (in press).
Korean title: 기계 번역의 성별 편향에 대한 타깃 언어 효과 분석: 성 중립 언어의
교차 언어 연구.

```bibtex
@article{kim2026target,
  author  = {Kim, Seung Yeon and Lee, Kong Joo},
  title   = {Target-Language Effects on Gender Bias in Machine Translation:
             A Cross-Lingual Study of Gender-Neutral Source Languages},
  journal = {The Transactions of the Korea Information Processing Society},
  year    = {2026},
  note    = {in press}
}
```

A `CITATION.cff` file is included for GitHub's citation feature.
