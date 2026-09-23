# Provenance of the label validation

## Ensemble run

- Run date: 2026-08-29 (see the `ts` fields in `raw/*.jsonl`).
- Ensemble members (request model IDs, as recorded in the raw vote logs):
  - Anthropic: `claude-haiku-4-5`
  - OpenAI: `gpt-5-mini`
  - Google: `gemini-flash-latest` (a rolling alias)
- Alias resolution: on the run date, a single 8-token `generateContent` probe
  against `gemini-flash-latest` returned `modelVersion: gemini-3.7-flash` in
  the API response. The raw vote logs store the request alias in their
  `model` field; the resolved version above is what served those requests at
  run time.
- The paper therefore cites the members as Claude Haiku 4.5, GPT-5 mini, and
  Gemini 3.7 Flash (accessed August 2026).

## Request settings

- Prompt: `src/ensemble/prompts.py`. Each request classifies 20 sentences of
  one language; labels are male, female, neutral, one_of_them (the detector's
  none_one) and omitted.
- 6 parallel requests per provider. Each request is attempted up to 5 times
  on HTTP 408, 409, 429, 5xx and network errors (the Anthropic SDK applies
  the same limit itself). A chunk that fails twice or cannot be parsed is
  split in half and each half is retried.
- Anthropic: Messages API, `max_tokens` 2000, no temperature parameter (API
  default).
- OpenAI: Chat Completions, `max_completion_tokens` 6000,
  `reasoning_effort` "low".
- Google: `generateContent`, `temperature` 0, `maxOutputTokens` 6000,
  `thinkingBudget` 0.
- If an endpoint rejects `reasoning_effort` or `thinkingConfig`, the code
  drops that parameter for the rest of the run. The raw logs do not record
  whether this happened during the paper run.

## Human gold set and adjudication

- `gold_sample.xlsx` / `gold_sample.csv`: 800 rows, 50 from each of the 16
  translation files (200 per language), drawn by `src/ensemble/sample_gold.py`
  with a fixed seed on 2026-08-29. The labeling sheet contains no machine
  labels; the answer key (`gold_answer_key.csv`) is a separate file.
- `labeled_human_labels.csv`: the 800 rows with the labels given by the
  authors; scored against the rule labels and the ensemble majority by
  `src/ensemble/score_gold.py` (`gold_scoring.md`, `gold_mismatches.csv`).
- `adjudication.csv`: the 22 units (24 rows) on which the ensemble majority
  and the rule label disagree, adjudicated by the authors on 2026-08-29 with
  a short reason per unit. `src/ensemble/aggregate.py` carries these labels
  into the `human_label` column of `disagreements.csv`.
