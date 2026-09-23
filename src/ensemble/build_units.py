"""Extract classification units from the 16 translation result files.

The rule-based detector only ever looks at the second sentence, so two rows
with the same (language, second sentence) necessarily receive the same label.
Classifying each unique pair once is therefore exactly equivalent to
classifying all 281,600 rows, at ~1/24 of the cost.

Outputs (results/validation/):
  units.jsonl    one line per unique (language, second sentence):
                 {unit_id, language, text, rule_label, rule_word, row_count}
  rows_map.json  {file_name: [unit_id per row, in row order]}
  sanity_check.md  per-file rule-label frequencies vs results/data_summary.json
"""

import hashlib
import json
from collections import Counter

from ..detector.pronoun_gender import detect_pronoun_gender_detailed, split_sentences
from .config import OUT_DIR, ROOT, ROWS_MAP_PATH, UNITS_PATH, translation_files

DATA_SUMMARY_PATH = ROOT / "results" / "data_summary.json"
FREQ_KEYS = ["male", "female", "neutral", "none_one", "omitted", "exception"]


def unit_id_for(language: str, text: str) -> str:
    return hashlib.sha1(f"{language}\x00{text}".encode("utf-8")).hexdigest()[:16]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    units: dict[str, dict] = {}
    rows_map: dict[str, list[str]] = {}
    per_file_freq: dict[str, Counter] = {}

    for entry in translation_files():
        with open(entry["path"], encoding="utf-8") as fh:
            rows = json.load(fh)
        ids = []
        freq = Counter()
        for row in rows:
            language = row["language"]
            parts = split_sentences(row["sentence"])
            text = parts[1] if len(parts) >= 2 else row["sentence"]
            uid = unit_id_for(language, text)
            if uid not in units:
                detail = detect_pronoun_gender_detailed(
                    {"sentence": row["sentence"], "language": language}
                )
                units[uid] = {
                    "unit_id": uid,
                    "language": language,
                    "text": text,
                    "rule_label": detail["gender"],
                    "rule_word": detail["matched_word"],
                    "row_count": 0,
                }
            units[uid]["row_count"] += 1
            freq[units[uid]["rule_label"]] += 1
            ids.append(uid)
        rows_map[entry["file"]] = ids
        per_file_freq[entry["file"]] = freq
        print(f"{entry['file']}: {len(rows)} rows")

    with open(UNITS_PATH, "w", encoding="utf-8", newline="\n") as fh:
        for unit in units.values():
            fh.write(json.dumps(unit, ensure_ascii=False) + "\n")
    with open(ROWS_MAP_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows_map, fh)

    total_rows = sum(len(v) for v in rows_map.values())
    print(f"\nunits: {len(units)} unique / {total_rows} rows")

    # The per-file label frequencies must equal those recorded in
    # results/data_summary.json.
    lines = [
        "# Sanity check: rule labels vs results/data_summary.json",
        "",
        "| file | label | build_units | data_summary | match |",
        "|---|---|---:|---:|---|",
    ]
    mismatches = 0
    if DATA_SUMMARY_PATH.exists():
        with open(DATA_SUMMARY_PATH, encoding="utf-8") as fh:
            summary = json.load(fh)
        for fname, freq in per_file_freq.items():
            expected = summary.get(fname, {}).get("freq", {})
            for key in FREQ_KEYS:
                n_here, n_summary = freq.get(key, 0), expected.get(key, 0)
                if n_here or n_summary:
                    ok = "OK" if n_here == n_summary else "**MISMATCH**"
                    if n_here != n_summary:
                        mismatches += 1
                    lines.append(f"| {fname} | {key} | {n_here} | {n_summary} | {ok} |")
    lines.append("")
    lines.append(f"Mismatches: {mismatches}")
    (OUT_DIR / "sanity_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8",
                                             newline="\n")
    print(f"sanity check: {mismatches} mismatches -> {OUT_DIR / 'sanity_check.md'}")


if __name__ == "__main__":
    main()
