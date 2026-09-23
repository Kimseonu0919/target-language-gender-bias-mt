"""Draw the stratified gold sample for independent human labeling.

50 rows from each of the 16 (system x condition x language) files = 800 rows,
with a fixed seed for reproducibility. The labeling sheet hides all machine
labels; the answer key (with rule labels and unit ids) is written separately
so labeling stays blind.

Usage: python -m src.ensemble.sample_gold

Outputs (results/validation/):
  gold_sample.xlsx      labeling sheet (dropdown per row, machine labels hidden)
  gold_sample.csv       same rows as CSV (utf-8-sig)
  gold_answer_key.csv   gold_id -> unit_id, rule label (do not open while labeling)
"""

import csv
import json
import random
import re
import zipfile
from datetime import datetime, timezone

from openpyxl import Workbook
from openpyxl.worksheet.datavalidation import DataValidation

from ..detector.pronoun_gender import detect_pronoun_gender_detailed, split_sentences
from .build_units import unit_id_for
from .config import OUT_DIR, translation_files

SEED = 20260829
PER_FILE = 50
LABEL_CHOICES = ["male", "female", "neutral", "one_of_them", "omitted"]


def rewrite_zip(path, stamp):
    """Pin the timestamps openpyxl sets to the current time (zip entries and
    the modified date in core.xml), so the file is the same on every run."""
    stamp_text = stamp.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    with zipfile.ZipFile(path) as src:
        entries = [(info.filename, src.read(info.filename)) for info in src.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as dst:
        for name, data in entries:
            if name == "docProps/core.xml":
                data = re.sub(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)",
                              rb"\g<1>" + stamp_text + rb"\g<2>", data)
            info = zipfile.ZipInfo(name, date_time=stamp.timetuple()[:6])
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    sample = []
    for entry in translation_files():
        with open(entry["path"], encoding="utf-8") as fh:
            rows = json.load(fh)
        for row_idx in sorted(rng.sample(range(len(rows)), PER_FILE)):
            row = rows[row_idx]
            parts = split_sentences(row["sentence"])
            text = parts[1] if len(parts) >= 2 else row["sentence"]
            detail = detect_pronoun_gender_detailed(
                {"sentence": row["sentence"], "language": row["language"]}
            )
            sample.append(
                {
                    "file": entry["file"],
                    "row_idx": row_idx,
                    "system": entry["system"],
                    "condition": entry["condition"],
                    "language": entry["language"],
                    "sentence": row["sentence"],
                    "second_sentence": text,
                    "unit_id": unit_id_for(row["language"], text),
                    "rule_label": detail["gender"],
                }
            )

    rng.shuffle(sample)  # so the labeler cannot infer system/condition blocks
    for gold_id, item in enumerate(sample, 1):
        item["gold_id"] = gold_id

    # Labeling sheet (xlsx with a dropdown; no machine labels visible)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "gold_labeling"
    headers = ["gold_id", "language", "sentence", "second_sentence", "human_label", "note"]
    sheet.append(headers)
    validation = DataValidation(
        type="list", formula1='"' + ",".join(LABEL_CHOICES) + '"', allow_blank=True
    )
    sheet.add_data_validation(validation)
    for item in sample:
        sheet.append(
            [item["gold_id"], item["language"], item["sentence"],
             item["second_sentence"], "", ""]
        )
    validation.add(f"E2:E{len(sample) + 1}")
    widths = {"A": 8, "B": 10, "C": 70, "D": 45, "E": 14, "F": 20}
    for col, width in widths.items():
        sheet.column_dimensions[col].width = width
    sheet.freeze_panes = "A2"
    # Fixed document timestamps so the file is byte-for-byte reproducible.
    stamp = datetime(2026, 8, 29, 10, 9, 36, tzinfo=timezone.utc)
    workbook.properties.created = stamp
    workbook.properties.modified = stamp
    workbook.save(OUT_DIR / "gold_sample.xlsx")
    rewrite_zip(OUT_DIR / "gold_sample.xlsx", stamp)

    with open(OUT_DIR / "gold_sample.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        for item in sample:
            writer.writerow(
                {"gold_id": item["gold_id"], "language": item["language"],
                 "sentence": item["sentence"],
                 "second_sentence": item["second_sentence"],
                 "human_label": "", "note": ""}
            )

    key_fields = ["gold_id", "file", "row_idx", "system", "condition",
                  "language", "unit_id", "rule_label"]
    with open(OUT_DIR / "gold_answer_key.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=key_fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for item in sorted(sample, key=lambda x: x["gold_id"]):
            writer.writerow(item)

    print(f"gold sample: {len(sample)} rows (seed={SEED})")
    print(f"-> {OUT_DIR / 'gold_sample.xlsx'} (labeling sheet)")
    print(f"-> {OUT_DIR / 'gold_answer_key.csv'} (keep closed while labeling)")


if __name__ == "__main__":
    main()
