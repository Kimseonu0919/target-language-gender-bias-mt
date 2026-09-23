"""Score the rule-based detector and the LLM ensemble against the human
gold labels.

Usage: python -m src.ensemble.score_gold

Inputs (results/validation/):
  labeled_human_labels.csv  gold_sample.csv with the human_label column filled
  gold_answer_key.csv       gold_id -> unit_id, rule label, strata
  ensemble_labels.jsonl     unit-level ensemble majority
Outputs:
  gold_scoring.md           accuracy + Cohen's kappa tables (reported in Sec. 3.3 of the paper)
  gold_mismatches.csv       rows where human differs from rule or ensemble
"""

import csv
import json
from collections import Counter

from .aggregate import COARSE, cohen_kappa
from .config import LABEL_TO_RULE, OUT_DIR

LABELED_PATH = OUT_DIR / "labeled_human_labels.csv"


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    labeled = {int(r["gold_id"]): r for r in read_csv(LABELED_PATH)}
    key = {int(r["gold_id"]): r for r in read_csv(OUT_DIR / "gold_answer_key.csv")}
    majority = {}
    with open(OUT_DIR / "ensemble_labels.jsonl", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            majority[record["unit_id"]] = record["majority"]

    rows, invalid, undecided = [], [], []
    for gold_id, entry in sorted(labeled.items()):
        human_raw = entry["human_label"].strip().lower()
        if human_raw not in LABEL_TO_RULE:
            invalid.append((gold_id, entry["human_label"]))
            continue
        meta = key[gold_id]
        # missing unit, tie, or too few votes: scored as a mismatch below
        ensemble = majority.get(meta["unit_id"])
        if ensemble not in COARSE:
            ensemble = "undecided"
            undecided.append(gold_id)
        rows.append(
            {
                "gold_id": gold_id,
                "language": meta["language"],
                "system": meta["system"],
                "condition": meta["condition"],
                "text": entry["second_sentence"],
                "human": LABEL_TO_RULE[human_raw],
                "rule": meta["rule_label"],
                "ensemble": ensemble,
            }
        )
    if invalid:
        print(f"WARNING: {len(invalid)} rows with empty/unknown labels "
              f"(excluded): {invalid[:5]}")
    if undecided:
        print(f"WARNING: ensemble undecided on {len(undecided)} rows "
              f"(scored as mismatches): {undecided[:5]}")

    def block(subset, judge):
        pairs = [(r["human"], r[judge], 1) for r in subset]
        fine = sum(1 for r in subset if r["human"] == r[judge]) / len(subset)
        coarse = sum(
            1 for r in subset if COARSE[r["human"]] == COARSE.get(r[judge])
        ) / len(subset)
        return fine, coarse, cohen_kappa(pairs)

    lines = [
        f"# Gold-standard scoring ({len(labeled)} stratified rows, blind human labels)",
        "",
        f"- labeled rows used: {len(rows)} / {len(labeled)}",
    ]
    if undecided:
        lines.append(f"- ensemble undecided (scored as mismatch): {len(undecided)}")
    lines += [
        "",
        "| judge | scope | n | fine acc % | M/F/N acc % | Cohen's kappa |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for judge in ("rule", "ensemble"):
        fine, coarse, kappa = block(rows, judge)
        lines.append(f"| {judge} | overall | {len(rows)} "
                     f"| {100 * fine:.2f} | {100 * coarse:.2f} | {kappa:.4f} |")
        for lang in sorted({r["language"] for r in rows}):
            subset = [r for r in rows if r["language"] == lang]
            fine, coarse, kappa = block(subset, judge)
            lines.append(f"| {judge} | {lang} | {len(subset)} "
                         f"| {100 * fine:.2f} | {100 * coarse:.2f} | {kappa:.4f} |")

    label_dist = Counter(r["human"] for r in rows)
    lines += ["", "Human label distribution: "
              + ", ".join(f"{k}={v}" for k, v in label_dist.most_common())]

    mismatches = [r for r in rows
                  if r["human"] != r["rule"] or r["human"] != r["ensemble"]]
    with open(OUT_DIR / "gold_mismatches.csv", "w", encoding="utf-8-sig",
              newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0]) if rows else [], lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(mismatches)
    lines += ["", f"Mismatch rows (human vs rule or ensemble): {len(mismatches)} "
              "-> gold_mismatches.csv"]

    (OUT_DIR / "gold_scoring.md").write_text("\n".join(lines) + "\n", encoding="utf-8",
                                             newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
