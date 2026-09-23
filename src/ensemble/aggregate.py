"""Aggregate ensemble votes, compare against rule-based labels, and emit
paper-ready validation statistics.

Usage: python -m src.ensemble.aggregate

Inputs : units.jsonl, rows_map.json, raw/<provider>.jsonl
Outputs (results/validation/):
  ensemble_labels.jsonl  per unit: votes, majority, rule label, agreement
  disagreements.csv      units where majority != rule label (largest impact first;
                         human_label carried over from adjudication.csv if present)
  agreement_summary.md   coverage, agreement tables, kappas - reported in Sec. 3.3 of the paper
"""

import csv
import json
from collections import Counter, defaultdict

from .config import (
    LABEL_TO_RULE,
    OUT_DIR,
    PROVIDERS,
    RAW_DIR,
    ROWS_MAP_PATH,
    UNITS_PATH,
    translation_files,
)

# The paper's coarse categories: N pools every non-gendered outcome.
COARSE = {
    "male": "M",
    "female": "F",
    "neutral": "N",
    "none_one": "N",
    "omitted": "N",
    "exception": "N",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def load_votes() -> dict[str, dict[str, str]]:
    """unit_id -> {provider: rule-space label} (last vote wins per provider)."""
    votes: dict[str, dict[str, str]] = defaultdict(dict)
    for provider in PROVIDERS:
        path = RAW_DIR / f"{provider}.jsonl"
        if not path.exists():
            continue
        for record in load_jsonl(path):
            votes[record["unit_id"]][provider] = LABEL_TO_RULE[record["label"]]
    return votes


def majority_label(vote_map: dict[str, str]) -> str:
    if len(vote_map) < 2:
        return "insufficient_votes"
    counts = Counter(vote_map.values())
    label, top = counts.most_common(1)[0]
    return label if top >= 2 else "no_majority"


def cohen_kappa(pairs: list[tuple[str, str, int]]) -> float:
    """Row-count-weighted Cohen's kappa over (label_a, label_b, weight) pairs."""
    total = sum(w for _, _, w in pairs)
    if not total:
        return float("nan")
    observed = sum(w for a, b, w in pairs if a == b) / total
    freq_a, freq_b = Counter(), Counter()
    for a, b, w in pairs:
        freq_a[a] += w
        freq_b[b] += w
    expected = sum(
        freq_a[label] * freq_b[label] for label in set(freq_a) | set(freq_b)
    ) / total**2
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def fleiss_kappa(rating_rows: list[tuple[Counter, int]]) -> float:
    """Row-count-weighted Fleiss' kappa; each row is (label counts of n raters, weight)."""
    rows = [(c, w) for c, w in rating_rows if sum(c.values()) >= 2]
    if not rows:
        return float("nan")
    total_weight = sum(w for _, w in rows)
    categories = sorted({label for counts, _ in rows for label in counts})
    p_j = {
        cat: sum(counts[cat] * w for counts, w in rows)
        / sum(sum(counts.values()) * w for counts, w in rows)
        for cat in categories
    }
    p_bar = 0.0
    for counts, w in rows:
        n = sum(counts.values())
        agree = (sum(v * v for v in counts.values()) - n) / (n * (n - 1))
        p_bar += agree * w
    p_bar /= total_weight
    p_e = sum(v * v for v in p_j.values())
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else float("nan")


def main() -> None:
    units = {u["unit_id"]: u for u in load_jsonl(UNITS_PATH)}
    with open(ROWS_MAP_PATH, encoding="utf-8") as fh:
        rows_map = json.load(fh)
    votes = load_votes()
    file_meta = {e["file"]: e for e in translation_files()}
    providers_present = [p for p in PROVIDERS if (RAW_DIR / f"{p}.jsonl").exists()]

    # Per-unit result records
    results = {}
    for uid, unit in units.items():
        vote_map = votes.get(uid, {})
        majority = majority_label(vote_map)
        results[uid] = {
            **unit,
            "votes": vote_map,
            "majority": majority,
            "decided": majority not in ("insufficient_votes", "no_majority"),
            "agree_fine": majority == unit["rule_label"],
            "agree_coarse": COARSE.get(majority) == COARSE.get(unit["rule_label"]),
        }

    with open(OUT_DIR / "ensemble_labels.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for record in results.values():
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Row-weighted agreement per (system, condition, language)
    lines = ["# Ensemble validation summary", ""]
    covered = sum(1 for r in results.values() if len(r["votes"]) == len(providers_present))
    lines += [
        f"- providers with results: {', '.join(providers_present) or 'none'}",
        f"- units: {len(results)}, fully voted: {covered}",
        "",
        "## Row-weighted agreement (rule label vs ensemble majority)",
        "",
        "| system | condition | language | rows | decided % | fine agree % | M/F/N agree % |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for fname, unit_ids in sorted(rows_map.items()):
        meta = file_meta[fname]
        n = len(unit_ids)
        decided = fine = coarse = 0
        for uid in unit_ids:
            r = results[uid]
            if r["decided"]:
                decided += 1
                fine += r["agree_fine"]
                coarse += r["agree_coarse"]
        lines.append(
            f"| {meta['system']} | {meta['condition']} | {meta['language']} | {n} "
            f"| {100 * decided / n:.2f} "
            f"| {100 * fine / decided if decided else 0:.2f} "
            f"| {100 * coarse / decided if decided else 0:.2f} |"
        )

    # Overall + kappas (row-weighted at unit level)
    weighted_pairs_rule = []
    fleiss_rows = []
    pairwise = defaultdict(list)
    for r in results.values():
        w = r["row_count"]
        if r["decided"]:
            weighted_pairs_rule.append((r["rule_label"], r["majority"], w))
        vote_map = r["votes"]
        fleiss_rows.append((Counter(vote_map.values()), w))
        names = sorted(vote_map)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairwise[(names[i], names[j])].append(
                    (vote_map[names[i]], vote_map[names[j]], w)
                )

    total_w = sum(w for _, _, w in weighted_pairs_rule)
    agree_w = sum(w for a, b, w in weighted_pairs_rule if a == b)
    coarse_w = sum(w for a, b, w in weighted_pairs_rule if COARSE[a] == COARSE.get(b))
    lines += [
        "",
        "## Overall (row-weighted)",
        "",
        f"- rule vs majority, fine labels: {100 * agree_w / total_w if total_w else 0:.3f}% "
        f"(kappa = {cohen_kappa(weighted_pairs_rule):.4f})",
        f"- rule vs majority, M/F/N: {100 * coarse_w / total_w if total_w else 0:.3f}%",
        f"- inter-model Fleiss' kappa: {fleiss_kappa(fleiss_rows):.4f}",
        "",
        "## Pairwise model agreement (row-weighted)",
        "",
        "| pair | agree % | kappa |",
        "|---|---:|---:|",
    ]
    for (a, b), pairs in sorted(pairwise.items()):
        tw = sum(w for _, _, w in pairs)
        ag = sum(w for x, y, w in pairs if x == y)
        lines.append(f"| {a}-{b} | {100 * ag / tw if tw else 0:.3f} | {cohen_kappa(pairs):.4f} |")

    (OUT_DIR / "agreement_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8",
                                                  newline="\n")

    # Disagreements for human adjudication, biggest impact first. Labels that
    # were already adjudicated (adjudication.csv) are carried over so that
    # re-running this script does not blank the human_label column.
    adjudicated: dict[str, str] = {}
    adjudication_path = OUT_DIR / "adjudication.csv"
    if adjudication_path.exists():
        with open(adjudication_path, encoding="utf-8-sig", newline="") as fh:
            adjudicated = {
                row["unit_id"]: row["human_label"] for row in csv.DictReader(fh)
            }
    disagreements = [
        r for r in results.values() if r["decided"] and not r["agree_fine"]
    ]
    disagreements.sort(key=lambda r: -r["row_count"])
    with open(OUT_DIR / "disagreements.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(
            ["unit_id", "language", "text", "rule_label", "rule_word",
             *providers_present, "majority", "row_count", "human_label"]
        )
        for r in disagreements:
            writer.writerow(
                [r["unit_id"], r["language"], r["text"], r["rule_label"],
                 r["rule_word"], *(r["votes"].get(p, "") for p in providers_present),
                 r["majority"], r["row_count"], adjudicated.get(r["unit_id"], "")]
            )

    undecided = [r for r in results.values() if not r["decided"]]
    print(f"summary -> {OUT_DIR / 'agreement_summary.md'}")
    print(f"disagreements: {len(disagreements)} units "
          f"({sum(r['row_count'] for r in disagreements)} rows) -> disagreements.csv")
    print(f"undecided (ties/missing votes): {len(undecided)} units")


if __name__ == "__main__":
    main()
