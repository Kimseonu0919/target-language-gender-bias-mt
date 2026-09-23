"""Tables 3 to 6 of the paper.

  Table 3  average NGB per clothing item over the 8 (system x language)
           combinations, both conditions, the NGB shift, and the average JSD
  Table 4  average NGB (Clothing + Color) and JSD per color
  Table 5  pairwise agreement of the M/F/N label between target languages
  Table 6  proportion of inputs that receive the same label in all four
           target languages

Tables 3 and 4 read results/data_summary.json. Tables 5 and 6 classify the
translation files again with the rule-based detector (about 20 seconds).

Usage: python -m src.analysis.make_paper_tables
Output: results/tables/table3_clothing.md/.csv, table4_color.md/.csv,
        table5_pairwise_agreement.md/.csv, table6_all_identical.md/.csv
"""

import csv
import json
from collections import Counter
from pathlib import Path

from .calculate_metrics import (
    CLOTHING_ITEMS,
    COLOR_NAMES,
    NUM_COLORS,
    calculate_jsd,
    check_row_count,
    classify_mfn,
    detect_item,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_SUMMARY = ROOT / "results" / "data_summary.json"
TRANSLATIONS = ROOT / "translations"
OUT_DIR = ROOT / "results" / "tables"

SYSTEMS = [("deepl", "DeepL"), ("google", "Google")]
LANGS = ["kr", "en", "ja", "zh"]
CONDITIONS = [("default", "Clothing Only"), ("color", "+ Color")]
PAIRS = [("kr", "zh"), ("en", "zh"), ("en", "kr"), ("ja", "kr"), ("ja", "zh"), ("en", "ja")]
MFN = ("male", "female", "neutral")


def fname(system, cond, lang):
    return f"{system}_{cond}_tr_{lang}.json"


def dist(counts):
    total = sum(counts.get(k, 0) for k in MFN)
    return [counts.get(k, 0) / total if total else 0.0 for k in MFN]


def mean(values):
    return sum(values) / len(values)


def fmt(value, places, sign=False):
    return f"{value:+.{places}f}" if sign else f"{value:.{places}f}"


def display_color(name):
    return name.replace("_", " ")


def write_table(stem, caption, fields, rows, note=None, align_right=()):
    with open(OUT_DIR / f"{stem}.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# {caption}",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---:" if f in align_right else "---" for f in fields) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[f]) for f in fields) + " |")
    lines.append("")
    if note:
        lines += [note, ""]
    (OUT_DIR / f"{stem}.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"-> {OUT_DIR / stem}.md / .csv ({len(rows)} rows)")


def table3_and_4(summary):
    """Per clothing item and per color, averaged over the 8 combinations."""
    combos = [(s, lang) for s, _ in SYSTEMS for lang in LANGS]

    ngb_default = {item: [] for item in CLOTHING_ITEMS}
    ngb_color = {item: [] for item in CLOTHING_ITEMS}
    jsd_item = {item: [] for item in CLOTHING_ITEMS}
    ngb_col = {c: [] for c in range(NUM_COLORS)}
    jsd_col = {c: [] for c in range(NUM_COLORS)}

    for system, lang in combos:
        d_entry = summary[fname(system, "default", lang)]
        c_entry = summary[fname(system, "color", lang)]
        for item in CLOTHING_ITEMS:
            ngb_default[item].append(d_entry["ngb_per_clothing"][item]["ngb"])
            ngb_color[item].append(c_entry["ngb_per_clothing"][item]["ngb"])
            p = dist(d_entry["ngb_per_clothing"][item]["counts"])
            for c in range(NUM_COLORS):
                q = dist(c_entry["ngb_per_clothing_color"][f"{item}|{c}"]["counts"])
                jsd = calculate_jsd(p, q)
                jsd_item[item].append(jsd)
                jsd_col[c].append(jsd)
        for c in range(NUM_COLORS):
            ngb_col[c].append(c_entry["ngb_per_color"][str(c)]["ngb"])

    rows3 = []
    for item in CLOTHING_ITEMS:
        a, b = mean(ngb_default[item]), mean(ngb_color[item])
        shift = b - a
        rows3.append({
            "Category": item,
            "NGB Clothing Only": fmt(a, 3, sign=True),
            "NGB + Color": fmt(b, 3, sign=True),
            "NGB Shift": fmt(shift, 2, sign=True) + ("*" if abs(shift) > 0.5 else ""),
            "Average JSD": fmt(mean(jsd_item[item]), 4),
            "_jsd": mean(jsd_item[item]),
        })
    rows3.sort(key=lambda r: -r["_jsd"])
    for r in rows3:
        del r["_jsd"]
    write_table(
        "table3_clothing",
        "Table 3. Average NGB per clothing item over the 8 (system x language) "
        "combinations, sorted by JSD",
        ["Category", "NGB Clothing Only", "NGB + Color", "NGB Shift", "Average JSD"],
        rows3,
        note="* = |NGB Shift| > 0.5. NGB Shift = NGB (+ Color) - NGB (Clothing Only). "
             "Average JSD: mean over 21 colors and 8 (system x language) combinations "
             "of the JSD between the clothing-only and the clothing+color M/F/N "
             "distributions. Two Clothing Only means are exact ties at the fourth "
             "decimal (sweater -0.3375, suit -0.9825); the paper prints them as "
             "-0.337 and -0.982.",
    )

    rows4 = []
    for c in range(NUM_COLORS):
        rows4.append({
            "Color": display_color(COLOR_NAMES[c]),
            "NGB": fmt(mean(ngb_col[c]), 3, sign=True),
            "JSD": fmt(mean(jsd_col[c]), 4),
            "_ngb": mean(ngb_col[c]),
        })
    rows4.sort(key=lambda r: -r["_ngb"])
    for r in rows4:
        del r["_ngb"]
    write_table(
        "table4_color",
        "Table 4. Average NGB and JSD per color over 16 clothing items and 8 "
        "(system x language) combinations, sorted by NGB",
        ["Color", "NGB", "JSD"],
        rows4,
        note="NGB is computed under the Clothing + Color condition. JSD compares each "
             "clothing-color distribution with the corresponding Clothing Only "
             "distribution.",
    )


def labels_per_row(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    check_row_count(len(data), "_color_" in path.name, path.name)
    return [classify_mfn(detect_item(item)) for item in data]


def table5_and_6():
    """Agreement of the M/F/N label across the four target languages."""
    rows5, rows6 = [], []
    for system, system_label in SYSTEMS:
        for cond, cond_label in CONDITIONS:
            labels = {lang: labels_per_row(TRANSLATIONS / fname(system, cond, lang))
                      for lang in LANGS}
            n = len(labels[LANGS[0]])

            row = {"System": system_label, "Dataset": cond_label}
            rates = []
            for a, b in PAIRS:
                agree = sum(x == y for x, y in zip(labels[a], labels[b]))
                rates.append(100 * agree / n)
                row[f"{a.upper()}-{b.upper()}"] = f"{rates[-1]:.2f}"
            row["Average"] = f"{mean(rates):.2f}"
            rows5.append(row)

            same = Counter()
            for i in range(n):
                values = {labels[lang][i] for lang in LANGS}
                if len(values) == 1:
                    same[values.pop()] += 1
            identical = sum(same.values())
            rows6.append({
                "System": system_label,
                "Dataset": cond_label,
                "M (all male)": f"{same['male']:,}",
                "F (all female)": f"{same['female']:,}",
                "N (all neutral)": f"{same['neutral']:,}",
                "All identical": f"{identical:,}",
                "Not all identical": f"{n - identical:,}",
                "Total": f"{n:,}",
                "Agreement (%)": f"{100 * identical / n:.2f}",
            })

    write_table(
        "table5_pairwise_agreement",
        "Table 5. Pairwise agreement of gender labels (M/F/N) between target languages "
        "on identical inputs (%)",
        ["System", "Dataset"] + [f"{a.upper()}-{b.upper()}" for a, b in PAIRS] + ["Average"],
        rows5,
    )
    write_table(
        "table6_all_identical",
        "Table 6. Proportion of inputs receiving the same gender label in all four "
        "target languages (%)",
        ["System", "Dataset", "M (all male)", "F (all female)", "N (all neutral)",
         "All identical", "Not all identical", "Total", "Agreement (%)"],
        rows6,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    table3_and_4(summary)
    table5_and_6()


if __name__ == "__main__":
    main()
