"""Table 2 of the paper: gender distribution and bias metrics per system and
target language under both conditions.

Reads results/data_summary.json and results/stats/ngb_ci.csv.

Usage: python -m src.analysis.make_integrated_table
Output: results/tables/integrated_table.md / .csv
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_SUMMARY = ROOT / "results" / "data_summary.json"
CI_CSV = ROOT / "results" / "stats" / "ngb_ci.csv"
OUT_DIR = ROOT / "results" / "tables"

SYSTEMS = [("deepl", "DeepL"), ("google", "Google")]
LANGS = [("kr", "KR"), ("en", "EN"), ("ja", "JA"), ("zh", "ZH")]
CONDITIONS = [("default", "Clothing Only"), ("color", "+ Color")]

CAPTION = ("Table 2. Gender distribution and bias metrics per system and target "
           "language under both conditions")
NOTE = ("Note: n = 1,600 (Clothing Only) and 33,600 (+ Color) per combination. "
        "N pools neutral, one-person, and subject-omission outputs. dF: change in "
        "Female ratio (%p). NGB = GPS x GAR; brackets: bootstrap 95% CI (B = 10,000).")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    ci = {}
    with open(CI_CSV, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cond_key = "default" if row["condition"] == "clothing_only" else "color"
            ci[(row["system"], cond_key, row["language"])] = row

    records = []
    for system, system_label in SYSTEMS:
        for lang, lang_label in LANGS:
            base_f = None
            for cond_key, cond_label in CONDITIONS:
                entry = summary[f"{system}_{cond_key}_tr_{lang}.json"]
                freq = entry["freq"]
                n = entry["total"]
                n_m, n_f = freq["male"], freq["female"]
                n_n = freq["neutral"] + freq["none_one"] + freq["omitted"] + freq["exception"]
                assert n_m + n_f + n_n == n, (system, cond_key, lang)
                f_ratio = 100 * n_f / n
                delta_f = None if base_f is None else f_ratio - base_f
                if base_f is None:
                    base_f = f_ratio
                stats = ci[(system, cond_key, lang)]
                records.append(
                    {
                        "System": system_label,
                        "Lang": lang_label,
                        "Condition": cond_label,
                        "n": n,
                        "M (%)": f"{n_m:,} ({100 * n_m / n:.1f})",
                        "F (%)": f"{n_f:,} ({f_ratio:.1f})",
                        "N (%)": f"{n_n:,} ({100 * n_n / n:.1f})",
                        "dF (%p)": "-" if delta_f is None else f"{delta_f:+.1f}",
                        "GPS": f"{entry['gps_overall']:+.3f}",
                        "GAR": f"{entry['gar_overall']:.3f}",
                        "NGB": f"{entry['ngb_overall']:+.3f}",
                        "NGB 95% CI": (
                            f"[{float(stats['NGB_lo']):+.3f}, {float(stats['NGB_hi']):+.3f}]"
                        ),
                        "UCA": f"{entry['uca_avg']:.4f}",
                    }
                )

    fields = list(records[0])
    with open(OUT_DIR / "integrated_table.csv", "w", encoding="utf-8-sig",
              newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        f"# {CAPTION}",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join("---:" if f == "n" else "---" for f in fields) + "|",
    ]
    for r in records:
        cells = [f"{r[f]:,}" if f == "n" else str(r[f]) for f in fields]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", NOTE, ""]
    (OUT_DIR / "integrated_table.md").write_text("\n".join(lines), encoding="utf-8",
                                                 newline="\n")
    print(f"-> {OUT_DIR / 'integrated_table.md'} / integrated_table.csv "
          f"({len(records)} rows)")


if __name__ == "__main__":
    main()
