"""Build the source sentence files from data/fashion_data.xlsx.

Usage: python -m src.preprocessing.preprocessing [--lang tr|fi|hu|et|all]

Each of the 100 context sentences is paired with each of the 16 clothing
sentences (default_{lang}.json) and with each of the 336 color + clothing
sentences (color_{lang}.json). The files are written to data/generated/.
The study is Turkish-only, so tr is the default; the workbook also carries
the Estonian, Finnish and Hungarian columns of an earlier study, which
--lang all processes as well.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "fashion_data.xlsx"
OUTPUT_DIR = ROOT / "data" / "generated"

WEAR_SHEET = "DATA-입고있다"
COLOR_SHEET = "DATA-색깔-입고있다"
CONTEXT_COLS = [2, 3]

# Zero-based column of each language's sentences in the two DATA sheets:
# the clothing sheet lists the languages in columns C to F, the color sheet
# in D to G. The context sentences are columns C and D of CONTEXT-<name>.
LANG_CONFIG = {
    "tr": {"name": "터키어", "wear_col": 2, "color_col": 3},
    "fi": {"name": "핀란드어", "wear_col": 3, "color_col": 4},
    "hu": {"name": "헝가리어", "wear_col": 4, "color_col": 5},
    "et": {"name": "에스토니아어", "wear_col": 5, "color_col": 6},
}


def column(df, idx):
    return [s.strip() for s in df.iloc[:, idx].dropna().astype(str)]


def combine(contexts, items, lang):
    return [
        {"sentence": f"{ctx} {item}", "language": lang}
        for ctx, item in itertools.product(contexts, items)
    ]


def write_json(rows, path):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=4)
    print(f"{len(rows)} sentences -> {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lang", default="tr", choices=[*LANG_CONFIG, "all"])
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        sys.exit(f"workbook not found: {INPUT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(INPUT_PATH)
    df_wear = pd.read_excel(xls, sheet_name=WEAR_SHEET)
    df_color = pd.read_excel(xls, sheet_name=COLOR_SHEET)

    for lang in list(LANG_CONFIG) if args.lang == "all" else [args.lang]:
        config = LANG_CONFIG[lang]
        df_ctx = pd.read_excel(xls, sheet_name=f"CONTEXT-{config['name']}")
        contexts = [s for idx in CONTEXT_COLS for s in column(df_ctx, idx)]
        wear = column(df_wear, config["wear_col"])
        color = column(df_color, config["color_col"])
        write_json(combine(contexts, wear, lang), OUTPUT_DIR / f"default_{lang}.json")
        write_json(combine(contexts, color, lang), OUTPUT_DIR / f"color_{lang}.json")


if __name__ == "__main__":
    main()
