"""Build results/data_summary.json from the translation files.

For each of the 16 files the summary holds the label frequencies, NGB / GAR /
GPS, UCA per clothing item, NGB per clothing item, per color and per
(clothing, color) cell, and the matched-word counts. The "_jsd_pairs" block
holds, for each (system, language), the mean JSD between the clothing-only
and clothing+color label distributions and the ten (clothing, color) pairs
with the largest JSD. Tables 2 to 4 and the figures are derived from this
file; Tables 5 and 6 classify the translation files directly.

Usage: python -m src.analysis.compute_summary [--out PATH]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ..detector.pronoun_gender import detect_pronoun_gender_detailed
from .calculate_metrics import (
    CLOTHING_ITEMS,
    COLOR_NAMES,
    NUM_COLORS,
    PAPER_UC_MAX,
    calculate_uca_from_list,
    check_row_count,
    get_clothing_idx_color,
    get_clothing_idx_default,
    get_color_idx,
    jsd_analysis,
)

ROOT = Path(__file__).resolve().parents[2]
TRANSLATIONS = ROOT / "translations"
DEFAULT_OUT = ROOT / "results" / "data_summary.json"

GENDER_KEYS = ["male", "female", "neutral", "none_one", "omitted", "exception"]


def mf_counts(group_counter):
    """Counts for an NGB block: zero-valued keys are left out and every non-M/F
    outcome is pooled under "neutral"."""
    out = {}
    for key in ("male", "female"):
        if group_counter.get(key, 0):
            out[key] = group_counter[key]
    pooled = sum(group_counter.get(k, 0)
                 for k in ("neutral", "none_one", "omitted", "exception"))
    if pooled:
        out["neutral"] = pooled
    return out


def ngb_block(group_counter, total):
    n_m = group_counter.get("male", 0)
    n_f = group_counter.get("female", 0)
    return {
        "ngb": (n_f - n_m) / total if total else 0.0,
        "gar": (n_m + n_f) / total if total else 0.0,
        "gps": (n_f - n_m) / (n_m + n_f) if (n_m + n_f) else 0.0,
        "counts": mf_counts(group_counter),
        "total": total,
    }


def summarize_file(path: Path):
    """Return (summary entry, per-row detector output) for one translation file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    is_color = "_color_" in path.name
    language = data[0]["language"] if data else ""
    check_row_count(len(data), is_color, path.name)

    freq = Counter()
    word_counts = Counter()
    details = []
    for item in data:
        result = detect_pronoun_gender_detailed(item)
        details.append(result)
        freq[result["gender"]] += 1
        word_counts[result["matched_word"] or f"({result['gender']})"] += 1
    if freq["exception"]:
        print(f"  warning: {path.name}: {freq['exception']} rows could not be split "
              f"into two sentences and are counted as N")

    total = len(data)
    n_m, n_f = freq["male"], freq["female"]
    n_n = total - n_m - n_f

    mfn = {"male": n_m, "female": n_f}
    if n_n:
        mfn["neutral"] = n_n

    by_clothing = defaultdict(Counter)
    words_by_clothing = defaultdict(list)
    by_color = defaultdict(Counter)
    by_clothing_color = defaultdict(Counter)
    for i, det in enumerate(details):
        cidx = get_clothing_idx_color(i) if is_color else get_clothing_idx_default(i)
        name = CLOTHING_ITEMS[cidx]
        by_clothing[name][det["gender"]] += 1
        words_by_clothing[name].append(det["matched_word"] or det["gender"])
        if is_color:
            kidx = get_color_idx(i)
            by_color[kidx][det["gender"]] += 1
            by_clothing_color[(name, kidx)][det["gender"]] += 1

    uc_max = PAPER_UC_MAX["ko" if language == "kr" else language]
    uca_pc = {
        name: {
            "uca": calculate_uca_from_list(words_by_clothing[name], uc_max=uc_max),
            "count": len(words_by_clothing[name]),
        }
        for name in CLOTHING_ITEMS
    }
    valid = [v["uca"] for v in uca_pc.values() if v["count"]]

    entry = {
        "language": language,
        "is_color": is_color,
        "total": total,
        "freq": {k: freq.get(k, 0) for k in GENDER_KEYS},
        "mfn_counts": mfn,
        "ngb_overall": (n_f - n_m) / total,
        "gar_overall": (n_m + n_f) / total,
        "gps_overall": (n_f - n_m) / (n_m + n_f) if (n_m + n_f) else 0.0,
        "uca_avg": sum(valid) / len(valid) if valid else 0.0,
        "uca_per_clothing": uca_pc,
        "ngb_per_clothing": {
            name: ngb_block(by_clothing[name], sum(by_clothing[name].values()))
            for name in CLOTHING_ITEMS
        },
        "word_counts": dict(word_counts),
    }
    if is_color:
        entry["ngb_per_color"] = {
            str(k): {
                "color": COLOR_NAMES[k],
                "ngb": ngb_block(by_color[k], sum(by_color[k].values()))["ngb"],
                "counts": mf_counts(by_color[k]),
                "total": sum(by_color[k].values()),
            }
            for k in range(NUM_COLORS)
        }
        entry["ngb_per_clothing_color"] = {
            f"{name}|{k}": {
                "clothing": name,
                "color_idx": k,
                "color": COLOR_NAMES[k],
                "ngb": ngb_block(by_clothing_color[(name, k)],
                                 sum(by_clothing_color[(name, k)].values()))["ngb"],
                "counts": mf_counts(by_clothing_color[(name, k)]),
            }
            for name in CLOTHING_ITEMS
            for k in range(NUM_COLORS)
        }
    return entry, details


def jsd_pair_block(dname, cname, default_details, color_details):
    rows, avg_jsd = jsd_analysis(default_details, color_details)
    return {
        "default": dname,
        "color": cname,
        "avg_jsd": avg_jsd,
        "top10": [
            {
                "clothing": r["clothing"],
                "color_idx": r["color_idx"],
                "color": r["color"],
                "jsd": r["jsd"],
                "p_counts": list(r["p_counts"]),
                "q_counts": list(r["q_counts"]),
                "p_ngb": r["p_ngb"],
                "q_ngb": r["q_ngb"],
            }
            for r in rows[:10]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build results/data_summary.json.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    summary = {}
    details_by_file = {}
    for path in sorted(TRANSLATIONS.glob("*_tr_*.json")):
        summary[path.name], details_by_file[path.name] = summarize_file(path)
        print("summarized", path.name)

    summary["_jsd_pairs"] = {}
    for dname in sorted(n for n in details_by_file if "_default_" in n):
        cname = dname.replace("_default_", "_color_")
        if cname in details_by_file:
            summary["_jsd_pairs"][f"{dname}__VS__{cname}"] = jsd_pair_block(
                dname, cname, details_by_file[dname], details_by_file[cname])

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
