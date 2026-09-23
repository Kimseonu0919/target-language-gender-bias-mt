"""Bias metrics computed from detector labels.

Per (system, condition, target language) file:
  frequency  counts of male / female / neutral / none_one / omitted / exception
  NGB        (n_F - n_M) / (n_M + n_F + n_N), the primary bias metric;
             NGB = GPS x GAR with GPS = (n_F - n_M) / (n_M + n_F) and
             GAR = (n_M + n_F) / (n_M + n_F + n_N)
  UCA        adjusted coefficient of unalikeability over the gender words the
             translator chose (lexical variability, as in Park et al. 2026)
  JSD        Jensen-Shannon divergence (natural log) between the M/F/N
             distribution of a clothing item without color and with each
             color

N pools neutral, none_one, omitted and exception. Rows are grouped by clothing
item, color and context by position: the source files are written in a fixed
order (context-major, then clothing, then color), so a file must be complete
and unshuffled (1,600 rows without color, 33,600 with color).

Usage: python -m src.analysis.calculate_metrics FILE [FILE ...]
       python -m src.analysis.calculate_metrics --jsd DEFAULT_FILE COLOR_FILE
       python -m src.analysis.calculate_metrics --all translations
The numbers used in the paper are produced by src.analysis.compute_summary.
"""

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ..detector.pronoun_gender import detect_pronoun_gender_detailed

NUM_CLOTHING_ITEMS = 16
NUM_COLORS = 21
NUM_CONTEXTS = 100

CLOTHING_ITEMS = [
    "jeans", "jacket", "knitwear", "T-shirt", "shirt", "coat",
    "cardigan", "windbreaker", "dress", "blouse", "pants", "sweater",
    "shorts", "hoodie", "skirt", "suit",
]

# Display names for the 21 color slots. The Turkish color term of a slot
# can differ between clothing items (see README, "Notes on the stimuli").
COLOR_NAMES = [
    "gold", "light_gray", "gray", "mint", "green", "olive",
    "red", "burgundy", "beige", "brown", "black", "light_blue",
    "blue", "navy", "ivory", "yellow", "orange", "purple",
    "plum", "pink", "white",
]

# UC_max = n/(n-1) * (1 - 1/K) evaluated at n = 100 with K fixed per language
# (KR 9, EN 2, JA 9, ZH 9), the convention of Park et al. (2026). The same
# constant is used for both conditions, so UCA is comparable across cells.
# 0.8978 is the truncated value used for the paper's tables.
PAPER_UC_MAX = {
    "ko": 0.8978,
    "en": 0.5051,
    "ja": 0.8978,
    "zh": 0.8978,
}

EXPECTED_ROWS = {False: NUM_CLOTHING_ITEMS * NUM_CONTEXTS,
                 True: NUM_CLOTHING_ITEMS * NUM_COLORS * NUM_CONTEXTS}


def load_json(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def check_row_count(n_rows: int, is_color: bool, name: str) -> None:
    expected = EXPECTED_ROWS[is_color]
    if n_rows != expected:
        raise SystemExit(f"{name}: {n_rows} rows, expected {expected} "
                         f"(positional grouping needs the complete file)")


# Positional indexing. Without color: row i = context i // 16, clothing i % 16.
# With color: row i = context i // 336, clothing (i % 336) // 21, color i % 21.

def get_clothing_idx_default(item_index: int) -> int:
    return item_index % NUM_CLOTHING_ITEMS


def get_clothing_idx_color(item_index: int) -> int:
    return (item_index % (NUM_CLOTHING_ITEMS * NUM_COLORS)) // NUM_COLORS


def get_color_idx(item_index: int) -> int:
    return item_index % NUM_COLORS


def detect_item(item: Dict[str, Any]) -> Dict[str, str]:
    return detect_pronoun_gender_detailed({
        "sentence": item.get("sentence", ""),
        "language": item.get("language", ""),
    })


def build_frequency_table(data: List[Dict[str, Any]]):
    """Return (gender counts, matched-word counts, per-row detector output)."""
    gender_counts: Counter = Counter()
    word_counts: Counter = Counter()
    items_detail: List[Dict[str, str]] = []
    for item in data:
        result = detect_item(item)
        gender_counts[result["gender"]] += 1
        word_counts[result["matched_word"] or f"({result['gender']})"] += 1
        items_detail.append(result)
    return gender_counts, word_counts, items_detail


def classify_mfn(detail: Dict[str, str]) -> str:
    """Map a detector label to male / female / neutral (N pools everything else)."""
    g = detail["gender"]
    if g in ("male", "female"):
        return g
    return "neutral"


def mfn_counts(items_detail: List[Dict[str, str]]) -> Counter:
    c: Counter = Counter()
    for d in items_detail:
        c[classify_mfn(d)] += 1
    return c


def calculate_ngb(counts: Counter) -> float:
    n_m, n_f, n_n = counts.get("male", 0), counts.get("female", 0), counts.get("neutral", 0)
    total = n_m + n_f + n_n
    return (n_f - n_m) / total if total else 0.0


def calculate_gar(counts: Counter) -> float:
    n_m, n_f, n_n = counts.get("male", 0), counts.get("female", 0), counts.get("neutral", 0)
    total = n_m + n_f + n_n
    return (n_m + n_f) / total if total else 0.0


def calculate_gps(counts: Counter) -> float:
    n_m, n_f = counts.get("male", 0), counts.get("female", 0)
    return (n_f - n_m) / (n_m + n_f) if (n_m + n_f) else 0.0


def _metric_block(items: List[Dict[str, str]]) -> Dict[str, Any]:
    counts = mfn_counts(items)
    return {
        "ngb": calculate_ngb(counts),
        "gar": calculate_gar(counts),
        "gps": calculate_gps(counts),
        "counts": counts,
        "total": sum(counts.values()),
    }


def ngb_per_clothing(items_detail, is_color=False) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for i, detail in enumerate(items_detail):
        cidx = get_clothing_idx_color(i) if is_color else get_clothing_idx_default(i)
        groups[CLOTHING_ITEMS[cidx]].append(detail)
    return {name: _metric_block(groups.get(name, [])) for name in CLOTHING_ITEMS}


def compute_uc_max(n: int, k: int) -> float:
    if k <= 1 or n <= 1:
        return 1.0
    return (n / (n - 1)) * (1 - 1 / k)


def calculate_uca_from_list(labels: List[str], uc_max: Optional[float] = None) -> float:
    """UCA of a list of categorical labels; uc_max defaults to the sample's own K and n."""
    n = len(labels)
    if n <= 1:
        return 0.0
    counts = Counter(labels)
    if uc_max is None:
        uc_max = compute_uc_max(n, len(counts))
    sum_p_sq = sum((c / n) ** 2 for c in counts.values())
    uc = (n / (n - 1)) * (1 - sum_p_sq)
    return uc / uc_max if uc_max > 0 else 0.0


def uca_per_clothing(items_detail, is_color=False, uc_max=None) -> Dict[str, Dict[str, Any]]:
    """UCA per clothing item over the matched surface forms (label name when no word matched)."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for i, detail in enumerate(items_detail):
        cidx = get_clothing_idx_color(i) if is_color else get_clothing_idx_default(i)
        groups[CLOTHING_ITEMS[cidx]].append(detail["matched_word"] or detail["gender"])
    out = {}
    for name in CLOTHING_ITEMS:
        labels = groups.get(name, [])
        out[name] = {
            "uca": calculate_uca_from_list(labels, uc_max=uc_max) if labels else 0.0,
            "count": len(labels),
            "labels": Counter(labels),
        }
    return out


def _kl_divergence(p, q) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0 and qi > 0:
            s += pi * math.log(pi / qi)
    return s


def calculate_jsd(p_dist, q_dist) -> float:
    assert len(p_dist) == len(q_dist), "Distributions must have same length"
    m = [(p + q) / 2 for p, q in zip(p_dist, q_dist)]
    return 0.5 * _kl_divergence(p_dist, m) + 0.5 * _kl_divergence(q_dist, m)


_MFN = ("male", "female", "neutral")


def _counts_to_dist(counts: Counter, categories=_MFN) -> List[float]:
    total = sum(counts.get(c, 0) for c in categories)
    if total == 0:
        return [0.0] * len(categories)
    return [counts.get(c, 0) / total for c in categories]


def jsd_analysis(default_details, color_details) -> Tuple[List[Dict[str, Any]], float]:
    """JSD between P_i (clothing i, no color) and Q_ic (clothing i with color c).

    Returns the 336 (clothing, color) rows sorted by JSD, descending, and
    their mean.
    """
    p_by: Dict[str, Counter] = defaultdict(Counter)
    for i, d in enumerate(default_details):
        p_by[CLOTHING_ITEMS[get_clothing_idx_default(i)]][classify_mfn(d)] += 1

    q_by: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    for i, d in enumerate(color_details):
        q_by[(CLOTHING_ITEMS[get_clothing_idx_color(i)], get_color_idx(i))][classify_mfn(d)] += 1

    rows = []
    for (clothing, c), q_counts in q_by.items():
        p_counts = p_by[clothing]
        rows.append({
            "clothing": clothing,
            "color_idx": c,
            "color": COLOR_NAMES[c],
            "jsd": calculate_jsd(_counts_to_dist(p_counts), _counts_to_dist(q_counts)),
            "p_counts": tuple(p_counts.get(k, 0) for k in _MFN),
            "q_counts": tuple(q_counts.get(k, 0) for k in _MFN),
            "p_ngb": calculate_ngb(p_counts),
            "q_ngb": calculate_ngb(q_counts),
        })
    rows.sort(key=lambda x: x["jsd"], reverse=True)
    avg = sum(r["jsd"] for r in rows) / len(rows) if rows else 0.0
    return rows, avg


# Console reports

def analyze_single_file(filepath: str):
    data = load_json(filepath)
    filename = os.path.basename(filepath)
    lang = data[0].get("language", "??") if data else "??"
    lang_key = "ko" if lang == "kr" else lang
    is_color = "_color_" in filename
    check_row_count(len(data), is_color, filename)

    print(f"\n{'=' * 60}")
    print(f"[{filename}] ({len(data)} sentences, lang={lang})")
    print(f"{'=' * 60}")

    gc, wc, details = build_frequency_table(data)

    print("\n1. Gender category frequency")
    total = sum(gc.values())
    for g in ["male", "female", "neutral", "none_one", "omitted", "exception"]:
        n = gc.get(g, 0)
        print(f"   {g:10s}: {n:6d} ({n / total * 100:5.1f}%)")
    if gc.get("exception"):
        print(f"   warning: {gc['exception']} rows could not be split into two "
              f"sentences and are counted as N")

    counts_mfn = mfn_counts(details)
    ngb_overall = calculate_ngb(counts_mfn)
    gar_overall = calculate_gar(counts_mfn)
    gps_overall = calculate_gps(counts_mfn)

    print("\n2. NGB / GAR / GPS")
    print(f"   n_M={counts_mfn.get('male', 0)}, n_F={counts_mfn.get('female', 0)}, "
          f"n_N={counts_mfn.get('neutral', 0)}")
    print(f"   NGB = {ngb_overall:+.4f}   GAR = {gar_overall:.4f}   GPS = {gps_overall:+.4f}")

    print("\n3. UCA per clothing item")
    uca_d = uca_per_clothing(details, is_color=is_color, uc_max=PAPER_UC_MAX.get(lang_key))
    values = [uca_d[name]["uca"] for name in CLOTHING_ITEMS if uca_d[name]["count"]]
    for name in CLOTHING_ITEMS:
        if uca_d[name]["count"]:
            print(f"   {name:14s}  {uca_d[name]['uca']:8.4f}")
    if values:
        print(f"   {'average':14s}  {sum(values) / len(values):8.4f}")

    print("\n4. NGB per clothing item")
    ngb_d = ngb_per_clothing(details, is_color=is_color)
    print(f"   {'item':14s}  {'NGB':>8s}  {'M':>5s}  {'F':>5s}  {'N':>5s}")
    for name in CLOTHING_ITEMS:
        r = ngb_d[name]
        if r["total"]:
            c = r["counts"]
            print(f"   {name:14s}  {r['ngb']:+8.4f}  {c.get('male', 0):5d}  "
                  f"{c.get('female', 0):5d}  {c.get('neutral', 0):5d}")

    return gc, wc, details, uca_d, ngb_d


def analyze_jsd_pair(default_path: str, color_path: str):
    default_data = load_json(default_path)
    color_data = load_json(color_path)
    dname = os.path.basename(default_path)
    cname = os.path.basename(color_path)
    check_row_count(len(default_data), False, dname)
    check_row_count(len(color_data), True, cname)

    print(f"\n{'=' * 60}")
    print(f"JSD: {dname} vs {cname}")
    print(f"{'=' * 60}")
    _, _, default_details = build_frequency_table(default_data)
    _, _, color_details = build_frequency_table(color_data)
    rows, avg_jsd = jsd_analysis(default_details, color_details)

    print(f"\n  average JSD: {avg_jsd:.6f}")
    print("\n  top 10 (clothing, color) pairs by JSD:")
    print(f"  {'#':>3s}  {'clothing':14s}  {'color':12s}  {'JSD':>7s}  "
          f"{'P(M,F,N)':>14s}  {'Q(M,F,N)':>14s}  {'P_NGB':>7s}  {'Q_NGB':>7s}")
    for rank, r in enumerate(rows[:10], 1):
        p, q = r["p_counts"], r["q_counts"]
        print(f"  {rank:3d}  {r['clothing']:14s}  {r['color']:12s}  {r['jsd']:7.4f}  "
              f"({p[0]:>3d},{p[1]:>3d},{p[2]:>3d})  ({q[0]:>3d},{q[1]:>3d},{q[2]:>3d})  "
              f"{r['p_ngb']:+7.4f}  {r['q_ngb']:+7.4f}")
    return rows, avg_jsd


def analyze_all_files(result_dir: str):
    names = sorted(f for f in os.listdir(result_dir) if f.endswith(".json"))
    default_files = [f for f in names if "default" in f]
    color_files = [f for f in names if "color" in f]
    for fname in default_files + color_files:
        analyze_single_file(os.path.join(result_dir, fname))
    for dfname in default_files:
        cfname = dfname.replace("default", "color")
        if cfname in color_files:
            analyze_jsd_pair(os.path.join(result_dir, dfname), os.path.join(result_dir, cfname))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print bias metrics for translation files.")
    parser.add_argument("files", nargs="*", help="translation JSON file(s)")
    parser.add_argument("--jsd", nargs=2, metavar=("DEFAULT", "COLOR"),
                        help="JSD between a clothing-only file and its color file")
    parser.add_argument("--all", metavar="DIR", help="report every file in a directory")
    args = parser.parse_args()

    if args.all:
        analyze_all_files(args.all)
    elif args.jsd:
        analyze_jsd_pair(args.jsd[0], args.jsd[1])
    elif args.files:
        for fpath in args.files:
            analyze_single_file(fpath)
    else:
        parser.print_help()
