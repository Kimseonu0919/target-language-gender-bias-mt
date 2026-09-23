"""Bootstrap confidence intervals and chi-square tests.

Computes, from the M/F/N frequency counts in results/data_summary.json:
  1. Bootstrap 95% CIs (B=10,000, multinomial resampling) for NGB, GPS, GAR
     per (system x condition x target language).
  2. Chi-square tests with Cramer's V effect sizes, Holm-corrected within
     each hypothesis family:
       A. language effect   - M/F/N x 4 languages, per system x condition
       B. condition effect  - M/F/N x {clothing, +color}, per system x language
       C. system effect     - M/F/N x {DeepL, Google}, per condition x language

Cramer's V is reported alongside p-values because with n up to 134,400 even
trivial differences reach p < .05. Categories that are empty in every group
are dropped from a table before the test; scipy applies the Yates continuity
correction to the 2x2 tables that remain.

Usage: python -m src.analysis.significance
Output: results/stats/ (significance_summary.md, ngb_ci.csv, chi2_tests.csv)
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2_contingency

ROOT = Path(__file__).resolve().parents[2]
DATA_SUMMARY = ROOT / "results" / "data_summary.json"
OUT_DIR = ROOT / "results" / "stats"

B = 10_000
SEED = 20260829
CONDITIONS = {"default": "clothing_only", "color": "clothing_color"}
LANGS = ["kr", "en", "ja", "zh"]
SYSTEMS = ["deepl", "google"]


def load_counts() -> dict:
    """(system, condition, language) -> np.array([nM, nF, nN])."""
    with open(DATA_SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    counts = {}
    for fname, entry in summary.items():
        if "freq" not in entry:
            continue
        system, cond_key, _, lang = Path(fname).stem.split("_")
        freq = entry["freq"]
        n_neutral = freq["neutral"] + freq["none_one"] + freq["omitted"] + freq["exception"]
        counts[(system, CONDITIONS[cond_key], lang)] = np.array(
            [freq["male"], freq["female"], n_neutral]
        )
    return counts


def metrics(c: np.ndarray) -> tuple[float, float, float]:
    """Return (NGB, GPS, GAR) from [nM, nF, nN] counts."""
    n = c.sum()
    n_m, n_f = c[0], c[1]
    ngb = (n_f - n_m) / n
    gps = (n_f - n_m) / (n_m + n_f) if (n_m + n_f) else np.nan
    gar = (n_m + n_f) / n
    return ngb, gps, gar


def bootstrap_ci(c: np.ndarray, rng: np.random.Generator):
    """Percentile 95% CIs for NGB, GPS, GAR via multinomial resampling."""
    n = int(c.sum())
    samples = rng.multinomial(n, c / n, size=B).astype(float)
    n_m, n_f = samples[:, 0], samples[:, 1]
    ngb = (n_f - n_m) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        gps = np.where(n_m + n_f > 0, (n_f - n_m) / (n_m + n_f), np.nan)
    gar = (n_m + n_f) / n

    def ci(x):
        return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))
    return ci(ngb), ci(gps), ci(gar)


def cramers_v(table: np.ndarray) -> float:
    table = table[:, table.sum(axis=0) > 0]
    chi2, _, _, _ = chi2_contingency(table)
    n = table.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else np.nan


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted.tolist()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    counts = load_counts()

    # 1. Bootstrap CIs
    ci_rows = []
    for system in SYSTEMS:
        for condition in CONDITIONS.values():
            for lang in LANGS:
                c = counts[(system, condition, lang)]
                ngb, gps, gar = metrics(c)
                (ngb_ci, gps_ci, gar_ci) = bootstrap_ci(c, rng)
                ci_rows.append(
                    {
                        "system": system, "condition": condition, "language": lang,
                        "n": int(c.sum()),
                        "NGB": round(ngb, 6),
                        "NGB_lo": round(ngb_ci[0], 6), "NGB_hi": round(ngb_ci[1], 6),
                        "GPS": round(gps, 6),
                        "GPS_lo": round(gps_ci[0], 6), "GPS_hi": round(gps_ci[1], 6),
                        "GAR": round(gar, 6),
                        "GAR_lo": round(gar_ci[0], 6), "GAR_hi": round(gar_ci[1], 6),
                    }
                )
    with open(OUT_DIR / "ngb_ci.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ci_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ci_rows)

    # 2. Chi-square families
    tests = []

    def add_test(family, label, table):
        # Drop categories absent in every group (all-zero columns), which
        # would make expected frequencies zero; dof adjusts accordingly.
        table = table[:, table.sum(axis=0) > 0]
        chi2, p, dof, _ = chi2_contingency(table)
        tests.append(
            {
                "family": family, "test": label,
                "n": int(table.sum()), "dof": dof,
                "chi2": round(float(chi2), 2), "p_raw": float(p),
                "cramers_v": round(cramers_v(table), 4),
            }
        )

    for system in SYSTEMS:
        for condition in CONDITIONS.values():
            table = np.stack([counts[(system, condition, lang)] for lang in LANGS])
            add_test("A_language", f"{system}/{condition}: MFN x language", table)
    for system in SYSTEMS:
        for lang in LANGS:
            table = np.stack(
                [counts[(system, cond, lang)] for cond in CONDITIONS.values()]
            )
            add_test("B_condition", f"{system}/{lang}: MFN x condition", table)
    for condition in CONDITIONS.values():
        for lang in LANGS:
            table = np.stack([counts[(s, condition, lang)] for s in SYSTEMS])
            add_test("C_system", f"{condition}/{lang}: MFN x system", table)

    for family in sorted({t["family"] for t in tests}):
        family_tests = [t for t in tests if t["family"] == family]
        adjusted = holm([t["p_raw"] for t in family_tests])
        for test, p_adj in zip(family_tests, adjusted):
            test["p_holm"] = p_adj

    with open(OUT_DIR / "chi2_tests.csv", "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tests[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(tests)

    # 3. Markdown summary
    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    lines = [
        "# Statistical significance summary",
        "",
        f"Bootstrap: B={B:,}, multinomial resampling, percentile 95% CI, seed={SEED}.",
        "Chi-square on M/F/N counts (scipy chi2_contingency; categories empty in every",
        "group are dropped, and the Yates correction applies to 2x2 tables); Cramer's V",
        "effect size; Holm correction per family. Sample sizes reach 134,400, so",
        "p-values are close to zero for most tests and Cramer's V is the quantity to",
        "interpret.",
        "",
        "## NGB / GPS / GAR with 95% CI",
        "",
        "| system | condition | lang | n | NGB [95% CI] | GPS [95% CI] | GAR [95% CI] |",
        "|---|---|---|---:|---|---|---|",
    ]
    for r in ci_rows:
        lines.append(
            f"| {r['system']} | {r['condition']} | {r['language']} | {r['n']:,} "
            f"| {r['NGB']:+.3f} [{r['NGB_lo']:+.3f}, {r['NGB_hi']:+.3f}] "
            f"| {r['GPS']:+.3f} [{r['GPS_lo']:+.3f}, {r['GPS_hi']:+.3f}] "
            f"| {r['GAR']:.3f} [{r['GAR_lo']:.3f}, {r['GAR_hi']:.3f}] |"
        )
    lines += [
        "",
        "## Chi-square tests",
        "",
        "| family | test | n | dof | chi2 | p (Holm) | Cramer's V |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for t in tests:
        lines.append(
            f"| {t['family']} | {t['test']} | {t['n']:,} | {t['dof']} "
            f"| {t['chi2']:,.2f} | {fmt_p(t['p_holm'])} | {t['cramers_v']:.4f} |"
        )
    lines.append("")
    (OUT_DIR / "significance_summary.md").write_text("\n".join(lines), encoding="utf-8",
                                                     newline="\n")
    print(f"-> {OUT_DIR / 'significance_summary.md'}")
    print(f"-> {OUT_DIR / 'ngb_ci.csv'}, {OUT_DIR / 'chi2_tests.csv'}")


if __name__ == "__main__":
    main()
