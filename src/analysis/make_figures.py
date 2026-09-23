"""Figures 2 to 4 of the paper, drawn from results/data_summary.json.

  Fig. 2  average JSD per (system x language) combination
  Fig. 3  NGB heatmap, DeepL-KR, clothing x color
  Fig. 4  NGB heatmap, Google-KR, clothing x color

Usage: python -m src.analysis.make_figures
Output: figs/fig2_jsd_avg.png, figs/fig3_ngb_heatmap_deepl_kr.png,
        figs/fig4_ngb_heatmap_google_kr.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calculate_metrics import CLOTHING_ITEMS, COLOR_NAMES

ROOT = Path(__file__).resolve().parents[2]
DATA_SUMMARY = ROOT / "results" / "data_summary.json"
FIG_DIR = ROOT / "figs"

LANGS = ["kr", "en", "ja", "zh"]
C_DEEPL = "#0072B2"
C_GOOGLE = "#E69F00"


def fig_jsd(summary):
    pairs = summary["_jsd_pairs"]
    values = {
        system: [pairs[f"{system}_default_tr_{lang}.json__VS__{system}_color_tr_{lang}.json"]["avg_jsd"]
                 for lang in LANGS]
        for system in ("deepl", "google")
    }
    x = np.arange(len(LANGS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(3.35, 2.3), dpi=300)
    bars = [
        ax.bar(x - width / 2, values["deepl"], width, label="DeepL",
               color=C_DEEPL, edgecolor="white", linewidth=0.6),
        ax.bar(x + width / 2, values["google"], width, label="Google",
               color=C_GOOGLE, edgecolor="white", linewidth=0.6, hatch="///"),
    ]
    for group in bars:
        for b in group:
            ax.annotate(f"{b.get_height():.3f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 1.5),
                        ha="center", va="bottom", fontsize=5.5, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels([lang.upper() for lang in LANGS], fontsize=7)
    ax.set_ylabel("Average JSD", fontsize=7)
    ax.set_ylim(0, 0.092)
    ax.tick_params(axis="y", labelsize=6.5, length=2, color="#bbbbbb")
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.grid(axis="y", color="#e3e3e0", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left",
              handlelength=1.2, handletextpad=0.5, borderaxespad=0.1)
    fig.tight_layout(pad=0.4)
    out = FIG_DIR / "fig2_jsd_avg.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("->", out)


def fig_heatmap(summary, system, label, filename):
    cells = summary[f"{system}_color_tr_kr.json"]["ngb_per_clothing_color"]
    H = np.zeros((len(CLOTHING_ITEMS), len(COLOR_NAMES)))
    for i, item in enumerate(CLOTHING_ITEMS):
        for j in range(len(COLOR_NAMES)):
            H[i, j] = cells[f"{item}|{j}"]["ngb"]

    fig, ax = plt.subplots(figsize=(11.0, 5.5))
    im = ax.imshow(H, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(COLOR_NAMES)))
    ax.set_xticklabels([c.replace("_", " ") for c in COLOR_NAMES],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(CLOTHING_ITEMS)))
    ax.set_yticklabels(CLOTHING_ITEMS, fontsize=9)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f"{H[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(H[i, j]) > 0.55 else "black")
    pink = COLOR_NAMES.index("pink")
    ax.add_patch(plt.Rectangle((pink - 0.5, -0.5), 1, len(CLOTHING_ITEMS),
                               fill=False, edgecolor="#000000", linewidth=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("NGB", rotation=90, labelpad=10)
    ax.set_title(f"NGB heatmap - {label} (clothing x color)", fontsize=11)
    out = FIG_DIR / filename
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("->", out)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    fig_jsd(summary)
    fig_heatmap(summary, "deepl", "DeepL-KR", "fig3_ngb_heatmap_deepl_kr.png")
    fig_heatmap(summary, "google", "Google-KR", "fig4_ngb_heatmap_google_kr.png")


if __name__ == "__main__":
    main()
