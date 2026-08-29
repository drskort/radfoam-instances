"""Regenerate the README's result figures from the committed JSONs.

Every figure here is drawn from results/, so a reader can redraw them and get
the same picture, and a wrong number in the README shows up as a wrong bar.

    python scripts/make_figures.py            # writes assets/figures/*.png
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
OUT = REPO / "assets" / "figures"
INK = "#1b1b1b"
ACCENT = "#2f6f9f"
MUTED = "#b0b0b0"


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)


def fig_official_vs_ours():
    d = json.loads((RESULTS / "scannetpp_official.json").read_text())
    scenes = sorted(d["per_scene"])
    ours = np.array([d["per_scene"][s]["ours"][0] for s in scenes])
    off = np.array([d["per_scene"][s]["official"][0] for s in scenes])
    x = np.arange(len(scenes))
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    ax.bar(x - 0.2, ours, 0.4, label="this repo's scorer", color=MUTED)
    ax.bar(x + 0.2, off, 0.4, label="official ScanNet++ scorer", color=ACCENT)
    ax.set_xticks(x); ax.set_xticklabels([s[:6] for s in scenes], rotation=0)
    ax.set_ylabel("AP"); style(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Same predictions, two scorers: the reimplementation "
                 "understates AP by 6.2 on average", fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "official_vs_ours.png"); plt.close(fig)


def fig_tie_order():
    rows = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("tie_order/*.json"))]
    scenes = [r["scene"][:6] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=200)
    for i, r in enumerate(rows):
        a = r["AP"]
        ax.vlines(i, a["min"], a["max"], color=MUTED, lw=6, alpha=0.6)
        ax.plot([i], [a["mean"]], "o", color=ACCENT, ms=5, zorder=3)
        ax.plot([i], [a["as_emitted"]], "D", color="#c2482d", ms=5, zorder=4)
    ax.plot([], [], "o", color=ACCENT, label="mean over 100 permutations")
    ax.plot([], [], "D", color="#c2482d", label="the order actually reported")
    ax.plot([], [], lw=6, color=MUTED, alpha=0.6, label="min–max over permutations")
    ax.set_xticks(range(len(scenes))); ax.set_xticklabels(scenes)
    ax.set_ylabel("AP"); style(ax)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)   # headroom so the key clears the bars
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=3,
              handletextpad=0.5, columnspacing=1.4)
    ax.set_title("Uniform confidences make AP depend on arbitrary cluster order",
                 fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "tie_order.png"); plt.close(fig)


def fig_ablation():
    root = RESULTS / "ablation" / "guided_geometry"
    order = ["figurines", "ramen", "teatime"]
    geo, nogeo = [], []
    for s in order:
        geo.append(100 * json.loads((root / f"{s}_geo.json").read_text())["miou"])
        nogeo.append(100 * json.loads((root / f"{s}_nogeo.json").read_text())["miou"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=200)
    ax.bar(x - 0.2, nogeo, 0.4, label="features only", color=MUTED)
    ax.bar(x + 0.2, geo, 0.4, label="+ geometry guidance", color=ACCENT)
    for i, (a, b) in enumerate(zip(nogeo, geo)):
        ax.annotate(f"+{b - a:.1f}", (i + 0.2, b), ha="center", va="bottom",
                    fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("LERF-Mask mIoU"); ax.set_ylim(0, 118); style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2,
              columnspacing=1.6)
    ax.set_title("Letting the instance gradient move sites and densities",
                 fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "guided_geometry.png"); plt.close(fig)


def main():
    # No flags: any argv is a mistake, and running it silently rewrites the
    # committed PNGs and dirties the tree.
    if len(sys.argv) > 1:
        print(__doc__)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    fig_official_vs_ours()
    fig_tie_order()
    fig_ablation()
    for p in sorted(OUT.glob("*.png")):
        print(f"  wrote {p.relative_to(REPO)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
