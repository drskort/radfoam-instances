"""Rebuild the README's ScanNet++ tables from the committed eval outputs.

Every number in the README's ScanNet++ and clustering sections comes from
results/scannetpp/<scene>/<config>.json, which eval_scannetpp.py writes. Run
this to check the tables against the files rather than trusting the markdown.

    python scripts/summarize_results.py
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "results" / "scannetpp"
HEADLINE = "hdbscan_m512_fill_split"


def label(tag):
    """Human-readable name for a config filename stem."""
    if tag.startswith("hdbscan"):
        size = re.search(r"_m(\d+)", tag).group(1)
        name = f"HDBSCAN m={size}"
    elif tag.startswith("multicut_tau_sam"):
        m = re.search(r"_m(\d+)_w([0-9.]+)", tag)
        name = f"multicut+SAM m={m.group(1)} w={m.group(2)}"
    else:
        m = re.search(r"tau([0-9.]+)_m(\d+)", tag)
        name = f"multicut tau={m.group(1)} m={m.group(2)}"
    return name + ("" if "_fill_" in tag else " [no-fill]")


def load(root):
    runs = defaultdict(dict)
    for path in sorted(root.glob("*/*.json")):
        d = json.loads(path.read_text())
        runs[path.stem][path.parent.name] = d
    return runs


def table(runs, scenes, tags):
    print(f"{'config':<30}{'AP':>7}{'AP50':>8}{'AP25':>8}   per-scene AP")
    for tag in tags:
        v = runs[tag]
        ap = np.array([100 * v[s]["AP"] for s in scenes])
        print(f"{label(tag):<30}{ap.mean():>7.2f}"
              f"{np.mean([100 * v[s]['AP50'] for s in scenes]):>8.2f}"
              f"{np.mean([100 * v[s]['AP25'] for s in scenes]):>8.2f}   "
              + " ".join(f"{x:5.1f}" for x in ap))


def paired(runs, scenes, a, b):
    d = np.array([100 * (runs[a][s]["AP"] - runs[b][s]["AP"]) for s in scenes])
    return d.mean(), int((d > 0).sum()), d.std(ddof=1)


def lerf_mask(root):
    """Rebuild the LERF-Mask table from results/lerf_mask/<scene>.json."""
    rows = []
    order = {"figurines": 0, "ramen": 1, "teatime": 2}
    for path in sorted(root.glob("*.json")):
        d = json.loads(path.read_text())
        rows.append((d["scene"], 100 * d["miou"], 100 * d["mbiou"],
                     d.get("clustering"), d.get("readout")))
    rows.sort(key=lambda r: order.get(r[0], 99))
    if not rows:
        return
    print("\n=== LERF-Mask, grounded protocol ===")
    print(f"  {'scene':<12}{'mIoU':>8}{'mBIoU':>8}   config")
    for s, a, b, c, r in rows:
        print(f"  {s:<12}{a:>8.2f}{b:>8.2f}   {c}/{r}")
    print(f"  {'MEAN':<12}{np.mean([r[1] for r in rows]):>8.2f}"
          f"{np.mean([r[2] for r in rows]):>8.2f}")
    print("  Gaussian Grouping 72.8/67.6   ILGS 80.5/76.0   OpenSplat3D 84.0/-")


def ablation(root):
    """Rebuild the guided-geometry ablation from both committed arms."""
    arms = {}
    for path in sorted(root.glob("*_*.json")):
        scene, _, arm = path.stem.rpartition("_")
        d = json.loads(path.read_text())
        arms.setdefault(scene, {})[arm] = (100 * d["miou"], 100 * d["mbiou"])
    paired = {s: v for s, v in arms.items() if {"geo", "nogeo"} <= set(v)}
    if not paired:
        return
    order = {"figurines": 0, "ramen": 1, "teatime": 2}
    scenes = sorted(paired, key=lambda s: order.get(s, 99))
    print("\n=== ablation: instance gradients also shaping density ===")
    print(f"  {'scene':<12}{'with':>8}{'without':>9}{'d mIoU':>9}{'d mBIoU':>10}")
    for s in scenes:
        g, n = paired[s]["geo"], paired[s]["nogeo"]
        print(f"  {s:<12}{g[0]:>8.2f}{n[0]:>9.2f}{g[0]-n[0]:>+9.2f}"
              f"{g[1]-n[1]:>+10.2f}")
    dm = np.mean([paired[s]["geo"][0] - paired[s]["nogeo"][0] for s in scenes])
    db = np.mean([paired[s]["geo"][1] - paired[s]["nogeo"][1] for s in scenes])
    wins = sum(paired[s]["geo"][0] > paired[s]["nogeo"][0] for s in scenes)
    print(f"  {'MEAN':<12}"
          f"{np.mean([paired[s]['geo'][0] for s in scenes]):>8.2f}"
          f"{np.mean([paired[s]['nogeo'][0] for s in scenes]):>9.2f}"
          f"{dm:>+9.2f}{db:>+10.2f}   wins {wins}/{len(scenes)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = ap.parse_args()

    runs = load(args.results)
    if not runs:
        raise SystemExit(f"no results under {args.results}")
    scenes = sorted(runs[HEADLINE])
    complete = [t for t, v in runs.items() if set(v) == set(scenes)]
    print(f"{len(scenes)} scenes, {len(complete)}/{len(runs)} configs complete "
          f"on all of them\n")

    print("=== headline (fill-noise + connected-component split) ===")
    table(runs, scenes, sorted(t for t in complete if "_fill_" in t))

    nofill = sorted(t for t in complete if "_fill_" not in t)
    if nofill:
        print("\n=== without fill-noise ===")
        table(runs, scenes, nofill)

    print("\n=== paired comparisons (same scene, same checkpoint) ===")
    pairs = [
        ("multicut_tau0.3_m512_fill_split", HEADLINE),
        ("multicut_tau_sam_tau0.3_m1024_w1.0_fill_split",
         "multicut_tau_sam_tau0.3_m1024_w0.0_fill_split"),
        ("multicut_tau0.3_m512_split", "hdbscan_m512_split"),
        (HEADLINE, "hdbscan_m512_split"),
        ("multicut_tau0.3_m512_fill_split", "multicut_tau0.3_m512_split"),
    ]
    for a, b in pairs:
        if a not in runs or b not in runs:
            continue
        mean, wins, sd = paired(runs, scenes, a, b)
        print(f"  {label(a):<30} vs {label(b):<30} "
              f"{mean:+6.2f} AP  wins {wins}/{len(scenes)}  sd {sd:5.2f}")

    # Every metric's sem, not just AP's. Reporting the spread on the metric you
    # lose and omitting it on the ones you win is the easiest way to mislead
    # with a table like this.
    v = runs[HEADLINE]
    print("\n=== headline dispersion over scenes ===")
    print(f"  {'metric':<8}{'mean':>8}{'sd':>8}{'sem':>8}{'min':>7}{'max':>7}")
    for k in ("AP", "AP50", "AP25"):
        a = np.array([100 * v[s][k] for s in scenes])
        print(f"  {k:<8}{a.mean():>8.2f}{a.std(ddof=1):>8.2f}"
              f"{a.std(ddof=1) / np.sqrt(len(scenes)):>8.2f}"
              f"{a.min():>7.1f}{a.max():>7.1f}")
    ap = np.mean([100 * v[s]["AP"] for s in scenes])
    ap25 = np.mean([100 * v[s]["AP25"] for s in scenes])
    print(f"  AP25/AP ratio {ap25 / ap:.2f}   "
          f"(OpenSplat3D 2.93, with their denoising 2.33)")

    print("\n=== ground truth actually scored ===")
    total = sum(v[s]["n_gt"] for s in scenes)
    print("  " + "  ".join(f"{s[:6]}:{v[s]['n_gt']}" for s in scenes))
    print(f"  {total} instances scored, after the 83-class benchmark "
          f"restriction and the 100-vertex minimum")

    lerf = args.results.parent / "lerf_mask"
    if lerf.is_dir():
        lerf_mask(lerf)

    abl = args.results.parent / "ablation" / "guided_geometry"
    if abl.is_dir():
        ablation(abl)


if __name__ == "__main__":
    main()
