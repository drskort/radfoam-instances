"""How much of the ScanNet++ AP is decided by arbitrary tie order?

`--score uniform` gives every prediction confidence 1.0, matching OpenSplat3D's
class-agnostic export. average_precision then ranks with a stable sort, so the
precision-recall ordering is whatever order the clusters happened to come out
in -- which is arbitrary, and differs between clustering methods.

That makes AP an area under a curve whose x-ordering carries no information.
This script bounds the consequence: build the predictions exactly as
eval_scannetpp.py does, then re-score them under many random permutations and
report the spread. AP50 and AP25 are reported alongside because a threshold at
which every prediction either matches or does not should be far less sensitive.

    python scripts/tie_order_sensitivity.py --checkpoint output/<run> \
        --model model_020000.pt --trials 100
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import numpy as np
import torch

from configs import *  # noqa: F401,F403
from radfoam_model.checkpoint import load_model  # noqa: E402
from radfoam_model.instance_cluster import NOISE_ID  # noqa: E402
from radfoam_model.scannetpp_eval import (  # noqa: E402
    assign_by_containment,
    average_precision,
    fill_noise_labels,
    load_gt_instances,
    predictions_from_labels,
    split_connected,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="model_020000.pt")
    ap.add_argument("--min-cluster-size", type=int, default=512)
    ap.add_argument("--min-samples", type=int, default=16)
    ap.add_argument("--min-vertices", type=int, default=100)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    model, _, dataset_args = load_model(args.checkpoint, device, args.model)
    scene = dataset_args.scene

    from radfoam_model.instance_cluster import fit_clusters_full
    labels, clustering = fit_clusters_full(
        model.att_feat, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples)
    print(f"{scene}: {clustering.n_clusters} clusters, "
          f"{100 * clustering.noise_fraction:.1f}% noise", flush=True)

    labels = fill_noise_labels(labels, model.att_feat.detach(),
                               clustering.centroids)
    labels = labels.cpu().numpy()

    from radfoam_model.instance_graph import undirected_edges
    edges = undirected_edges(model.point_adjacency,
                             model.point_adjacency_offsets).cpu().numpy()
    labels = split_connected(labels, edges, min_size=args.min_vertices,
                             noise_id=NOISE_ID)

    vertices, gt_instances = load_gt_instances(scene)
    # void = every vertex outside the benchmark instances (wall, floor, and the
    # annotated objects whose class is not one of the 83). Built exactly as
    # eval_scannetpp.py does, so the comparison is against the same input.
    void = np.ones(len(vertices), dtype=bool)
    for _, idx in gt_instances:
        void[idx] = False
    vertex_labels = assign_by_containment(vertices, model, labels, device=device)
    predictions = predictions_from_labels(
        vertex_labels, noise_id=NOISE_ID, min_vertices=args.min_vertices,
        score="uniform")
    print(f"{len(predictions)} predictions, {len(gt_instances)} gt, "
          f"{args.trials} permutations", flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    for trial in range(args.trials):
        order = np.arange(len(predictions)) if trial == 0 else \
            rng.permutation(len(predictions))
        permuted = [predictions[i] for i in order]
        m = average_precision(permuted, gt_instances, len(vertices), void=void)
        rows.append((100 * m["AP"], 100 * m["AP50"], 100 * m["AP25"]))
        if trial == 0:
            print(f"  as-emitted: AP {rows[0][0]:.2f}  AP50 {rows[0][1]:.2f}  "
                  f"AP25 {rows[0][2]:.2f}", flush=True)

    arr = np.array(rows)
    print(f"\n{'metric':<7}{'as-emitted':>12}{'mean':>9}{'sd':>8}"
          f"{'min':>8}{'max':>8}{'range':>8}")
    out = {"scene": scene, "trials": args.trials,
           "n_pred": len(predictions), "n_gt": len(gt_instances)}
    for j, name in enumerate(("AP", "AP50", "AP25")):
        col = arr[:, j]
        print(f"{name:<7}{rows[0][j]:>12.2f}{col.mean():>9.2f}{col.std(ddof=1):>8.2f}"
              f"{col.min():>8.2f}{col.max():>8.2f}{col.max() - col.min():>8.2f}")
        out[name] = dict(as_emitted=rows[0][j], mean=float(col.mean()),
                         sd=float(col.std(ddof=1)), min=float(col.min()),
                         max=float(col.max()))
    path = Path(args.checkpoint) / f"tie_order_{Path(args.model).stem}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
