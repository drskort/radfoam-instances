"""Score a trained ScanNet++ scene on 3D instance segmentation.

    MODEL=model_020000.pt sbatch scripts/eval_scannetpp_slurm.sh output/snpp_7b6477cb95

Reports AP / AP50 / AP25 against points of the aligned mesh, which is the
protocol OpenSplat3D uses (19.2 / 37.3 / 56.2 class-agnostic, or 24.5 / 41.7 /
57.1 once their graph smoothing is applied). No smoothing is applied here --
whether that gap is needed at all is the thing being tested.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from configs import *  # noqa: F401,F403
from radfoam_model.instance_cluster import NOISE_ID, load_cached_clustering
from radfoam_model.scannetpp_eval import (
    assign_by_containment,
    average_precision,
    load_gt_instances,
    predictions_from_labels,
)

from extract_instance_language import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--min-vertices", type=int, default=100)
    parser.add_argument("--score", default="uniform", choices=["uniform", "size"],
                        help="uniform matches OpenSplat3D's class-agnostic "
                             "export, which writes confidence 1.0 for every "
                             "prediction. size is for diagnosis only.")
    parser.add_argument("--all-classes", action="store_true",
                        help="Score against every annotated instance, not just "
                             "the 83 instance-benchmark classes. Includes wall "
                             "and floor, which inflates the numbers.")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, dataset_args = load_model(args.checkpoint, device, args.model)
    scene = dataset_args.scene

    clustering, labels = load_cached_clustering(args.checkpoint, model.att_feat)
    if clustering is None or labels is None:
        raise SystemExit("no cached clustering; run `foamviz.py cluster "
                         "--method full` on this checkpoint first")
    labels = labels.cpu().numpy()
    print(f"{clustering.n_clusters} instances "
          f"({100 * clustering.noise_fraction:.1f}% unassigned)", flush=True)

    vertices, gt = load_gt_instances(scene, not args.all_classes)
    print(f"{len(vertices):,} mesh vertices, {len(gt)} ground-truth instances",
          flush=True)

    sites = model.primal_points.detach().cpu().numpy()
    vertex_labels = assign_by_containment(vertices, sites, labels, device)
    covered = 100 * (vertex_labels != NOISE_ID).mean()
    print(f"{covered:.1f}% of mesh vertices land in a labelled cell", flush=True)

    predictions = predictions_from_labels(
        vertex_labels, NOISE_ID, args.min_vertices, args.score)
    metrics = average_precision(predictions, gt, len(vertices))
    metrics["scene"] = scene
    metrics["coverage"] = float(covered)
    metrics["n_clusters"] = clustering.n_clusters
    metrics["score"] = args.score

    print(f"\n{'AP':>8}{'AP50':>8}{'AP25':>8}{'pred':>7}{'gt':>5}")
    print(f"{100*metrics['AP']:>8.2f}{100*metrics['AP50']:>8.2f}"
          f"{100*metrics['AP25']:>8.2f}{metrics['n_pred']:>7}{metrics['n_gt']:>5}")
    print("\nOpenSplat3D, 50-scene mean: 19.2 / 37.3 / 56.2 "
          "(24.5 / 41.7 / 57.1 with graph smoothing)")

    out = (Path(args.checkpoint)
           / f"scannetpp_inst_{Path(args.model).stem}.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
