"""Score a trained ScanNet++ scene on 3D instance segmentation.

    MODEL=model_020000.pt sbatch scripts/eval_scannetpp_slurm.sh output/snpp_7b6477cb95

Reports AP / AP50 / AP25 against points of the aligned mesh, which is the
protocol OpenSplat3D uses (19.2 / 37.3 / 56.2 class-agnostic, or 24.5 / 41.7 /
57.1 once their graph smoothing is applied). No smoothing is applied here --
whether that gap is needed at all is the thing being tested.
"""


import sys
from pathlib import Path

# Run directly from a clone: the eval scripts live in scripts/ but import
# configs/, radfoam_model/ and data_loader/ from the repo root, which pip
# does not install (setup.cfg packages only src/). The Slurm wrappers set
# PYTHONPATH; this makes the plain commands in the README work too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    dbscan_denoise,
    split_connected,
    fill_noise_labels,
    load_gt_instances,
    predictions_from_labels,
)

from extract_instance_language import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--min-vertices", type=int, default=100)
    parser.add_argument("--fill-noise", action="store_true",
                        help="Give abstaining cells their nearest cluster. "
                             "HDBSCAN leaves ~70%% of cells unlabelled, which "
                             "costs nothing on a 2D readout that skips them "
                             "and is a guaranteed miss here.")
    parser.add_argument("--clustering", default="cached",
                        choices=["cached", "hdbscan", "multicut", "multicut_sam",
                                 "multicut_tau_sam"],
                        help="cached = instances/clustering.pt. hdbscan = "
                             "refit at --min-cluster-size. multicut = GAEC on "
                             "the Delaunay graph at --tau.")
    parser.add_argument("--min-cluster-size", type=int, default=32)
    parser.add_argument("--min-samples", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--occupancy", action="store_true",
                        help="multicut_sam: add edge solidity as a third "
                             "log-odds term.")
    parser.add_argument("--sam-views", type=int, default=1000)
    parser.add_argument("--sam-weight", type=float, default=1.0,
                        help="multicut_tau_sam: how far a fully-agreeing edge "
                             "shifts the base weight, in units of tau.")
    parser.add_argument("--split-connected", action="store_true",
                        help="Split instances into spatially connected "
                             "components using the Delaunay adjacency. The "
                             "exact form of OpenSplat3D's DBSCAN denoising.")
    parser.add_argument("--dbscan", action="store_true",
                        help="OpenSplat3D's post-processing: split each "
                             "instance into spatially connected components. "
                             "Worth +5.3 AP to them.")
    parser.add_argument("--dbscan-eps", type=float, default=0.5)
    parser.add_argument("--any-cell", action="store_true",
                        help="Assign vertices to the nearest cell of any "
                             "density, not just cells that render. Reproduces "
                             "the original behaviour.")
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

    if args.clustering == "hdbscan":
        from radfoam_model.instance_cluster import fit_clusters_full
        labels, clustering = fit_clusters_full(
            model.att_feat, min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples)
        print(f"hdbscan min_cluster_size={args.min_cluster_size}: "
              f"{clustering.n_clusters} clusters, "
              f"{100*clustering.noise_fraction:.1f}% noise", flush=True)
    elif args.clustering in ("multicut_sam", "multicut_tau_sam"):
        # The one configuration whose edge weights use information the feature
        # clustering cannot see: how often two cells land in the same SAM mask.
        from data_loader import DataHandler
        from radfoam_model.instance_graph import (
            clustering_from_labels, multicut_sam, sam_edge_counts,
            undirected_edges)
        from radfoam_model.instance_masks import resolve_mask_dir

        mask_dir = resolve_mask_dir(scene)
        if mask_dir is None:
            raise SystemExit(f"no SAM masks for {scene}")
        data = DataHandler(dataset_args, rays_per_batch=0, device=device)
        data.reload(split="train", downsample=min(dataset_args.downsample))
        edges = undirected_edges(model.point_adjacency,
                                 model.point_adjacency_offsets)
        step = max(1, len(data.image_names) // args.sam_views)
        agree, disagree = sam_edge_counts(
            model, data, edges, mask_dir, data.image_names[::step],
            device, report=True)
        if args.clustering == "multicut_tau_sam":
            from radfoam_model.instance_graph import multicut_tau_sam
            labels, _, _ = multicut_tau_sam(
                model.att_feat, edges, agree, disagree, tau=args.tau,
                min_size=args.min_cluster_size, sam_weight=args.sam_weight,
                report=True)
        else:
            labels, _, _ = multicut_sam(
                model.att_feat, edges, agree, disagree,
                min_size=args.min_cluster_size, metric="euclidean", report=True,
                occupancy=(model.get_primal_density().detach().float()
                           .reshape(-1)[edges].min(dim=1).values.cpu().numpy()
                           if args.occupancy else None))
        clustering = clustering_from_labels(model.att_feat, labels)
        print(f"multicut_sam: {clustering.n_clusters} clusters, "
              f"{100*clustering.noise_fraction:.1f}% noise", flush=True)
    elif args.clustering == "multicut":
        from radfoam_model.instance_graph import (
            clustering_from_labels, fit_graph_clusters)
        result = fit_graph_clusters(
            model.att_feat, model.point_adjacency,
            model.point_adjacency_offsets, method="multicut",
            min_size=args.min_cluster_size, tau=args.tau, metric="euclidean")
        labels = result.labels
        clustering = clustering_from_labels(model.att_feat, labels)
        print(f"multicut tau={args.tau}: {clustering.n_clusters} clusters, "
              f"{100*clustering.noise_fraction:.1f}% noise", flush=True)
    else:
        clustering, labels = load_cached_clustering(args.checkpoint,
                                                    model.att_feat)
        if clustering is None or labels is None:
            raise SystemExit("no cached clustering; run `foamviz.py cluster "
                             "--method full` on this checkpoint first")
    if args.fill_noise:
        before = (labels == NOISE_ID).float().mean()
        labels = fill_noise_labels(labels, model.att_feat.detach(),
                                   clustering.centroids)
        print(f"filled abstaining cells: {100*before:.1f}% -> "
              f"{100*(labels == NOISE_ID).float().mean():.1f}% unlabelled",
              flush=True)
    labels = labels.cpu().numpy()
    if args.split_connected:
        from radfoam_model.instance_graph import undirected_edges
        edges = undirected_edges(model.point_adjacency,
                                 model.point_adjacency_offsets).cpu().numpy()
        labels = split_connected(labels, edges, min_size=args.min_vertices,
                                 noise_id=NOISE_ID, report=True)
    print(f"{clustering.n_clusters} instances "
          f"({100 * clustering.noise_fraction:.1f}% unassigned)", flush=True)

    vertices, gt = load_gt_instances(scene, not args.all_classes)
    print(f"{len(vertices):,} mesh vertices, {len(gt)} ground-truth instances",
          flush=True)

    vertex_labels = assign_by_containment(
        vertices, model, labels, device,
        min_density=None if args.any_cell else 1e-3)
    covered = 100 * (vertex_labels != NOISE_ID).mean()
    print(f"{covered:.1f}% of mesh vertices land in a labelled cell", flush=True)

    if args.dbscan:
        vertex_labels = dbscan_denoise(
            vertex_labels, vertices, eps=args.dbscan_eps,
            noise_id=NOISE_ID, report=True)
    predictions = predictions_from_labels(
        vertex_labels, NOISE_ID, args.min_vertices, args.score)
    # void = every vertex outside the benchmark instances, i.e. wall, floor,
    # and the 45 annotated objects whose class is not one of the 83.
    void = np.ones(len(vertices), dtype=bool)
    for _, idx in gt:
        void[idx] = False
    print(f"{100*void.mean():.1f}% of mesh vertices are unannotated (void)",
          flush=True)
    metrics = average_precision(predictions, gt, len(vertices), void)
    metrics["scene"] = scene
    metrics["coverage"] = float(covered)
    metrics["n_clusters"] = clustering.n_clusters
    metrics["score"] = args.score
    metrics["clustering"] = args.clustering
    metrics["min_cluster_size"] = args.min_cluster_size
    metrics["tau"] = args.tau
    metrics["fill_noise"] = args.fill_noise
    metrics["split_connected"] = args.split_connected

    print(f"\n{'AP':>8}{'AP50':>8}{'AP25':>8}{'pred':>7}{'gt':>5}")
    print(f"{100*metrics['AP']:>8.2f}{100*metrics['AP50']:>8.2f}"
          f"{100*metrics['AP25']:>8.2f}{metrics['n_pred']:>7}{metrics['n_gt']:>5}")
    print("\nOpenSplat3D, 50-scene mean: 19.2 / 37.3 / 56.2 "
          "(24.5 / 41.7 / 57.1 with graph smoothing)")

    # min_cluster_size belongs in the name for BOTH clusterings: it is the
    # granularity knob for each, and omitting it for multicut silently
    # overwrote every cell of a tau x min_size sweep but one.
    tag = (f"{args.clustering}"
           + (f"_m{args.min_cluster_size}" if args.clustering == "hdbscan"
              else f"_tau{args.tau}_m{args.min_cluster_size}"
              if args.clustering == "multicut"
              else f"_m{args.min_cluster_size}" + ("_occ" if args.occupancy else "")
              if args.clustering == "multicut_sam"
              else f"_tau{args.tau}_m{args.min_cluster_size}_w{args.sam_weight}"
              if args.clustering == "multicut_tau_sam" else "")
           + ("_fill" if args.fill_noise else "")
           + ("_split" if args.split_connected else ""))
    out = (Path(args.checkpoint)
           / f"scannetpp_inst_{Path(args.model).stem}_{tag}.json")
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
