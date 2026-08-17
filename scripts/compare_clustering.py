"""Compare feature-space clustering against Delaunay-graph partitioning.

HDBSCAN sees only feature vectors; the graph methods also see which cells touch.
The question this answers is whether that changes the decomposition at all, and
in which direction -- fewer/larger instances, less noise, or just different.

    srun -p 3090-lo --gres=gpu:1 --time=00:40:00 \
        .venv/bin/python scripts/compare_clustering.py \
            --checkpoint output/garden_inst_geo --model model_006000.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from configs import *  # noqa: F401,F403
from radfoam_model.instance_cluster import fit_clusters
from radfoam_model.instance_graph import (
    NOISE_ID,
    edge_dissimilarity,
    fit_graph_clusters,
    undirected_edges,
)
from radfoam_model.scene import RadFoamScene


def load_model(checkpoint, device, model_file):
    import configargparse

    config = Path(checkpoint) / "config.yaml"
    parser = configargparse.ArgParser(default_config_files=[str(config)])
    parser.add_argument("-c", "--config", is_config_file=True)
    model_params = ModelParams(parser)          # noqa: F405
    PipelineParams(parser)                      # noqa: F405
    OptimizationParams(parser)                  # noqa: F405
    DatasetParams(parser)                       # noqa: F405
    args = parser.parse_args(["-c", str(config)])
    model = RadFoamScene(args=model_params.extract(args), device=device)
    model.load_pt(str(Path(checkpoint) / model_file))
    return model


def summarise(name, n_clusters, noise, labels, seconds):
    if labels is not None:
        sizes = np.bincount(labels[labels >= NOISE_ID + 1].cpu().numpy()) \
            if n_clusters else np.array([])
        biggest = sizes.max() if sizes.size else 0
        median = int(np.median(sizes)) if sizes.size else 0
    else:
        biggest = median = 0
    print(f"{name:<34} {n_clusters:>6} {100 * noise:>7.1f}% "
          f"{median:>9} {biggest:>10} {seconds:>7.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--min-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = load_model(args.checkpoint, device, args.model)
    features = model.att_feat.detach().float()
    print(f"{features.shape[0]:,} primitives, feat_dim={features.shape[1]}")

    # load_pt restores point_adjacency from the checkpoint; rebuilding the
    # triangulation would need the training optimizer, which does not exist here.
    edges = undirected_edges(model.point_adjacency, model.point_adjacency_offsets)
    distance = edge_dissimilarity(features, edges)
    print(f"{edges.shape[0]:,} undirected Delaunay edges "
          f"(mean degree {2 * edges.shape[0] / features.shape[0]:.1f})")
    quantiles = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=device)
    euclid = edge_dissimilarity(features, edges, "euclidean")
    print("edge euclidean distance quantiles "
          + "  ".join(f"{q:.2f}:{v:.3f}" for q, v in
                      zip([0.1, 0.25, 0.5, 0.75, 0.9],
                          torch.quantile(euclid[:1_000_000],
                                         torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9],
                                                      device=device)).tolist())))
    print("edge cosine distance quantiles "
          + "  ".join(f"{q:.2f}:{v:.4f}" for q, v in
                      zip(quantiles.tolist(),
                          torch.quantile(distance[:1_000_000], quantiles).tolist())))

    print(f"\n{'method':<34} {'inst':>6} {'noise':>8} "
          f"{'median':>9} {'largest':>10} {'time':>8}")

    start = time.time()
    hdb = fit_clusters(features)
    summarise("HDBSCAN (feature space)", hdb.n_clusters, hdb.noise_fraction,
              None, time.time() - start)

    for tau in (0.05, 0.1, 0.2, 0.4):
        start = time.time()
        result = fit_graph_clusters(
            features, model.point_adjacency, model.point_adjacency_offsets,
            method="threshold", tau=tau, min_size=args.min_size,
        )
        summarise(f"graph: threshold tau={tau}", result.n_clusters,
                  result.noise_fraction, result.labels, time.time() - start)

    # tau = sqrt(gamma) from the contrastive loss margin, plus a sweep around it.
    for tau in (0.5, 1.0, 1.5, 2.0):
        start = time.time()
        result = fit_graph_clusters(
            features, model.point_adjacency, model.point_adjacency_offsets,
            method="multicut", tau=tau, metric="euclidean",
            min_size=args.min_size,
        )
        summarise(f"graph: MULTICUT tau={tau}", result.n_clusters,
                  result.noise_fraction, result.labels, time.time() - start)

    for k in (0.005, 0.02, 0.1):
        start = time.time()
        result = fit_graph_clusters(
            features, model.point_adjacency, model.point_adjacency_offsets,
            method="felzenszwalb", k=k, min_size=args.min_size,
        )
        summarise(f"graph: felzenszwalb k={k}", result.n_clusters,
                  result.noise_fraction, result.labels, time.time() - start)


if __name__ == "__main__":
    main()
