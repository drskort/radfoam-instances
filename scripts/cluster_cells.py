"""Fit and cache a per-cell instance clustering for a trained run.

Every downstream consumer -- the ScanNet++ and LERF harnesses, the language
table, the renderers -- reads this cache rather than refitting. Two fits of the
same checkpoint disagree on both the number of instances and their ids, so an
eval that refits is not comparable with anything else produced from that run.

    python scripts/cluster_cells.py --checkpoint output/<run> --method full

The cache lands at output/<run>/instances/clustering.pt and carries a
fingerprint of the features it was fitted on, so a stale one is detected rather
than silently reused.
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from radfoam_model.instance_cluster import (  # noqa: E402
    NOISE_ID,
    assign,
    fit_clusters,
)

def clustering_cache_path(checkpoint: Path) -> Path:
    return Path(checkpoint) / "instances" / "clustering.pt"


def feature_fingerprint(features) -> str:
    """Cheap identity check for a feature block.

    Guards against a cache fitted on a different checkpoint or iteration. Hashes
    the shape plus a deterministic stride through the rows -- hashing 265 MB on
    every run would cost more than the fit it protects.
    """
    import hashlib

    h = hashlib.sha1()
    h.update(repr(tuple(features.shape)).encode())
    step = max(1, features.shape[0] // 4096)
    sample = features[::step][:4096]
    h.update(np.ascontiguousarray(
        sample.cpu().numpy() if hasattr(sample, "cpu") else sample
    ).tobytes())
    return h.hexdigest()


def checkpoint_features(checkpoint: Path, model_file="model.pt"):
    """att_feat straight out of the checkpoint, in its ORIGINAL order.

    Not from feat.bin: the converter may have reordered points along a Z-curve,
    and fit_clusters samples via randperm, so a permuted block would give a
    different clustering. Fitting here keeps instance ids identical to the ones
    render_instances and extract_instance_language produced.

    Reads the tensor directly rather than constructing a RadFoamScene, whose
    __init__ builds a CUDA Triangulation before loading anything -- which fails
    outright on a card the extension was not compiled for.
    """
    data = torch.load(Path(checkpoint) / model_file, map_location="cpu",
                      weights_only=False)
    features = data.get("att_feat")
    if features is None:
        raise SystemExit(
            f"{checkpoint}/{model_file} has no att_feat "
            "(run trained with feat_dim = 0)."
        )
    return features.float()


def fit_or_load_clustering(checkpoint, seed, device, refit=False,
                           model_file="model.pt", method="full",
                           min_cluster_size=None, min_samples=None):
    """The clustering for a run, from cache when it is valid.

    method="full" runs cuML HDBSCAN over EVERY primitive and caches an exact
    label per cell. method="sample" is instance_cluster.fit_clusters, which
    fits on a 60k subsample -- 1.5% of a 4M-point cloud -- and hands the rest to
    nearest-centroid. The full fit is the default because the viewer's whole
    point is selecting and isolating objects, and the sampled fit loses small
    ones outright: an object holding 0.1% of the cells contributes ~60 sampled
    points against min_cluster_size=32.

    Returns (clustering, labels_or_None, features). `labels` is per-cell and in
    the CHECKPOINT's point order.
    """
    from radfoam_model.instance_cluster import Clustering

    cache = clustering_cache_path(checkpoint)
    features = checkpoint_features(checkpoint, model_file)
    fingerprint = feature_fingerprint(features)

    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }

    if cache.exists() and not refit:
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        matches = (
            blob.get("fingerprint") == fingerprint
            and blob.get("seed") == seed
            and blob.get("method") == method
            and blob.get("params") == params
        )
        if matches:
            labels = blob.get("labels")
            print(f"clustering cached  {blob['n_clusters']} clusters, "
                  f"method={method}, "
                  f"{'exact labels' if labels is not None else 'centroids only'} "
                  f"({cache})")
            return Clustering(
                mean=blob["mean"], basis=blob["basis"], lo=blob["lo"],
                hi=blob["hi"], centroids=blob["centroids"],
                colours=blob["colours"], noise_fraction=blob["noise_fraction"],
            ), labels, features
        print(f"clustering cache stale ({cache}), refitting", flush=True)

    t0 = time.time()
    labels = None
    if method == "full":
        from radfoam_model.instance_cluster import fit_clusters_full

        kwargs = {k: v for k, v in params.items() if v is not None}
        labels, clustering = fit_clusters_full(
            features.to(device), seed=seed, **kwargs
        )
        labels = labels.cpu().to(torch.int32)
    elif method == "sample":
        clustering = fit_clusters(features.to(device), seed=seed)
    else:
        raise SystemExit(f"unknown clustering method {method!r}")

    print(f"clustering fitted  {clustering.n_clusters} clusters, "
          f"{100 * clustering.noise_fraction:.1f}% noise, method={method} "
          f"({time.time() - t0:.1f}s)", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": clustering.mean.cpu(),
        "basis": clustering.basis.cpu(),
        "lo": clustering.lo.cpu(),
        "hi": clustering.hi.cpu(),
        "centroids": clustering.centroids.cpu(),
        "colours": clustering.colours,
        "noise_fraction": clustering.noise_fraction,
        "n_clusters": clustering.n_clusters,
        # Exact per-cell labels, in the checkpoint's point order. This is the
        # expensive artefact -- cuML HDBSCAN over 4M points -- and the reason
        # the cache is worth having at all.
        "labels": labels,
        "method": method,
        "params": params,
        "seed": seed,
        "fingerprint": fingerprint,
        "model_file": model_file,
        "featDim": int(features.shape[1]),
        "pointCount": int(features.shape[0]),
    }, cache)
    print(f"cached -> {cache}  ({cache.stat().st_size / 1e6:.0f} MB)")
    return clustering, labels, features


def do_cluster(args):
    """Fit and cache a clustering for a run directory."""
    device = torch.device(args.device)
    clustering, labels, features = fit_or_load_clustering(
        Path(args.checkpoint), args.seed, device, args.refit, args.model,
        args.method, args.min_cluster_size, args.min_samples,
    )
    detail = ""
    if labels is not None:
        assigned = int((labels >= 0).sum())
        detail = (f", {assigned:,}/{labels.numel():,} cells assigned "
                  f"({100 * assigned / labels.numel():.1f}%)")
    print(f"{args.checkpoint}: {clustering.n_clusters} clusters over "
          f"{features.shape[0]:,} points{detail}")



def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="model.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refit", action="store_true",
                   help="Ignore an existing cache and fit again.")
    p.add_argument("--method", default="full", choices=["full", "sample"],
                   help="full fits every cell (cuML, exact per-cell labels). "
                        "sample fits a 60k subsample and assigns the rest by "
                        "nearest centroid, which loses small objects.")
    p.add_argument("--min-cluster-size", type=int, default=None)
    p.add_argument("--min-samples", type=int, default=None)
    do_cluster(p.parse_args())


if __name__ == "__main__":
    main()
