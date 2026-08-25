"""Cluster the learned instance features into 3D instances.

Shared by the visualiser and the language-embedding extractor. It has to be
shared: if the two clustered independently, instance 7 in the video would be a
different object from instance 7 in the language table, and every query result
would point at the wrong thing.

Clustering happens on the per-primitive features of the 3D field, not on
rendered pixels -- instances live in the field, and views are projections of
them. This is also what OpenSplat3D does (HDBSCAN over Gaussian features), with
its ScanNet++ settings reproduced below.
"""

from dataclasses import dataclass

import numpy as np
import torch

HDBSCAN_MIN_CLUSTER_SIZE = 32
HDBSCAN_MIN_SAMPLES = 16
PCA_SAMPLE = 200_000
CLUSTER_SAMPLE = 60_000
NOISE_ID = -1


@dataclass
class Clustering:
    mean: torch.Tensor        # (1, D) feature-space centre
    basis: torch.Tensor       # (D, 3) PCA basis, for visualisation
    lo: torch.Tensor          # (3,) 2nd percentile of the projection
    hi: torch.Tensor          # (3,) 98th percentile
    centroids: torch.Tensor   # (K, D) cluster centres, in raw feature space
    colours: np.ndarray       # (K, 3) uint8, one stable colour per cluster
    noise_fraction: float

    @property
    def n_clusters(self):
        return int(self.centroids.shape[0])


def load_cached_clustering(checkpoint, features, seed=0, verbose=True):
    """The clustering cached by `scripts/cluster_cells.py`, if it is valid.

    Returns (Clustering, labels_or_None) or (None, None). `labels` is per-cell
    and in the checkpoint's point order; it is present only for methods that
    cluster every primitive.

    Exists so that every consumer -- the viewer export, the language table, the
    evals -- reads ONE clustering instead of each re-fitting its own. Re-fitting
    is not merely wasteful: a 60k-sample fit and a full fit disagree on both the
    number of instances and their ids, and nothing downstream would notice.
    """
    import hashlib
    from pathlib import Path

    cache = Path(checkpoint) / "instances" / "clustering.pt"
    if not cache.exists():
        return None, None

    h = hashlib.sha1()
    h.update(repr(tuple(features.shape)).encode())
    step = max(1, features.shape[0] // 4096)
    sample = features[::step][:4096]
    h.update(np.ascontiguousarray(sample.detach().float().cpu().numpy()).tobytes())
    fingerprint = h.hexdigest()

    blob = torch.load(cache, map_location="cpu", weights_only=False)
    if blob.get("fingerprint") != fingerprint or blob.get("seed") != seed:
        if verbose:
            print(f"clustering cache at {cache} does not match these features; "
                  f"ignoring it")
        return None, None

    if verbose:
        print(f"using cached clustering: {blob['n_clusters']} clusters, "
              f"method={blob.get('method')}, "
              f"{'exact labels' if blob.get('labels') is not None else 'centroids only'}")

    device = features.device
    clustering = Clustering(
        mean=blob["mean"].to(device),
        basis=blob["basis"].to(device),
        lo=blob["lo"].to(device),
        hi=blob["hi"].to(device),
        centroids=blob["centroids"].to(device),
        colours=blob["colours"],
        noise_fraction=blob["noise_fraction"],
    )
    labels = blob.get("labels")
    return clustering, (labels.to(device) if labels is not None else None)


def _stable_palette(count):
    """Golden-ratio hues: maximally separated, and cluster k always gets hue k."""
    import cv2

    count = max(count, 1)
    hues = (np.arange(count) * 0.61803398875) % 1.0
    hsv = np.stack(
        [hues * 179, np.full(count, 230.0), np.full(count, 240.0)], axis=1
    ).astype(np.uint8)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def fit_clusters(features, seed=0):
    """HDBSCAN over per-primitive features, plus a PCA basis for viewing."""
    from sklearn.cluster import HDBSCAN

    features = features.detach().float()
    mean = features.mean(dim=0, keepdim=True)
    centred = features - mean

    generator = torch.Generator(device=centred.device).manual_seed(seed)
    order = torch.randperm(centred.shape[0], generator=generator,
                           device=centred.device)
    sample = centred[order[: min(PCA_SAMPLE, centred.shape[0])]]

    _, _, basis = torch.pca_lowrank(sample, q=3)
    projected = sample @ basis
    lo = torch.quantile(projected, 0.02, dim=0)
    hi = torch.quantile(projected, 0.98, dim=0)

    subset = sample[: min(CLUSTER_SAMPLE, sample.shape[0])]
    labels = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
    ).fit(subset.cpu().numpy()).labels_

    n_clusters = int(labels.max()) + 1 if labels.size else 0
    if n_clusters:
        centroids = torch.stack([
            subset[torch.from_numpy(labels == k).to(subset.device)].mean(dim=0)
            for k in range(n_clusters)
        ]) + mean
    else:
        centroids = torch.zeros(0, features.shape[1], device=features.device)

    return Clustering(
        mean=mean,
        basis=basis,
        lo=lo,
        hi=hi,
        centroids=centroids,
        colours=_stable_palette(n_clusters),
        noise_fraction=float((labels < 0).mean()) if labels.size else 1.0,
    )


def assign(feature_map, clustering, max_distance=None):
    """Nearest-centroid instance id per element; NOISE_ID where too far.

    feature_map: (..., D). Returns a long tensor of the leading shape.
    """
    shape = feature_map.shape[:-1]
    if clustering.n_clusters == 0:
        return torch.full(shape, NOISE_ID, dtype=torch.long,
                          device=feature_map.device)

    flat = feature_map.reshape(-1, feature_map.shape[-1])
    distances = torch.cdist(flat, clustering.centroids)
    nearest = distances.argmin(dim=1)
    if max_distance is not None:
        closest = distances.gather(1, nearest[:, None]).squeeze(1)
        nearest = torch.where(
            closest > max_distance,
            torch.full_like(nearest, NOISE_ID),
            nearest,
        )
    return nearest.reshape(shape)


def to_pca_rgb(feature_map, clustering):
    """Project features through the fixed PCA basis into uint8 RGB.

    The basis follows the features onto whichever device they are on. A cached
    clustering is loaded wherever torch.load put it, which is not necessarily
    where the caller is working -- and the mismatch only shows up on the
    checkpoints whose cache happened to be written from CPU.
    """
    device = feature_map.device
    mean = clustering.mean.to(device)
    basis = clustering.basis.to(device)
    lo, hi = clustering.lo.to(device), clustering.hi.to(device)
    projected = (feature_map - mean) @ basis
    span = (hi - lo).clamp(min=1e-6)
    normalized = ((projected - lo) / span).clamp(0, 1)
    return (normalized * 255).to(torch.uint8).cpu().numpy()


def fit_clusters_full(features, min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                      min_samples=HDBSCAN_MIN_SAMPLES, seed=0,
                      cluster_selection_epsilon=0.0,
                      cluster_selection_method="eom"):
    """HDBSCAN over EVERY primitive on the GPU, returning a label per cell.

    `fit_clusters` fits on a 60k sample of what is now a 4M-point cloud -- 1.5%
    -- and then hands the rest to nearest-centroid. An object holding 0.1% of
    the cells contributes ~60 sampled points against min_cluster_size=32, so
    small objects are lost to sampling before language ever sees them. This is
    what OpenSplat3D does instead: cuML's HDBSCAN on the full set.

    Returns (labels, clustering). The labels are per-cell and exact, so they
    should be consumed through the argmax readout rather than through the
    Clustering's centroids -- the centroids are only kept so the PCA
    visualisation keeps working.
    """
    from cuml.cluster import HDBSCAN as GPUHDBSCAN

    features = features.detach().float()
    labels = GPUHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
    ).fit_predict(features.cpu().numpy().astype(np.float32))
    labels = torch.from_numpy(np.asarray(labels).astype(np.int64)).to(
        features.device
    )

    n_clusters = int(labels.max().item()) + 1 if (labels >= 0).any() else 0
    centroids = (
        torch.stack([features[labels == k].mean(dim=0) for k in range(n_clusters)])
        if n_clusters
        else torch.zeros(0, features.shape[1], device=features.device)
    )

    mean = features.mean(dim=0, keepdim=True)
    centred = features - mean
    generator = torch.Generator(device=centred.device).manual_seed(seed)
    order = torch.randperm(centred.shape[0], generator=generator,
                           device=centred.device)
    sample = centred[order[: min(PCA_SAMPLE, centred.shape[0])]]
    _, _, basis = torch.pca_lowrank(sample, q=3)
    projected = sample @ basis

    return labels, Clustering(
        mean=mean, basis=basis,
        lo=torch.quantile(projected, 0.02, dim=0),
        hi=torch.quantile(projected, 0.98, dim=0),
        centroids=centroids,
        colours=_stable_palette(n_clusters),
        noise_fraction=float((labels < 0).float().mean().item()),
    )


SH_C0 = 0.28209479177387814


def cluster_feature_matrix(features, positions=None, colours=None,
                           with_position=0.0, with_color=0.0):
    """OpenSplat3D's `get_cluster_features`: optionally glue xyz and/or base
    colour onto the learned feature before clustering.

    Weights are raw multipliers on the raw quantities, as in the reference, so
    they are not comparable between the two channels. On a LERF scene xyz has
    per-dim std ~3.0 against the feature's ~0.53, so position at weight 1.0
    contributes roughly six times the variance of all 16 feature dims combined;
    base colour at weight 1.0 contributes under a tenth. Equal-variance points
    are near 0.4 for position and 3.5 for colour.
    """
    blocks = []
    if with_color > 0 and colours is not None:
        blocks.append((colours.reshape(colours.shape[0], -1) * SH_C0 + 0.5)
                      * with_color)
    if with_position > 0 and positions is not None:
        blocks.append(positions * with_position)
    blocks.append(features)
    return torch.cat(blocks, dim=-1) if len(blocks) > 1 else features


def fit_clusters_augmented(features, positions=None, colours=None,
                           with_position=0.0, with_color=0.0,
                           min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                           min_samples=HDBSCAN_MIN_SAMPLES, seed=0):
    """Sampled HDBSCAN over augmented features, returning a label per cell.

    Assignment happens over *cells*, not pixels: with position glued on, a pixel
    has no coordinate to match against, but every cell does. So this hands back
    per-cell labels for the argmax readout rather than centroids for a
    per-pixel nearest-centroid pass.
    """
    from sklearn.cluster import HDBSCAN

    features = features.detach().float()
    augmented = cluster_feature_matrix(features, positions, colours,
                                       with_position, with_color)

    generator = torch.Generator(device=augmented.device).manual_seed(seed)
    order = torch.randperm(augmented.shape[0], generator=generator,
                           device=augmented.device)
    subset = augmented[order[: min(CLUSTER_SAMPLE, augmented.shape[0])]]
    fitted = HDBSCAN(min_cluster_size=min_cluster_size,
                     min_samples=min_samples).fit(subset.cpu().numpy()).labels_

    n_clusters = int(fitted.max()) + 1 if fitted.size else 0
    if not n_clusters:
        labels = torch.full((features.shape[0],), NOISE_ID, dtype=torch.long,
                            device=features.device)
        return labels, fit_clusters(features, seed=seed)

    membership = torch.from_numpy(fitted).to(subset.device)
    centroids = torch.stack([
        subset[membership == k].mean(dim=0) for k in range(n_clusters)
    ])
    labels = torch.cdist(augmented, centroids).argmin(dim=1)

    # The Clustering is only used for the palette and the PCA basis here; the
    # per-cell labels above carry the actual partition.
    mean = features.mean(dim=0, keepdim=True)
    centred = features - mean
    sample = centred[order[: min(PCA_SAMPLE, centred.shape[0])]]
    _, _, basis = torch.pca_lowrank(sample, q=3)
    projected = sample @ basis
    return labels, Clustering(
        mean=mean, basis=basis,
        lo=torch.quantile(projected, 0.02, dim=0),
        hi=torch.quantile(projected, 0.98, dim=0),
        centroids=torch.stack([
            features[labels == k].mean(dim=0) for k in range(n_clusters)
        ]),
        colours=_stable_palette(n_clusters),
        noise_fraction=float((fitted < 0).mean()),
    )
