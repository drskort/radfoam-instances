"""ScanNet++ 3D instance segmentation: AP / AP50 / AP25 on mesh points.

The benchmark OpenSplat3D reports, and the one worth caring about here, because
it scores the 3D partition directly instead of a rendered mask against a 2D
polygon. Every LERF-family number is a projection; this one is not.

It is also where a space-tiling representation has something specific to say.
Their pipeline assigns each ground-truth point "the instance ID of the nearest
Gaussian mean" and then applies graph-based smoothing worth +5.3 AP (19.2 ->
24.5) -- the smoothing exists because Gaussians overlap and do not tile, so
nearest-centre is an approximation of an ill-posed question. For a Voronoi
tessellation the same query is exact: a point lies in exactly one cell, and
nearest-site IS containment, by definition. If the claim holds, we should reach
their post-processed number without post-processing.

Ground truth is stored in two files. scans/segments.json gives one
over-segmentation id per mesh vertex; scans/segments_anno.json groups those
segment ids into labelled instances.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("/shared/scannetpp/data")
BENCHMARK_CLASSES = Path(
    "/shared/scannetpp/metadata/semantic_benchmark/top100_instance.txt"
)
# ScanNet++'s own scorer, which OpenSplat3D uses:
#   overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# Note 0.95 is EXCLUDED -- that arange stops at 0.9, so AP averages ten
# thresholds from 0.50 to 0.90, not 0.50 to 0.95. Rounded because arange on a
# float step yields 0.6000000000000001, and an instance overlapping at exactly
# 0.6 would otherwise fail its own threshold.
# (submodules/scannetpp/scannetpp/semantic/utils/instance_utils.py)
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.95, 0.05), 2)
# Ground-truth instances smaller than this are dropped by their scorer.
MIN_REGION_SIZE = 100


def load_benchmark_classes():
    with open(BENCHMARK_CLASSES) as handle:
        return {line.strip() for line in handle if line.strip()}


def load_gt_instances(scene, restrict_to_benchmark=True):
    """(vertices, list of (label, vertex_index_array)) for one scene.

    Instances outside the 83-class instance benchmark are dropped by default.
    They are mostly structural -- wall, floor, ceiling -- and scoring against
    them would reward a method for segmenting the room shell rather than its
    objects.
    """
    from plyfile import PlyData

    scans = ROOT / scene / "scans"
    ply = PlyData.read(str(scans / "mesh_aligned_0.05.ply"))["vertex"]
    vertices = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)

    seg_of_vertex = np.asarray(
        json.loads((scans / "segments.json").read_text())["segIndices"]
    )
    groups = json.loads((scans / "segments_anno.json").read_text())["segGroups"]
    allowed = load_benchmark_classes() if restrict_to_benchmark else None

    # One pass over the vertices instead of one per instance: segment ids are
    # sparse and arbitrary, so map them to a dense index first.
    order = np.argsort(seg_of_vertex, kind="stable")
    sorted_segments = seg_of_vertex[order]
    unique, first = np.unique(sorted_segments, return_index=True)
    bounds = np.append(first, len(sorted_segments))
    members = {int(u): order[bounds[i]:bounds[i + 1]]
               for i, u in enumerate(unique)}

    instances = []
    for group in groups:
        label = group.get("label", "")
        if allowed is not None and label not in allowed:
            continue
        chunks = [members[s] for s in group["segments"] if s in members]
        if not chunks:
            continue
        instances.append((label, np.concatenate(chunks)))
    return vertices, instances


def assign_by_containment(vertices, model, labels, device="cuda",
                          chunk=500_000, min_density=1e-3):
    """Instance label per mesh vertex, from the nearest cell that renders it.

    Nearest site is the containing Voronoi cell, exactly -- that is the
    definition -- and it is the query "nearest Gaussian mean" only approximates,
    which is why their pipeline needs graph smoothing afterwards. But exactness
    is about the wrong thing when a third of the cells are empty: RadFoam
    represents objects as opaque shells over vacuum, so the cell CONTAINING a
    surface point is often the unobserved cell just behind it, whose features
    sit near initialisation. Measured on 7b6477cb95: 35.7% of cells never
    render, 25.4% of mesh vertices land in one of them, and 99.4% of those come
    back unlabelled.

    Restricting the query to cells that actually render asks the question we
    mean -- which rendering cell owns this surface point -- and the surface is
    close enough for that to be well posed: the median mesh vertex is 1.7 cm
    from the nearest rendering cell, p90 5.8 cm.

    A pairwise distance matrix is not an option at this scale: 1.3M vertices
    against 2M sites is 1452 GiB, and chunking the queries does not help when
    the other side is the whole scene.
    """
    import radfoam
    import torch

    points = model.get_trace_data()[0].detach()
    index = torch.arange(points.shape[0], device=points.device)
    if min_density is not None:
        keep = model.get_primal_density().detach().reshape(-1) > min_density
        points, index = points[keep].contiguous(), index[keep]
    tree = radfoam.build_aabb_tree(points)

    out = np.empty(len(vertices), dtype=np.int64)
    for start in range(0, len(vertices), chunk):
        block = torch.as_tensor(
            vertices[start:start + chunk], dtype=torch.float32, device=device
        ).contiguous()
        nearest = index[radfoam.nn(points, tree, block).long()].cpu().numpy()
        out[start:start + chunk] = labels[nearest]
    return out


def fill_noise_labels(labels, features, centroids, noise_id=-1,
                      chunk=500_000):
    """Give every abstaining cell the label of its nearest labelled neighbour.

    HDBSCAN abstains on ~70% of cells, and on ScanNet++ that is not free: the
    metric scores every mesh vertex, so an unlabelled cell is a guaranteed miss
    against all 48 ground-truth instances. On LERF it cost nothing, because the
    argmax renderer simply skips noise and the 2D mask is scored on what is
    drawn -- the configuration was tuned under a metric that did not charge for
    abstention.

    Nearest cluster CENTROID in feature space, not nearest labelled cell:
    there are ~200 centroids and ~600k labelled cells, and a pairwise distance
    against the latter is 120 billion entries. This cannot invent structure; it
    only stops discarding cells whose features already sit closest to an
    instance the clustering did find.
    """
    import torch

    labelled = labels != noise_id
    if labelled.all() or not labelled.any():
        return labels
    filled = labels.clone()
    missing = torch.nonzero(~labelled, as_tuple=True)[0]
    centroids = centroids.to(features.device, features.dtype)
    for start in range(0, len(missing), chunk):
        block = missing[start:start + chunk]
        nearest = torch.cdist(features[block], centroids).argmin(dim=1)
        filled[block] = nearest.to(filled.dtype)
    return filled


def split_connected(labels, edges, min_size=0, noise_id=-1, report=False):
    """Split each instance into its spatially connected components, exactly.

    Same purpose as OpenSplat3D's dbscan_denoising -- worth +5.3 AP to them
    (19.2 -> 24.5) -- but done on the adjacency instead of an eps-ball.
    They cluster Gaussians, which have no neighbour structure, so DBSCAN over
    positions is the only way to ask "are these two blobs the same object or
    two lookalikes across the room". The foam already answers that: cells that
    touch share a Voronoi face. Connected components over same-label edges is
    the exact version of the query, one pass over the edge list rather than a
    radius search, and it needs no eps to be chosen for the scene's scale.

    That matters practically as well: DBSCAN at their eps=0.5 over 1.3M mesh
    vertices needed more than 48 GB.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(labels)
    same = (labels[edges[:, 0]] == labels[edges[:, 1]]) & (
        labels[edges[:, 0]] != noise_id)
    keep = edges[same]
    graph = coo_matrix(
        (np.ones(len(keep), bool), (keep[:, 0], keep[:, 1])), shape=(n, n))
    _, component = connected_components(graph, directed=False)

    # a component is only an instance where the cells were labelled at all
    out = np.full_like(labels, noise_id)
    valid = labels != noise_id
    comp = component[valid]
    uniq, inverse, counts = np.unique(comp, return_inverse=True,
                                      return_counts=True)
    big = counts >= max(min_size, 1)
    remap = np.full(len(uniq), noise_id, dtype=labels.dtype)
    remap[big] = np.arange(big.sum(), dtype=labels.dtype)
    out[valid] = remap[inverse]
    if report:
        before = len(np.unique(labels[labels != noise_id]))
        print(f"  connected-component split: {before} -> {int(big.sum())} "
              f"instances", flush=True)
    return out


def dbscan_denoise(labels, positions, eps=0.5, min_samples=5,
                   min_cluster_size=0, noise_id=-1, report=False):
    """Split each instance into its spatially connected components.

    This is OpenSplat3D's post-processing, and it is worth +5.3 AP to them
    (19.2 -> 24.5). It is not an objectness score -- their class-agnostic
    export writes confidence 1.0 for every prediction and never ranks
    anything. It is geometry: HDBSCAN clusters in FEATURE space, so a single
    cluster can span two objects that merely look alike -- two identical
    chairs, or a wall segment and the matching wall across the room. DBSCAN
    over the cells' positions separates them, and components below
    min_cluster_size are dropped back to noise.

    Their defaults, from export_predictions.py: eps 0.5, min_samples 5,
    min_cluster_size 0 (opensplat3d/cluster/hdbscan.py::dbscan_denoising).
    """
    from sklearn.cluster import DBSCAN

    out = np.full_like(labels, noise_id)
    next_label = 0
    for instance in np.unique(labels):
        if instance == noise_id:
            continue
        mask = labels == instance
        parts = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(
            positions[mask])
        for part in np.unique(parts):
            if part == noise_id or (parts == part).sum() < min_cluster_size:
                continue
            member = mask.copy()
            member[mask] = parts == part
            out[member] = next_label
            next_label += 1
    if report:
        print(f"  dbscan denoise: {len(np.unique(labels)) - 1} -> {next_label} "
              f"instances, {100 * (out == noise_id).mean():.1f}% now noise",
              flush=True)
    return out


def predictions_from_labels(vertex_labels, noise_id=-1, min_vertices=100,
                            score="uniform"):
    """(vertex_index_array, score) per predicted instance.

    "uniform" gives every prediction confidence 1.0, which is what OpenSplat3D
    exports for the class-agnostic table -- their inst_agnostic_pred_out writes
    a literal 1.0 and only the open-vocabulary table carries a real confidence
    (the max softmax over language similarity). Matching that matters: ranking
    by anything informative reorders the PR curve and inflates AP relative to
    their number.

    "size" ranks by vertex count. Kept for diagnosis only -- if AP is weak
    while AP50 is healthy, comparing the two says whether the partition or the
    absence of an objectness score is responsible.
    """
    keep = vertex_labels != noise_id
    ids, counts = np.unique(vertex_labels[keep], return_counts=True)
    order = np.argsort(vertex_labels, kind="stable")
    sorted_labels = vertex_labels[order]
    bounds = np.searchsorted(sorted_labels, np.append(ids, ids[-1] + 1))
    out = []
    for i, instance in enumerate(ids):
        members = order[bounds[i]:bounds[i + 1]]
        if len(members) >= min_vertices:
            out.append((members,
                        float(len(members)) if score == "size" else 1.0))
    return out


def average_precision(predictions, gt_instances, n_vertices, void=None):
    """Class-agnostic instance AP, ScanNet++ protocol.

    Predictions are ranked by score; each is matched greedily to the unmatched
    ground-truth instance it overlaps most, at each IoU threshold in turn.

    `void` marks vertices belonging to no benchmark instance -- wall, floor,
    and every object outside the 83 instance classes. A prediction lying mostly
    in void is neither a true nor a false positive, it is DROPPED:

        num_ignore = pred["void_intersection"]
        proportion_ignore = num_ignore / pred["vert_count"]
        if proportion_ignore <= overlap_th:   # only then a false positive

    (submodules/scannetpp/.../eval_instance.py). Without this a method that
    segments the whole scene is punished for correctly finding the wall, while
    one that only proposes vocabulary objects is not -- which is a difference
    in what the methods attempt, not in how well they do it. Omitting it cost
    us most of our AP.
    """
    gt_instances = [g for g in gt_instances if len(g[1]) >= MIN_REGION_SIZE]
    gt_masks = np.zeros((len(gt_instances), n_vertices), dtype=bool)
    for i, (_, idx) in enumerate(gt_instances):
        gt_masks[i, idx] = True
    gt_sizes = gt_masks.sum(axis=1)

    ranked = sorted(predictions, key=lambda p: -p[1])
    overlaps = np.zeros((len(ranked), len(gt_instances)))
    void_share = np.zeros(len(ranked))
    for i, (idx, _) in enumerate(ranked):
        hit = gt_masks[:, idx].sum(axis=1)
        overlaps[i] = hit / (len(idx) + gt_sizes - hit)
        if void is not None:
            void_share[i] = void[idx].mean()

    results = {}
    for threshold in np.append(IOU_THRESHOLDS, [0.25]):
        taken = np.zeros(len(gt_instances), dtype=bool)
        tp, counted = [], []
        for i in range(len(ranked)):
            candidates = np.where((overlaps[i] >= threshold) & ~taken)[0]
            if candidates.size:
                best = candidates[np.argmax(overlaps[i, candidates])]
                taken[best] = True
                tp.append(1.0); counted.append(True)
            else:
                # a prediction lying mostly in unannotated space is dropped
                # rather than charged; see the docstring.
                ignored = void_share[i] > threshold
                tp.append(0.0); counted.append(not ignored)
        tp = np.array(tp)[np.array(counted)]
        cum_tp = np.cumsum(tp)
        precision = cum_tp / np.arange(1, len(tp) + 1)
        recall = cum_tp / max(len(gt_instances), 1)
        # All-point interpolation, as ScanNet's scorer uses.
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        results[round(float(threshold), 2)] = float(
            np.sum(np.diff(np.append(0.0, recall)) * precision)
        )
    return {
        "AP": float(np.mean([results[round(float(t), 2)] for t in IOU_THRESHOLDS])),
        "AP50": results[0.5],
        "AP25": results[0.25],
        "n_pred": len(ranked),
        "n_gt": len(gt_instances),
        "void_dropped_at_50": int((void_share > 0.5).sum()),
    }
