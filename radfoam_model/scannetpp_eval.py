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


def assign_by_containment(vertices, sites, labels, device="cuda", chunk=200_000):
    """Instance label per mesh vertex, by which Voronoi cell contains it.

    Nearest site IS the containing cell -- that is the definition of a Voronoi
    diagram -- so this is exact, not an approximation that needs smoothing
    afterwards. Chunked because the vertex count runs past a million and the
    site count past two.
    """
    import torch

    sites_t = torch.as_tensor(sites, dtype=torch.float32, device=device)
    out = np.empty(len(vertices), dtype=np.int64)
    for start in range(0, len(vertices), chunk):
        block = torch.as_tensor(
            vertices[start:start + chunk], dtype=torch.float32, device=device
        )
        nearest = torch.cdist(block, sites_t).argmin(dim=1)
        out[start:start + chunk] = labels[nearest.cpu().numpy()]
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


def average_precision(predictions, gt_instances, n_vertices):
    """Class-agnostic instance AP, ScanNet protocol.

    Predictions are ranked by score; each is matched greedily to the unmatched
    ground-truth instance it overlaps most, at each IoU threshold in turn.
    """
    gt_instances = [g for g in gt_instances if len(g[1]) >= MIN_REGION_SIZE]
    gt_masks = np.zeros((len(gt_instances), n_vertices), dtype=bool)
    for i, (_, idx) in enumerate(gt_instances):
        gt_masks[i, idx] = True
    gt_sizes = gt_masks.sum(axis=1)

    ranked = sorted(predictions, key=lambda p: -p[1])
    overlaps = np.zeros((len(ranked), len(gt_instances)))
    for i, (idx, _) in enumerate(ranked):
        hit = gt_masks[:, idx].sum(axis=1)
        overlaps[i] = hit / (len(idx) + gt_sizes - hit)

    results = {}
    for threshold in np.append(IOU_THRESHOLDS, [0.25]):
        taken = np.zeros(len(gt_instances), dtype=bool)
        tp = np.zeros(len(ranked))
        for i in range(len(ranked)):
            candidates = np.where((overlaps[i] >= threshold) & ~taken)[0]
            if candidates.size:
                best = candidates[np.argmax(overlaps[i, candidates])]
                taken[best] = True
                tp[i] = 1.0
        cum_tp = np.cumsum(tp)
        precision = cum_tp / np.arange(1, len(ranked) + 1)
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
    }
