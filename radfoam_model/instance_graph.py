"""Decompose the scene into instances by partitioning the Delaunay graph.

The alternative in instance_cluster.py runs HDBSCAN over feature vectors and
never looks at which cells actually touch. That throws away the one thing this
representation has and Gaussian splatting does not: the primitives tile space,
so there is a canonical neighbour graph. Two cells on opposite walls with
similar features are not the same object, and only the graph knows that.

The objective is *correlation clustering* (equivalently multicut): edges carry
signed weights, positive to attract and negative to repel, and we look for the
partition minimising the total weight of cut edges. The number of clusters is
not a parameter -- it falls out of where the signs flip. This matters because
the usual image-segmentation formulation does not apply here: s-t min cut needs
a source and a sink, i.e. seeds, and it only ever answers a binary question.
With non-negative weights and no terminals the minimum cut is the empty one, so
signed weights are what makes the problem non-degenerate.

Multicut is NP-hard, so this implements the standard heuristic plus two
simpler baselines to tell how much of any gain is the algorithm rather than
merely the graph:

* `threshold_components` -- cut every edge below a similarity and take connected
  components. The trivial baseline, and worth having because it isolates how
  much of the result is the graph rather than the algorithm.
* `felzenszwalb` -- greedy agglomeration in increasing order of dissimilarity,
  merging two components only while the edge joining them is no worse than the
  internal variation of both, relaxed by k/|C| so small components merge more
  readily. Since merges happen in sorted order under union-find, only minimum
  spanning tree edges can ever cause one, and the MST is all this needs to see.
* `multicut_gaec` -- greedy additive edge contraction, the usual heuristic for
  the multicut objective itself.
"""

from dataclasses import dataclass

import json
from pathlib import Path

import numpy as np
import torch

DEFAULT_MIN_SIZE = 64
NOISE_ID = -1


@dataclass
class GraphClustering:
    labels: torch.Tensor      # (N,) instance id per primitive, NOISE_ID for dropped
    colours: np.ndarray       # (K, 3) uint8
    n_clusters: int
    noise_fraction: float
    mean: torch.Tensor        # (1, D) kept so PCA visualisation still works
    basis: torch.Tensor       # (D, 3)
    lo: torch.Tensor
    hi: torch.Tensor


def undirected_edges(adjacency, offsets):
    """CSR neighbour lists -> (E, 2) unique edges with u < v."""
    adjacency = adjacency.to(torch.int64)
    offsets = offsets.to(torch.int64)
    counts = offsets[1:] - offsets[:-1]
    source = torch.repeat_interleave(
        torch.arange(counts.shape[0], device=adjacency.device), counts
    )
    target = adjacency[: source.shape[0]]
    keep = source < target
    return torch.stack([source[keep], target[keep]], dim=1)


def edge_dissimilarity(features, edges, metric="cosine"):
    """Distance along each edge; small means the two cells look alike."""
    a = features[edges[:, 0]]
    b = features[edges[:, 1]]
    if metric == "cosine":
        a = torch.nn.functional.normalize(a, dim=-1)
        b = torch.nn.functional.normalize(b, dim=-1)
        return 1.0 - (a * b).sum(dim=-1)
    return (a - b).norm(dim=-1)


def _roots(parent):
    """Resolve every node to its root, doubling the path length each pass."""
    while True:
        nxt = parent[parent]
        if np.array_equal(nxt, parent):
            return parent
        parent = nxt


def _relabel(parent, min_size, n_points, device):
    """Union-find roots -> contiguous ids, dropping components below min_size."""
    return relabel_components(_roots(parent), min_size, device)


def relabel_components(component, min_size, device):
    """Arbitrary component ids -> contiguous ids, dropping small components.

    Takes a label per point, NOT a union-find parent array: the two are only the
    same thing when component ids happen to live in the point index space.
    """
    unique, inverse, counts = np.unique(
        component, return_inverse=True, return_counts=True
    )
    big = counts >= min_size
    remap = np.full(unique.shape[0], NOISE_ID, dtype=np.int64)
    remap[big] = np.arange(int(big.sum()))
    labels = remap[inverse]
    return (
        torch.from_numpy(labels).to(device),
        int(big.sum()),
        float((labels < 0).mean()),
    )


def _find(parent, i):
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:      # path compression, iterative
        parent[i], i = root, parent[i]
    return root


def threshold_components(features, edges, tau, min_size=DEFAULT_MIN_SIZE):
    """Keep edges with dissimilarity below tau, then take connected components."""
    distance = edge_dissimilarity(features, edges).cpu().numpy()
    kept = edges.cpu().numpy()[distance < tau]

    parent = np.arange(features.shape[0], dtype=np.int64)
    for u, v in kept:
        ru, rv = _find(parent, u), _find(parent, v)
        if ru != rv:
            parent[ru] = rv
    return _relabel(parent, min_size, features.shape[0], features.device)


def felzenszwalb(features, edges, k=0.02, min_size=DEFAULT_MIN_SIZE):
    """Graph-based agglomeration with an adaptive merge threshold.

    Merge components A and B across an edge of weight w only if

        w <= min(Int(A) + k / |A|, Int(B) + k / |B|)

    where Int(C) is the largest edge weight inside C so far. k sets the scale at
    which a component is considered large enough to defend its own boundary, so
    it behaves like a soft preference for cluster size rather than a count.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree

    n_points = features.shape[0]
    distance = edge_dissimilarity(features, edges).cpu().numpy()
    pairs = edges.cpu().numpy()

    # Only MST edges can trigger a merge: agglomerating in sorted order under
    # union-find is exactly Kruskal, so every non-MST edge closes a cycle and
    # finds both endpoints already joined.
    graph = coo_matrix(
        # Shift off zero: scipy's sparse MST treats a stored 0 as "no edge",
        # and identical features legitimately give distance 0.
        (distance + 1e-6, (pairs[:, 0], pairs[:, 1])),
        shape=(n_points, n_points),
    )
    mst = minimum_spanning_tree(graph).tocoo()
    order = np.argsort(mst.data)
    mst_u, mst_v, mst_w = mst.row[order], mst.col[order], mst.data[order] - 1e-6

    parent = np.arange(n_points, dtype=np.int64)
    size = np.ones(n_points, dtype=np.int64)
    internal = np.zeros(n_points, dtype=np.float64)

    for u, v, w in zip(mst_u, mst_v, mst_w):
        ru, rv = _find(parent, u), _find(parent, v)
        if ru == rv:
            continue
        if w <= min(internal[ru] + k / size[ru], internal[rv] + k / size[rv]):
            parent[ru] = rv
            size[rv] += size[ru]
            internal[rv] = max(w, internal[ru], internal[rv])
    return _relabel(parent, min_size, n_points, features.device)


def _stable_palette(count):
    import cv2

    count = max(count, 1)
    hues = (np.arange(count) * 0.61803398875) % 1.0
    hsv = np.stack(
        [hues * 179, np.full(count, 230.0), np.full(count, 240.0)], axis=1
    ).astype(np.uint8)[None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]


def fit_graph_clusters(features, adjacency, offsets, method="felzenszwalb",
                       min_size=DEFAULT_MIN_SIZE, density=None, **kwargs):
    """Partition the primitives; returns labels plus a PCA basis for viewing."""
    features = features.detach().float()
    edges = undirected_edges(adjacency, offsets)
    if density is not None and method in ("multicut", "multicut_logodds"):
        kwargs["occupancy"] = edge_occupancy(density.detach().float(), edges)

    if method == "felzenszwalb":
        labels, n_clusters, noise = felzenszwalb(
            features, edges, min_size=min_size, **kwargs
        )
    elif method == "threshold":
        labels, n_clusters, noise = threshold_components(
            features, edges, min_size=min_size, **kwargs
        )
    elif method == "multicut":
        labels, n_clusters, noise = multicut_gaec(
            features, edges, min_size=min_size, **kwargs
        )
    elif method == "multicut_logodds":
        labels, n_clusters, noise = multicut_logodds(
            features, edges, min_size=min_size, **kwargs
        )
    else:
        raise ValueError(f"unknown method {method!r}")

    mean = features.mean(dim=0, keepdim=True)
    centred = features - mean
    sample = centred[torch.randperm(centred.shape[0], device=centred.device)[:200_000]]
    _, _, basis = torch.pca_lowrank(sample, q=3)
    projected = sample @ basis

    return GraphClustering(
        labels=labels,
        colours=_stable_palette(n_clusters),
        n_clusters=n_clusters,
        noise_fraction=noise,
        mean=mean,
        basis=basis,
        lo=torch.quantile(projected, 0.02, dim=0),
        hi=torch.quantile(projected, 0.98, dim=0),
    )


def edge_occupancy(density, edges):
    """How solid the thinner of the two cells is, per edge.

    Delaunay adjacency is not contact. In sparse regions cells are enormous, so
    a site inside the bowl and a site on the table can share a Voronoi face
    across the air between them. Feature distance cannot see that, and with
    15M edges any connected chain of attractive ones lets GAEC percolate across
    the whole scene -- which is exactly what it did.

    min(sigma_u, sigma_v) is the cheap stand-in for "do these two actually
    touch": an edge is only as solid as its emptier endpoint, so a link through
    a void gets almost no say.
    """
    solidity = density.reshape(-1)
    return torch.minimum(solidity[edges[:, 0]], solidity[edges[:, 1]])


def _gaec(pairs, weight, safe, n_points, min_size, device):
    """Greedy additive edge contraction over signed edge weights.

    Positive weight pulls two cells together, negative pushes them apart; the
    partition minimising the total weight of cut edges is the minimum cost
    multicut. It is NP-hard, and GAEC is its standard heuristic -- repeatedly
    contract the most attractive edge, *summing* the weights of the parallel
    edges that contraction creates, and stop once no attractive edge is left.
    Summation is what makes the number of clusters emerge instead of being
    chosen: two fragments joined by many weakly positive edges merge, two
    objects joined by a thin seam do not.

    Contraction is done in two phases because a heap over ~15M edges in Python
    is not viable. `safe` marks edges whose merge is not in doubt; those are
    contracted at once via connected components, and phase two runs true GAEC
    on the resulting supernode graph, which is small enough to do properly.

    The weights are supplied rather than computed here, because what belongs in
    them is the whole question -- see multicut_gaec for the tau surrogate and
    multicut_logodds for the calibrated version.
    """
    import heapq

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    # --- phase 1: contract the unambiguous merges ------------------------
    _, coarse = connected_components(
        coo_matrix((np.ones(safe.sum(), bool), (pairs[safe, 0], pairs[safe, 1])),
                   shape=(n_points, n_points)),
        directed=False,
    )
    n_super = int(coarse.max()) + 1

    # --- phase 2: GAEC on the supernode graph ----------------------------
    su, sv = coarse[pairs[:, 0]], coarse[pairs[:, 1]]
    cross = su != sv
    su, sv, sw = su[cross], sv[cross], weight[cross]
    lo, hi = np.minimum(su, sv), np.maximum(su, sv)
    # Sum weights of parallel edges -- this is the contraction rule applied to
    # everything phase one already merged.
    merged = coo_matrix((sw, (lo, hi)), shape=(n_super, n_super)).tocsr().tocoo()

    adjacency = [dict() for _ in range(n_super)]
    for u, v, w in zip(merged.row, merged.col, merged.data):
        adjacency[u][v] = w
        adjacency[v][u] = w

    parent = np.arange(n_super, dtype=np.int64)
    heap = [(-w, int(u), int(v)) for u, v, w in
            zip(merged.row, merged.col, merged.data) if w > 0]
    heapq.heapify(heap)

    while heap:
        negative_w, u, v = heapq.heappop(heap)
        ru, rv = _find(parent, u), _find(parent, v)
        if ru == rv:
            continue
        current = adjacency[ru].get(rv)
        # Stale entry: the weight changed under us since it was pushed.
        if current is None or abs(current + negative_w) > 1e-9:
            continue
        if current <= 0:
            break

        # Merge the smaller neighbourhood into the larger.
        if len(adjacency[ru]) < len(adjacency[rv]):
            ru, rv = rv, ru
        parent[rv] = ru
        del adjacency[ru][rv], adjacency[rv][ru]
        for other, w in adjacency[rv].items():
            total = adjacency[ru].get(other, 0.0) + w
            adjacency[ru][other] = total
            adjacency[other][ru] = total
            adjacency[other].pop(rv, None)
            if total > 0:
                heapq.heappush(heap, (-total, ru, other))
        adjacency[rv].clear()

    # coarse maps point -> supernode; super_root maps supernode -> its cluster.
    # The result is a label per point, which is not a union-find parent array.
    super_root = _roots(parent)
    return relabel_components(super_root[coarse], min_size, device)


def multicut_gaec(features, edges, tau, safe_fraction=0.5,
                  min_size=DEFAULT_MIN_SIZE, metric="euclidean",
                  occupancy=None, occupancy_floor=1e-3):
    """Multicut with the surrogate weight w_uv = tau - d(f_u, f_v).

    Kept because the LERF-Mask numbers were measured with it, but it is not the
    right weight and the failure is measurable. GAEC *sums* weights on
    contraction, and summing (tau - d) accumulates with the NUMBER of edges
    between two supernodes rather than with the evidence they carry, so tau
    trades off against graph density instead of meaning anything fixed. On
    teatime the criterion ranks edges at AUC 0.988, yet 97.7% of Delaunay edges
    are intra-instance, so even 99% precision leaves ~12k wrongly attractive
    edges among 418 objects; past tau = 0.3 they percolate and the partition
    collapses. multicut_logodds is the principled replacement.
    """
    n_points = features.shape[0]
    distance = edge_dissimilarity(features, edges, metric).cpu().numpy()
    pairs = edges.cpu().numpy()
    weight = tau - distance

    if occupancy is not None:
        # Scale, do not threshold: a weak edge should be quiet rather than
        # repulsive, so that thin genuine contacts are not turned into cuts.
        scale = (occupancy / occupancy.max()).clamp(min=0).cpu().numpy()
        weight = weight * scale
        # Phase one must respect it too, or the safe-merge pass alone
        # reconnects everything the weighting was meant to separate.
        distance = np.where(scale > occupancy_floor, distance, np.inf)

    safe = distance < safe_fraction * tau
    return _gaec(pairs, weight, safe, n_points, min_size, features.device)


LOGIT_CLIP = 12.0          # +-12 nats is already p = 1 - 6e-6; beyond it the
                           # sums are dominated by float noise, not evidence.


def _fit_two_component(x, iterations=60, seed=0):
    """EM for a two-Gaussian mixture on a 1-D sample. Returns (pi, mu, sigma).

    Component 0 is the one with the smaller mean, i.e. "these two cells are the
    same object". No sklearn: it is thirty lines and avoids a dependency in the
    clustering path.
    """
    rng = np.random.default_rng(seed)
    lo, hi = np.quantile(x, [0.1, 0.9])
    mu = np.array([lo, hi], dtype=np.float64)
    sigma = np.full(2, x.std() / 2 + 1e-6)
    pi = np.array([0.5, 0.5])
    for _ in range(iterations):
        # E step
        z = np.stack([
            np.log(pi[k] + 1e-300) - np.log(sigma[k]) - 0.5 * ((x - mu[k]) / sigma[k]) ** 2
            for k in range(2)
        ])
        z -= z.max(axis=0, keepdims=True)
        r = np.exp(z)
        r /= r.sum(axis=0, keepdims=True)
        # M step
        n = r.sum(axis=1) + 1e-12
        pi = n / n.sum()
        mu = (r * x).sum(axis=1) / n
        sigma = np.sqrt((r * (x - mu[:, None]) ** 2).sum(axis=1) / n) + 1e-6
    order = np.argsort(mu)
    return pi[order], mu[order], sigma[order]


def edge_log_odds(distance, sample=500_000, seed=0, report=False):
    """log P(same object) / P(different), per edge, from the distance alone.

    Multicut's weights are supposed to be log-odds. That is not cosmetic: GAEC
    sums them when it contracts, and summing log-odds is Bayesian evidence
    accumulation, whereas summing (tau - d) is not -- it just counts edges. It
    also removes tau: the contraction rule "stop when no positive weight is
    left" becomes "stop when the evidence no longer favours merging", and zero
    is a meaningful place to stop rather than a tuned one.

    Calibration is label-free on purpose. The obvious alternative -- fit
    against HDBSCAN labels -- is circular, since HDBSCAN partitions the same
    features the distance comes from, so it would score its own agreement with
    itself. Instead the distance histogram is fitted with a two-component
    mixture in log space, which is bimodal for the reason the task is possible
    at all: intra-object edges concentrate near zero, cross-object ones sit out
    near the contrastive loss's margin.

    The mixing weight carries the prior, which the tau surrogate ignored
    entirely. Roughly 97-98% of Delaunay edges are intra-object, so a random
    edge starts at about +3.7 nats in favour of merging and cutting has to
    earn its way past that. Getting the asymmetry into the weight is most of
    the point.
    """
    x = np.log(np.asarray(distance, dtype=np.float64) + 1e-6)
    finite = np.isfinite(x)
    rng = np.random.default_rng(seed)
    pool = x[finite]
    if pool.size > sample:
        pool = pool[rng.choice(pool.size, sample, replace=False)]
    pi, mu, sigma = _fit_two_component(pool, seed=seed)

    def log_density(k):
        return (np.log(pi[k] + 1e-300) - np.log(sigma[k])
                - 0.5 * ((x - mu[k]) / sigma[k]) ** 2)

    weight = np.clip(log_density(0) - log_density(1), -LOGIT_CLIP, LOGIT_CLIP)
    weight[~finite] = -LOGIT_CLIP
    if report:
        print(f"  mixture: p(same)={pi[0]:.4f} at log d {mu[0]:+.3f}+-{sigma[0]:.3f}, "
              f"p(diff)={pi[1]:.4f} at {mu[1]:+.3f}+-{sigma[1]:.3f}; "
              f"{100 * (weight > 0).mean():.1f}% of edges attractive", flush=True)
    return weight


def sam_edge_counts(model, data, edges, mask_dir, view_names, device,
                    level=0, quantiles=(0.2, 0.8), report=False):
    """Per edge, how often the two cells land in the same SAM mask.

    This is the evidence the feature distance is a proxy for, and unlike the
    distance it is independent of the learned features -- which matters,
    because calibrating against HDBSCAN labels would only measure how well
    feature distance agrees with feature-space clustering.

    A cell is "seen" in a view if it terminates a ray, so the same surface
    query the view ranking uses. Returns (agree, disagree) counts per edge;
    edges never co-observed get (0, 0) and contribute nothing.
    """
    from PIL import Image

    from radfoam_model.instance_language import surface_cells

    index = json.loads((Path(mask_dir) / "frame_index.json").read_text())
    frame_of = {name: i for i, name in enumerate(index["names"])}
    width, height = data.img_wh
    n_points = model.att_feat.shape[0]

    u, v = edges[:, 0], edges[:, 1]
    agree = torch.zeros(edges.shape[0], dtype=torch.int32, device=device)
    disagree = torch.zeros_like(agree)

    for n, name in enumerate(view_names):
        frame = frame_of.get(name)
        if frame is None:
            continue
        path = Path(mask_dir) / f"labels_l{level}" / f"{frame:06d}.png"
        if not path.exists():
            continue
        raw = Image.open(path)
        if raw.size != (width, height):
            raw = raw.resize((width, height), Image.NEAREST)
        # 0 is background in these maps, so -1 marks "no mask here".
        pixel_mask = torch.from_numpy(
            np.asarray(raw).astype(np.int64) - 1).reshape(-1).to(device)

        rays = data.rays[data.image_names.index(name)].to(device).reshape(-1, 6)
        cells = surface_cells(model, rays, device, quantiles)

        # One mask per cell: last write wins, which is adequate because a cell
        # projecting into two masks at once is exactly the ambiguous case we
        # want counted weakly rather than resolved arbitrarily. Both depth
        # quantiles vote, which roughly doubles how much of the scene one view
        # can say anything about.
        mask_of_cell = torch.full((n_points,), -1, dtype=torch.long, device=device)
        for column in range(cells.shape[1] if cells.ndim > 1 else 1):
            slot = cells[:, column] if cells.ndim > 1 else cells
            mask_of_cell[slot] = pixel_mask

        mu, mv = mask_of_cell[u], mask_of_cell[v]
        both = (mu >= 0) & (mv >= 0)
        agree += (both & (mu == mv)).int()
        disagree += (both & (mu != mv)).int()
        if report and n % 10 == 0:
            print(f"\r  SAM co-occurrence {n + 1}/{len(view_names)} views",
                  end="", flush=True)
    if report:
        seen = (agree + disagree) > 0
        print(f"\r  SAM co-occurrence: {100 * seen.float().mean():.1f}% of edges "
              f"co-observed, {(agree + disagree).float().mean():.2f} votes/edge",
              flush=True)
    return agree.cpu().numpy(), disagree.cpu().numpy()


def _fit_bernoulli_mixture(agree, disagree, iterations=80):
    """EM for P(agreement counts | same/different). Returns (pi, alpha, beta).

    alpha is how often two cells of the SAME object are seen in one mask,
    beta the same for different objects. Counts are far more separable than
    distances -- an edge inside an object agrees nearly every time it is
    observed -- which is why this recovers a sane prior where the Gaussian
    mixture on log-distance did not.
    """
    n = agree + disagree
    keep = n > 0
    a, d, total = agree[keep].astype(np.float64), disagree[keep].astype(np.float64), n[keep]
    pi, alpha, beta = 0.9, 0.95, 0.15
    for _ in range(iterations):
        ll_same = a * np.log(alpha) + d * np.log1p(-alpha) + np.log(pi)
        ll_diff = a * np.log(beta) + d * np.log1p(-beta) + np.log1p(-pi)
        m = np.maximum(ll_same, ll_diff)
        r = np.exp(ll_same - m) / (np.exp(ll_same - m) + np.exp(ll_diff - m))
        pi = float(np.clip(r.mean(), 1e-4, 1 - 1e-4))
        alpha = float(np.clip((r * a).sum() / (r * total).sum(), 1e-3, 1 - 1e-3))
        beta = float(np.clip(((1 - r) * a).sum() / ((1 - r) * total).sum(),
                             1e-3, 1 - 1e-3))
    return pi, alpha, beta


def sam_log_odds(agree, disagree, report=False):
    """(prior log-odds, per-edge SAM log-likelihood ratio, soft same-labels).

    The prior is returned separately so it is added exactly once no matter how
    many evidence sources are combined.
    """
    pi, alpha, beta = _fit_bernoulli_mixture(agree, disagree)
    llr = (agree * np.log(alpha / beta)
           + disagree * (np.log1p(-alpha) - np.log1p(-beta)))
    prior = float(np.log(pi / (1 - pi)))
    n = agree + disagree
    posterior = np.full(len(agree), np.nan)
    seen = n > 0
    posterior[seen] = 1.0 / (1.0 + np.exp(-(prior + llr[seen])))
    if report:
        print(f"  SAM mixture: P(same)={pi:.4f}, agreement {alpha:.3f} within "
              f"an object vs {beta:.3f} across; prior {prior:+.2f} nats",
              flush=True)
    return prior, np.clip(llr, -LOGIT_CLIP * 2, LOGIT_CLIP * 2), posterior


def calibrate_distance(distance, posterior, bins=64, report=False):
    """LLR of the feature distance, calibrated against the SAM posterior.

    Non-parametric on purpose: fitting a two-Gaussian mixture to log-distance
    without labels put the prior at 0.45 when the truth is 0.977, and the
    resulting partition over-cut badly. Given even weak labels, a binned
    likelihood ratio needs no shape assumption at all.
    """
    x = np.log(np.asarray(distance, dtype=np.float64) + 1e-6)
    seen = np.isfinite(posterior)
    edges_ = np.quantile(x[seen], np.linspace(0, 1, bins + 1))
    edges_[0], edges_[-1] = -np.inf, np.inf
    which = np.clip(np.digitize(x, edges_) - 1, 0, bins - 1)
    w_same = np.bincount(which[seen], weights=posterior[seen], minlength=bins)
    w_diff = np.bincount(which[seen], weights=1 - posterior[seen], minlength=bins)
    # Laplace smoothing so an empty bin is agnostic rather than infinite.
    p_same = (w_same + 0.5) / (w_same.sum() + 0.5 * bins)
    p_diff = (w_diff + 0.5) / (w_diff.sum() + 0.5 * bins)
    table = np.clip(np.log(p_same / p_diff), -LOGIT_CLIP, LOGIT_CLIP)
    if report:
        print(f"  distance LLR spans {table.min():+.2f} to {table.max():+.2f} "
              f"nats over {bins} bins", flush=True)
    return table[which]


def multicut_sam(features, edges, agree, disagree, min_size=DEFAULT_MIN_SIZE,
                 metric="euclidean", safe_nats=6.0, occupancy=None,
                 use_prior=False, report=False):
    """Multicut on log-odds calibrated against SAM co-occurrence.

    Evidence sources add, because log-odds add under conditional independence:
    the feature distance calibrated against the SAM posterior, the
    co-occurrence counts themselves, and optionally occupancy.

    The marginal prior is NOT added by default, and that is a correction to
    the obvious reading of "use log-odds". Two separate reasons. Only about
    0.3-0.8 edges per edge carry any SAM vote, because two adjacent cells are
    rarely both front surfaces in the same view and vacuum cells are never
    observed at all -- so a +3.1 nat prior lands on ~99% of edges that have no
    evidence to argue back with, and the distance term spans barely 2 nats.
    More fundamentally, an independent per-edge prior is inconsistent with a
    partition: the events "u,v same object" are strongly dependent across the
    15M edges of a 400-object scene, and GAEC sums weights, so a constant
    positive offset compounds until everything contracts. Measured: with the
    prior on, 99.8% of edges come out attractive and the scene collapses.
    Leaving it at zero makes an unobserved edge agnostic, which is what it is.
    """
    features = features.detach().float()
    distance = edge_dissimilarity(features, edges, metric).cpu().numpy()
    prior, llr_sam, posterior = sam_log_odds(agree, disagree, report=report)
    weight = llr_sam + calibrate_distance(distance, posterior, report=report)
    if use_prior:
        weight = weight + prior
    if occupancy is not None:
        weight = weight + occupancy_log_odds(occupancy, posterior,
                                             report=report)
    weight = np.clip(weight, -LOGIT_CLIP * 4, LOGIT_CLIP * 4)
    safe = weight > safe_nats
    if report:
        print(f"  {100 * (weight > 0).mean():.1f}% attractive, "
              f"{100 * safe.mean():.1f}% safe to contract", flush=True)
    return _gaec(edges.cpu().numpy(), weight, safe, features.shape[0],
                 min_size, features.device)


def occupancy_log_odds(occupancy, posterior, bins=32, report=False):
    """LLR of edge solidity, calibrated the same way as the distance.

    Density alone separates within- from across-object edges at AUC 0.917, so
    it carries real information about whether a Delaunay edge is contact or
    air. The older path divided distance by occupancy and measurably destroyed
    signal (0.9875 -> 0.9841); adding a calibrated ratio cannot, because a
    source with nothing to say contributes zero.
    """
    x = np.log(np.asarray(occupancy, dtype=np.float64) + 1e-6)
    return calibrate_distance(np.exp(x), posterior, bins=bins, report=report)


def multicut_logodds(features, edges, min_size=DEFAULT_MIN_SIZE,
                     metric="euclidean", safe_nats=6.0, extra=None,
                     report=False):
    """Multicut on calibrated log-odds. No tau.

    `extra` is any further per-edge log-odds to add -- SAM co-occurrence,
    occupancy. Adding them assumes the sources are conditionally independent
    given the answer, which is the usual naive-Bayes bargain and is why they
    compose by addition rather than by multiplying distances together. The
    existing occupancy path divides distance by occupancy and measurably
    destroys signal (AUC 0.9875 -> 0.9841 on teatime); adding log-likelihood
    ratios cannot do that, because a source with nothing to say contributes
    zero.
    """
    distance = edge_dissimilarity(features, edges, metric).cpu().numpy()
    weight = edge_log_odds(distance, report=report)
    if extra is not None:
        weight = np.clip(weight + np.asarray(extra, dtype=np.float64),
                         -LOGIT_CLIP * 4, LOGIT_CLIP * 4)
    pairs = edges.cpu().numpy()
    # "Not in doubt" is now a statement about evidence, not a fraction of tau.
    safe = weight > safe_nats
    if report:
        print(f"  {100 * safe.mean():.1f}% of edges safe to contract "
              f"(> {safe_nats:.0f} nats)", flush=True)
    return _gaec(pairs, weight, safe, features.shape[0], min_size,
                 features.device)


def clustering_from_labels(features, labels, seed=0):
    """Wrap a graph partition as an instance_cluster.Clustering.

    The rendering path composites features and then assigns each pixel to the
    nearest centroid, so a graph partition is consumed by handing it the same
    interface: centroids become the per-cluster mean feature. Note this does
    give up some of what the graph bought -- a pixel takes its nearest centroid
    whether or not that cluster is anywhere near it in space -- but it isolates
    the effect of *which* clusters exist, which is the thing under test.
    """
    from radfoam_model.instance_cluster import Clustering

    features = features.detach().float()
    valid = labels >= 0
    n_clusters = int(labels.max().item()) + 1 if valid.any() else 0
    if n_clusters:
        centroids = torch.stack([
            features[labels == k].mean(dim=0) for k in range(n_clusters)
        ])
    else:
        centroids = torch.zeros(0, features.shape[1], device=features.device)

    mean = features.mean(dim=0, keepdim=True)
    centred = features - mean
    generator = torch.Generator(device=centred.device).manual_seed(seed)
    order = torch.randperm(centred.shape[0], generator=generator,
                           device=centred.device)
    sample = centred[order[:200_000]]
    _, _, basis = torch.pca_lowrank(sample, q=3)
    projected = sample @ basis

    return Clustering(
        mean=mean, basis=basis,
        lo=torch.quantile(projected, 0.02, dim=0),
        hi=torch.quantile(projected, 0.98, dim=0),
        centroids=centroids,
        colours=_stable_palette(n_clusters),
        noise_fraction=float((~valid).float().mean().item()),
    )


def graph_clustering(model, method="multicut", **kwargs):
    """Convenience: partition a loaded model's cells and return a Clustering."""
    result = fit_graph_clusters(
        model.att_feat, model.point_adjacency, model.point_adjacency_offsets,
        method=method, **kwargs
    )
    return clustering_from_labels(model.att_feat, result.labels)
