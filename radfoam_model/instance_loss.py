"""Contrastive instance loss over rendered per-ray features.

Follows OpenSplat3D (arXiv:2506.07697): each SAM mask gets a *prototype* -- the
mean rendered feature over its pixels -- and

  * every pixel is pulled toward its own mask's prototype,
  * prototypes of different masks are pushed apart, up to a margin.

Note this is prototype-based, not pairwise between pixels: O(rays) for the
positive term and O(masks^2) for the negative one.

Two departures from the reference, both forced by how radfoam trains:

1. radfoam shuffles rays globally across all views, so one batch mixes ~185
   views. Labels therefore carry a *global* (view, mask) identity rather than
   a per-view index.
2. The negative term is restricted to prototypes **within the same view**.
   Pushing apart prototypes from different views would train "this table seen
   from here" to differ from "the same table seen from there" -- the opposite
   of the goal. The reference gets this for free by processing one view at a
   time; here it has to be enforced explicitly.
"""

import torch
import torch.nn.functional as F

# Per-view mask ids are packed as view_idx * MASK_STRIDE + local_id, so a single
# integer carries both. Must exceed the largest local mask id; automask caps
# masks per frame at top_k = 256.
MASK_STRIDE = 1024
IGNORE_LABEL = -1


def _group_slots(group_index, num_groups):
    """Position of each item within its group, for scatter into a padded tensor."""
    order = torch.argsort(group_index, stable=True)
    sorted_groups = group_index[order]
    starts = torch.searchsorted(
        sorted_groups, torch.arange(num_groups, device=group_index.device)
    )
    slot_sorted = (
        torch.arange(group_index.numel(), device=group_index.device)
        - starts[sorted_groups]
    )
    slots = torch.empty_like(slot_sorted)
    slots[order] = slot_sorted
    return slots


def instance_contrastive_loss(
    features,
    labels,
    gamma=1.0,
    weights=(1.0, 1.0),
    mask_stride=MASK_STRIDE,
):
    """Prototype contrastive loss.

    features: (R, D) rendered features, one per ray.
    labels:   (R,) long, view_idx * mask_stride + local_id, or IGNORE_LABEL.

    Returns a dict with 'total', 'positive', 'negative' and diagnostics.
    """
    device = features.device
    zero = torch.zeros((), device=device, dtype=features.dtype)
    out = {"total": zero, "positive": zero, "negative": zero,
           "n_prototypes": 0, "n_pairs": 0}

    valid = labels >= 0
    if valid.sum() < 2:
        return out

    feats = features[valid]
    lab = labels[valid]

    # One prototype per (view, mask) actually present in this batch.
    unique_labels, inverse = torch.unique(lab, return_inverse=True)
    num_protos = unique_labels.numel()
    counts = torch.zeros(num_protos, device=device, dtype=feats.dtype)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=feats.dtype))
    protos = torch.zeros(num_protos, feats.shape[1], device=device,
                         dtype=feats.dtype)
    protos.index_add_(0, inverse, feats)
    protos = protos / counts.clamp(min=1).unsqueeze(-1)

    # Positive: pull each pixel to its own prototype.
    loss_pos = (feats - protos[inverse]).pow(2).sum(dim=-1).mean()

    # Negative: push apart prototypes, but only within a view.
    proto_views = unique_labels // mask_stride
    view_ids, view_index = torch.unique(proto_views, return_inverse=True)
    num_views = view_ids.numel()
    slots = _group_slots(view_index, num_views)
    width = int(slots.max().item()) + 1

    loss_neg = zero
    n_pairs = 0
    if width >= 2:
        padded = torch.zeros(num_views, width, feats.shape[1], device=device,
                             dtype=feats.dtype)
        padded[view_index, slots] = protos
        occupied = torch.zeros(num_views, width, device=device, dtype=torch.bool)
        occupied[view_index, slots] = True

        distances = torch.cdist(padded, padded)          # (V, width, width)
        pair_valid = occupied.unsqueeze(2) & occupied.unsqueeze(1)
        pair_valid &= torch.triu(
            torch.ones(width, width, device=device, dtype=torch.bool), diagonal=1
        )
        n_pairs = int(pair_valid.sum().item())
        if n_pairs:
            # Margin on the distance, not the squared distance -- matching the
            # reference implementation's torch.cdist, which the paper's
            # ||z_i - z_j||^2 notation does not.
            loss_neg = F.relu(gamma - distances)[pair_valid].mean()

    out["positive"] = loss_pos
    out["negative"] = loss_neg
    out["total"] = weights[0] * loss_pos + weights[1] * loss_neg
    out["n_prototypes"] = num_protos
    out["n_pairs"] = n_pairs
    return out


def multi_level_instance_loss(features, level_labels, gamma=1.0,
                              weights=(1.0, 1.0), level_weights=None):
    """Weighted sum of the contrastive loss over SAM's granularity levels.

    level_labels: (R, L) long. SAM's decoder emits subpart/part/whole masks per
    prompt point; supervising at several granularities lets the embedding carry
    the hierarchy instead of committing to one reading of it.

    `level_weights` exists because the levels are not comparable. Measured on
    garden: level 0 gives 97.8 masks/frame, level 2 gives 50.9, level 1 gives
    **4.1** covering 44% of pixels. With so few distinct prototypes per view
    level 1's negative term is identically 0 -- there is nothing to push apart --
    while its positive term is the largest of the three. Summed at equal weight
    it is a merge-everything force that outvotes the levels carrying the
    objects. None of that makes its masks wrong: level 1 supplies the single
    best mask for 18% of LERF-Mask objects at mean IoU 0.900.
    """
    device = features.device
    n_levels = level_labels.shape[-1]
    if level_weights is None:
        level_weights = [1.0] * n_levels
    totals = {"total": torch.zeros((), device=device, dtype=features.dtype)}
    for level in range(n_levels):
        result = instance_contrastive_loss(
            features, level_labels[:, level], gamma=gamma, weights=weights
        )
        totals["total"] = totals["total"] + level_weights[level] * result["total"]
        totals[f"l{level}_positive"] = result["positive"]
        totals[f"l{level}_negative"] = result["negative"]
        totals[f"l{level}_prototypes"] = result["n_prototypes"]
    return totals
