"""Model-independent everything-mode proposal logic.

SAM 3.1 ships no automatic mask generator, so this is the shared half of the one
we build; SAM 2.1's generator does the same work internally. Keeping it here, and
using it for both backends, means proposal behaviour is comparable across arms by
construction rather than by coincidence.

Defaults match SAM's published automatic mask generator settings.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AutomaskConfig:
    points_per_side: int = 32
    pred_iou_thresh: float = 0.8
    stability_thresh: float = 0.9
    nms_iou_thresh: float = 0.7
    top_k: int = 256
    min_mask_area: int = 1


def point_grid(points_per_side, height, width):
    """Return an (N, 2) integer array of (x, y) grid points inside the image.

    Points sit at cell centres, so none land on the image border.
    """
    offsets = (np.arange(points_per_side) + 0.5) / points_per_side
    xs = (offsets * width).astype(np.int64)
    ys = (offsets * height).astype(np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def mask_iou(a, b):
    """Intersection over union of two boolean masks; 0.0 if both are empty."""
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0


def filter_and_dedupe(masks, scores, config):
    """Score-filter, NMS-deduplicate and top-K a set of candidate masks.

    Returns (kept_masks, kept_scores) with masks ordered by descending score.
    """
    if masks.shape[0] == 0:
        return masks, []

    scores = np.asarray(scores, dtype=np.float64)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    keep = (scores >= config.pred_iou_thresh) & (areas >= config.min_mask_area)

    candidates = np.argsort(-scores)
    candidates = [i for i in candidates if keep[i]]

    selected = []
    for i in candidates:
        if all(mask_iou(masks[i], masks[j]) <= config.nms_iou_thresh for j in selected):
            selected.append(i)
        if len(selected) >= config.top_k:
            break

    if not selected:
        return np.zeros((0, *masks.shape[1:]), dtype=bool), []

    return masks[selected], [float(scores[i]) for i in selected]


def match_to_existing(proposals, existing, iou_thresh):
    """Return indices of proposals not matching any existing mask above iou_thresh.

    Used by the re-seeder: on a 360 orbit, frame 0 observes only part of the
    scene, so new proposals appearing later must be promoted to new objects
    unless they are already being tracked.
    """
    unmatched = []
    for i in range(proposals.shape[0]):
        if existing.shape[0] == 0:
            unmatched.append(i)
            continue
        if max(mask_iou(proposals[i], existing[j]) for j in range(existing.shape[0])) < iou_thresh:
            unmatched.append(i)
    return unmatched


def mask_to_interior_point(mask):
    """Return an (row, col) point well inside the mask, or None if it is empty.

    SAM 3.1 registers new objects from *point* prompts, not mask prompts, so the
    video runner's mask-based interface has to be converted. A centroid is wrong
    here: for a C-shaped or ring-shaped mask it lands in the hole, outside the
    object. The distance-transform maximum is the point furthest from any
    boundary, which is always inside.
    """
    import cv2

    if not mask.any():
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    row, col = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(row), int(col)
