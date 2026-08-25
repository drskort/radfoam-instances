"""Model-independent everything-mode proposal logic.

SAM 3.1 ships no automatic mask generator, so this is the shared half of the one
we build; SAM 2.1's generator does the same work internally. Keeping it here, and
using it for both backends, means proposal behaviour is comparable across arms by
construction rather than by coincidence.

Threshold defaults match SAM's published automatic mask generator. The point
grid does not -- see AutomaskConfig.points_per_side for the measurements that
set it.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AutomaskConfig:
    # 16x16, set by measurement rather than by SAM's published 32x32. SAM 3.1
    # tracks every grid point as a separate object with its own memory state, so
    # cost grows superlinearly in points and 32x32 (1024 objects) OOMs a 48 GB
    # A40. Measured on garden at 1297x840, one image:
    #     16x16  256 pts   27 s  17.5 GiB  239 masks  95.2% coverage
    #     20x20  400 pts   67 s  25.0 GiB  256 masks  96.5% coverage
    #     24x24  576 pts  142 s  33.9 GiB  256 masks  97.7% coverage
    #     32x32 1024 pts   OOM
    # At 20x20 and above the kept-mask count is exactly top_k, i.e. the extra
    # proposals are discarded by our own filtering -- 5.3x the compute for +2.5
    # points of coverage. 16x16 lands just under the cap, so nothing is wasted.
    points_per_side: int = 16
    pred_iou_thresh: float = 0.8
    stability_thresh: float = 0.95   # SAM2AutomaticMaskGenerator's own default
    nms_iou_thresh: float = 0.7
    top_k: int = 256
    min_mask_area: int = 1


# Named threshold sets. The arm directory is suffixed with the tag, and every
# reported number was produced with `t70` -- so the tag has to *imply* these
# settings rather than merely label them. Passing --tag t70 without them used to
# write default-threshold masks into a directory called t70, silently.
TAG_PRESETS = {
    "t70": dict(points_per_side=32, pred_iou_thresh=0.70, stability_thresh=0.88),
    "t60": dict(points_per_side=32, pred_iou_thresh=0.60, stability_thresh=0.80),
}


def apply_tag_preset(config, tag):
    """Fold a named preset into `config`, returning the names it set."""
    preset = TAG_PRESETS.get(tag)
    if not preset:
        return ()
    for key, value in preset.items():
        setattr(config, key, value)
    return tuple(preset)


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


def _bounding_boxes(masks):
    """Return (N, 4) int array of [row0, row1, col0, col1] half-open boxes.

    Empty masks get a degenerate box that intersects nothing.
    """
    boxes = np.zeros((masks.shape[0], 4), dtype=np.int64)
    for i, mask in enumerate(masks):
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size:
            boxes[i] = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)
    return boxes


def _iou_boxed(a, b, area_a, area_b, box_a, box_b):
    """Exact IoU of two masks, computed only inside their overlapping box.

    Identical results to mask_iou, but skips disjoint pairs outright and never
    scans the full image. Pairwise IoU over a few hundred masks at 840x1297 is
    the pipeline's hot loop: comparing whole images cost ~19 s per frame, which
    at 300 frames x 4 arms would have been ~6 hours of pure post-processing.
    """
    row0 = max(box_a[0], box_b[0])
    row1 = min(box_a[1], box_b[1])
    col0 = max(box_a[2], box_b[2])
    col1 = min(box_a[3], box_b[3])
    if row0 >= row1 or col0 >= col1:
        return 0.0            # disjoint boxes cannot overlap

    intersection = np.count_nonzero(
        np.logical_and(a[row0:row1, col0:col1], b[row0:row1, col0:col1])
    )
    union = area_a + area_b - intersection
    return float(intersection / union) if union else 0.0


def select_masks(masks, scores, config):
    """Return the indices kept by score filtering, NMS and top-K, best first.

    Split out from filter_and_dedupe so callers carrying per-mask side data --
    the SAM decoder level, for instance -- can apply the identical selection
    rather than reimplementing it and risking divergence.
    """
    if masks.shape[0] == 0:
        return []

    scores = np.asarray(scores, dtype=np.float64)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    keep = (scores >= config.pred_iou_thresh) & (areas >= config.min_mask_area)

    candidates = np.argsort(-scores)
    candidates = [i for i in candidates if keep[i]]

    boxes = _bounding_boxes(masks)
    selected = []
    for i in candidates:
        if all(
            _iou_boxed(masks[i], masks[j], areas[i], areas[j], boxes[i], boxes[j])
            <= config.nms_iou_thresh
            for j in selected
        ):
            selected.append(i)
        if len(selected) >= config.top_k:
            break

    return selected


def filter_and_dedupe(masks, scores, config):
    """Score-filter, NMS-deduplicate and top-K a set of candidate masks.

    Returns (kept_masks, kept_scores) with masks ordered by descending score.
    """
    selected = select_masks(masks, scores, config)
    if not selected:
        return np.zeros((0, *masks.shape[1:]), dtype=bool), []
    scores = np.asarray(scores, dtype=np.float64)
    return masks[selected], [float(scores[i]) for i in selected]


def match_to_existing(proposals, existing, iou_thresh):
    """Return indices of proposals not matching any existing mask above iou_thresh.

    Used by the re-seeder: on a 360 orbit, frame 0 observes only part of the
    scene, so new proposals appearing later must be promoted to new objects
    unless they are already being tracked.
    """
    if existing.shape[0] == 0:
        return list(range(proposals.shape[0]))

    proposal_areas = proposals.reshape(proposals.shape[0], -1).sum(axis=1)
    existing_areas = existing.reshape(existing.shape[0], -1).sum(axis=1)
    proposal_boxes = _bounding_boxes(proposals)
    existing_boxes = _bounding_boxes(existing)

    unmatched = []
    for i in range(proposals.shape[0]):
        best = max(
            _iou_boxed(
                proposals[i],
                existing[j],
                proposal_areas[i],
                existing_areas[j],
                proposal_boxes[i],
                existing_boxes[j],
            )
            for j in range(existing.shape[0])
        )
        if best < iou_thresh:
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
