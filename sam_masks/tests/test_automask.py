import numpy as np
import pytest

from sam_masks.automask import (
    AutomaskConfig,
    filter_and_dedupe,
    mask_iou,
    mask_to_interior_point,
    match_to_existing,
    point_grid,
)


def test_point_grid_has_expected_count():
    grid = point_grid(4, height=100, width=200)
    assert grid.shape == (16, 2)


def test_point_grid_points_lie_inside_the_image():
    grid = point_grid(8, height=50, width=70)
    assert grid[:, 0].min() >= 0 and grid[:, 0].max() < 70
    assert grid[:, 1].min() >= 0 and grid[:, 1].max() < 50


def test_point_grid_is_evenly_spaced():
    grid = point_grid(2, height=100, width=100)
    xs = sorted(set(grid[:, 0].tolist()))
    assert xs == [25, 75]


def test_mask_iou_of_identical_masks_is_one():
    m = np.zeros((10, 10), dtype=bool)
    m[2:6, 2:6] = True
    assert mask_iou(m, m) == pytest.approx(1.0)


def test_mask_iou_of_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    assert mask_iou(a, b) == pytest.approx(0.0)


def test_mask_iou_of_two_empty_masks_is_zero():
    empty = np.zeros((5, 5), dtype=bool)
    assert mask_iou(empty, empty) == pytest.approx(0.0)


def test_filter_drops_masks_below_score_threshold():
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 0:5, 0:5] = True
    masks[1, 5:10, 5:10] = True
    cfg = AutomaskConfig(pred_iou_thresh=0.85)

    kept, scores = filter_and_dedupe(masks, np.array([0.9, 0.1]), cfg)

    assert kept.shape[0] == 1
    assert scores == pytest.approx([0.9])


def test_nms_removes_near_duplicate_masks():
    base = np.zeros((10, 10), dtype=bool)
    base[2:8, 2:8] = True
    near = base.copy()
    near[2, 2] = False
    masks = np.stack([base, near])
    cfg = AutomaskConfig(pred_iou_thresh=0.0, nms_iou_thresh=0.7)

    kept, _ = filter_and_dedupe(masks, np.array([0.9, 0.8]), cfg)

    assert kept.shape[0] == 1


def test_nms_keeps_the_higher_scoring_duplicate():
    a = np.zeros((10, 10), dtype=bool)
    a[2:8, 2:8] = True
    b = a.copy()
    b[2, 2] = False
    masks = np.stack([a, b])
    cfg = AutomaskConfig(pred_iou_thresh=0.0, nms_iou_thresh=0.7)

    kept, scores = filter_and_dedupe(masks, np.array([0.4, 0.95]), cfg)

    assert scores == pytest.approx([0.95])
    assert kept[0].sum() == b.sum()


def test_top_k_caps_the_proposal_count():
    masks = np.zeros((5, 20, 20), dtype=bool)
    for i in range(5):
        masks[i, i * 4 : i * 4 + 3, :] = True
    cfg = AutomaskConfig(pred_iou_thresh=0.0, nms_iou_thresh=0.9, top_k=2)

    kept, scores = filter_and_dedupe(masks, np.array([0.1, 0.9, 0.5, 0.7, 0.3]), cfg)

    assert kept.shape[0] == 2
    assert scores == pytest.approx([0.9, 0.7])


def test_empty_input_returns_empty_output():
    cfg = AutomaskConfig()
    kept, scores = filter_and_dedupe(
        np.zeros((0, 10, 10), dtype=bool), np.array([]), cfg
    )
    assert kept.shape == (0, 10, 10)
    assert scores == []


def test_drops_empty_masks():
    masks = np.zeros((2, 10, 10), dtype=bool)
    masks[0, 1:3, 1:3] = True
    cfg = AutomaskConfig(pred_iou_thresh=0.0)

    kept, _ = filter_and_dedupe(masks, np.array([0.9, 0.9]), cfg)

    assert kept.shape[0] == 1


def test_match_to_existing_pairs_overlapping_masks():
    existing = np.zeros((1, 10, 10), dtype=bool)
    existing[0, 2:8, 2:8] = True
    proposals = np.zeros((2, 10, 10), dtype=bool)
    proposals[0, 2:8, 2:8] = True      # matches
    proposals[1, 0:2, 0:2] = True      # does not

    unmatched = match_to_existing(proposals, existing, iou_thresh=0.5)

    assert unmatched == [1]


def test_match_to_existing_returns_all_when_nothing_tracked():
    proposals = np.zeros((2, 10, 10), dtype=bool)
    proposals[0, 0:2, 0:2] = True
    proposals[1, 5:7, 5:7] = True

    unmatched = match_to_existing(proposals, np.zeros((0, 10, 10), dtype=bool), 0.5)

    assert unmatched == [0, 1]


def test_interior_point_lies_inside_the_mask():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True

    row, col = mask_to_interior_point(mask)

    assert mask[row, col]


def test_interior_point_of_a_c_shape_is_not_in_the_hole():
    # A C shape: its centroid falls in the notch, outside the mask. The
    # distance transform must pick a point in the material instead.
    mask = np.zeros((21, 21), dtype=bool)
    mask[4:17, 4:17] = True
    mask[8:13, 8:21] = False

    row, col = mask_to_interior_point(mask)

    assert mask[row, col]


def test_interior_point_of_empty_mask_is_none():
    assert mask_to_interior_point(np.zeros((10, 10), dtype=bool)) is None
