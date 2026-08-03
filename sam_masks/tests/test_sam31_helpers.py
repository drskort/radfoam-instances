import numpy as np
import pytest

from sam_masks.backends.sam31 import masks_to_normalized_points


def test_converts_each_mask_to_one_normalized_point():
    masks = np.zeros((2, 10, 20), dtype=bool)
    masks[0, 2:6, 2:6] = True
    masks[1, 6:9, 12:18] = True

    points, kept = masks_to_normalized_points(masks)

    assert len(points) == 2
    assert kept == [0, 1]


def test_normalized_points_are_in_unit_range():
    masks = np.zeros((1, 10, 20), dtype=bool)
    masks[0, 2:6, 2:6] = True

    points, _ = masks_to_normalized_points(masks)

    x, y = points[0]
    assert 0.0 <= x <= 1.0
    assert 0.0 <= y <= 1.0


def test_point_order_is_x_then_y():
    # A mask in the far right of a wide image must have x near 1, y near 0.5.
    masks = np.zeros((1, 10, 100), dtype=bool)
    masks[0, 4:6, 90:98] = True

    points, _ = masks_to_normalized_points(masks)

    x, y = points[0]
    assert x > 0.8
    assert 0.3 < y < 0.7


def test_empty_masks_are_dropped_and_reported():
    masks = np.zeros((3, 10, 10), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[2, 5:7, 5:7] = True   # index 1 left empty

    points, kept = masks_to_normalized_points(masks)

    assert len(points) == 2
    assert kept == [0, 2]


def test_no_masks_returns_empty():
    points, kept = masks_to_normalized_points(np.zeros((0, 10, 10), dtype=bool))
    assert points == []
    assert kept == []
