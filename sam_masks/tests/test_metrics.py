import numpy as np
import pytest

from sam_masks.metrics import (
    cosegmentation_agreement,
    label_purity,
    sample_point_labels,
    stability_descriptors,
)


def test_sample_point_labels_reads_id_at_pixel():
    labels = np.zeros((10, 10), dtype=np.uint16)
    labels[5, 5] = 42

    got = sample_point_labels(labels, [(1, 5, 5), (2, 0, 0)])

    assert got == {1: 42, 2: 0}


def test_perfectly_consistent_labelling_scores_one():
    # Two frames, four points, identical grouping and identical ids.
    per_frame = {
        0: {1: 10, 2: 10, 3: 20, 4: 20},
        1: {1: 10, 2: 10, 3: 20, 4: 20},
    }

    assert label_purity(per_frame)["mean_purity"] == pytest.approx(1.0)


def test_shuffled_labelling_scores_below_perfect():
    per_frame = {
        0: {1: 10, 2: 10, 3: 20, 4: 20},
        1: {1: 20, 2: 10, 3: 10, 4: 20},
    }

    assert label_purity(per_frame)["mean_purity"] < 1.0


def test_purity_ignores_background_observations():
    # Point 3 falls on background (0) in frame 1; it must not count as a label.
    per_frame = {0: {1: 10, 3: 20}, 1: {1: 10, 3: 0}}

    result = label_purity(per_frame)

    assert result["mean_purity"] == pytest.approx(1.0)
    assert result["n_points"] == 2


def test_purity_can_exclude_points_seen_in_a_single_view():
    # Point 3 survives in one view only, where it is trivially pure. Raising
    # min_observations drops it so mean_purity reflects real cross-view evidence.
    per_frame = {0: {1: 10, 3: 20}, 1: {1: 20, 3: 0}}

    assert label_purity(per_frame)["mean_purity"] == pytest.approx(0.75)

    strict = label_purity(per_frame, min_observations=2)

    assert strict["n_points"] == 1
    assert strict["mean_purity"] == pytest.approx(0.5)


def test_cosegmentation_perfect_when_grouping_is_preserved():
    # Ids differ between frames but the *grouping* is identical, which is what
    # this metric measures.
    per_frame = {
        0: {1: 10, 2: 10, 3: 20, 4: 20},
        1: {1: 77, 2: 77, 3: 88, 4: 88},
    }

    result = cosegmentation_agreement(per_frame, rng=np.random.default_rng(0))

    assert result["agreement"] == pytest.approx(1.0)
    assert result["same_pair_agreement"] == pytest.approx(1.0)
    assert result["diff_pair_agreement"] == pytest.approx(1.0)


def test_cosegmentation_penalises_broken_grouping():
    per_frame = {
        0: {1: 10, 2: 10, 3: 20, 4: 20},
        1: {1: 10, 2: 20, 3: 10, 4: 20},
    }

    result = cosegmentation_agreement(per_frame, rng=np.random.default_rng(0))

    assert result["agreement"] < 1.0


def test_degenerate_single_mask_fails_the_decomposition():
    # Everything in one mask in frame 1: same-pair agreement is perfect, but
    # diff-pair agreement collapses. The balanced score must expose this.
    per_frame = {
        0: {1: 10, 2: 10, 3: 20, 4: 20},
        1: {1: 99, 2: 99, 3: 99, 4: 99},
    }

    result = cosegmentation_agreement(per_frame, rng=np.random.default_rng(0))

    assert result["same_pair_agreement"] == pytest.approx(1.0)
    assert result["diff_pair_agreement"] == pytest.approx(0.0)
    assert result["balanced"] == pytest.approx(0.5)


def test_cosegmentation_needs_two_common_points():
    per_frame = {0: {1: 10}, 1: {1: 10}}

    result = cosegmentation_agreement(per_frame, rng=np.random.default_rng(0))

    assert result["n_view_pairs"] == 0
    assert np.isnan(result["agreement"])


def test_cosegmentation_respects_max_pairs_budget():
    rng = np.random.default_rng(0)
    per_frame = {
        0: {i: i % 5 + 1 for i in range(200)},
        1: {i: i % 5 + 1 for i in range(200)},
    }

    result = cosegmentation_agreement(per_frame, rng=rng, max_pairs=50)

    assert result["n_pairs"] <= 50


def test_stability_descriptors_report_counts_and_areas():
    counts = [3, 5, 4]
    areas = [[10, 20, 30], [1, 2, 3, 4, 5], [7, 7, 7, 7]]

    result = stability_descriptors(counts, areas)

    assert result["mean_masks_per_frame"] == pytest.approx(4.0)
    assert result["min_masks_per_frame"] == 3
    assert result["max_masks_per_frame"] == 5
    assert result["median_mask_area"] == pytest.approx(np.median(sum(areas, [])))


def test_stability_handles_no_frames():
    result = stability_descriptors([], [])
    assert np.isnan(result["mean_masks_per_frame"])


def test_total_degeneracy_is_undefined_not_perfect():
    # Every point in one mask in EVERY view: no differing pairs exist anywhere,
    # so there is no evidence the segmentation separates anything. Scoring this
    # 1.0 would tie the worst possible output with a perfect one.
    per_frame = {
        0: {1: 99, 2: 99, 3: 99, 4: 99},
        1: {1: 99, 2: 99, 3: 99, 4: 99},
    }

    result = cosegmentation_agreement(per_frame, rng=np.random.default_rng(0))

    assert result["same_pair_agreement"] == pytest.approx(1.0)
    assert np.isnan(result["diff_pair_agreement"])
    assert np.isnan(result["balanced"])
