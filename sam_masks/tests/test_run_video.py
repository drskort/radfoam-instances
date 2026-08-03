import numpy as np
import pytest

from sam_masks.run_video import promote_unmatched, reseed_frames


def test_reseed_frames_starts_at_zero_and_steps():
    assert reseed_frames(n_frames=25, every=10) == [0, 10, 20]


def test_reseed_frames_with_interval_larger_than_sequence():
    assert reseed_frames(n_frames=5, every=10) == [0]


def test_reseed_frames_rejects_non_positive_interval():
    with pytest.raises(ValueError, match="positive"):
        reseed_frames(n_frames=10, every=0)


def test_promote_unmatched_registers_only_new_objects():
    tracked = np.zeros((1, 10, 10), dtype=bool)
    tracked[0, 2:8, 2:8] = True
    proposals = np.zeros((2, 10, 10), dtype=bool)
    proposals[0, 2:8, 2:8] = True      # already tracked
    proposals[1, 0:2, 0:2] = True      # new

    registered = []

    def fake_add(frame_idx, masks):
        registered.append(masks.shape[0])
        return list(range(masks.shape[0]))

    new = promote_unmatched(fake_add, 10, proposals, tracked, iou_thresh=0.5)

    assert new.shape[0] == 1
    assert registered == [1]
    # The returned masks are the ones registered, so the caller can extend its
    # tracked set without re-running the match.
    np.testing.assert_array_equal(new[0], proposals[1])


def test_promote_unmatched_does_nothing_when_all_are_tracked():
    tracked = np.ones((1, 10, 10), dtype=bool)
    proposals = np.ones((1, 10, 10), dtype=bool)

    called = []

    def fake_add(frame_idx, masks):
        called.append(masks.shape[0])
        return []

    new = promote_unmatched(fake_add, 0, proposals, tracked, iou_thresh=0.5)

    assert new.shape[0] == 0
    assert called == [], "must not call the session when nothing is new"


def test_promote_unmatched_registers_everything_on_an_empty_tracker():
    proposals = np.zeros((3, 10, 10), dtype=bool)
    for i in range(3):
        proposals[i, i, i] = True

    def fake_add(frame_idx, masks):
        return list(range(masks.shape[0]))

    new = promote_unmatched(
        fake_add, 0, proposals, np.zeros((0, 10, 10), dtype=bool), 0.5
    )

    assert new.shape[0] == 3


def test_promote_unmatched_handles_no_proposals():
    def fake_add(frame_idx, masks):
        raise AssertionError("must not be called with no proposals")

    new = promote_unmatched(
        fake_add,
        0,
        np.zeros((0, 10, 10), dtype=bool),
        np.zeros((0, 10, 10), dtype=bool),
        0.5,
    )

    assert new.shape == (0, 10, 10)


def test_promote_unmatched_preserves_mask_shape():
    proposals = np.zeros((1, 7, 9), dtype=bool)
    proposals[0, 1, 1] = True

    new = promote_unmatched(
        lambda f, m: [1], 0, proposals, np.zeros((0, 7, 9), dtype=bool), 0.5
    )

    assert new.shape == (1, 7, 9)
