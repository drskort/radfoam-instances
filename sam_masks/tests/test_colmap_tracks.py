import numpy as np
import pytest

from sam_masks.colmap_tracks import Observations, observations_from_tracks


def test_scales_coordinates_by_downsample_factor():
    tracks = {
        100: [("a.JPG", np.array([40.0, 80.0]))],
    }
    name_to_frame = {"a.JPG": 0}

    obs = observations_from_tracks(
        tracks, name_to_frame, downsample=4, shape=(30, 30)
    )

    # 40/4 = 10, 80/4 = 20 -> (col 10, row 20)
    assert obs.xy[0] == (0, 100, 20, 10)


def test_drops_observations_outside_image_bounds():
    tracks = {
        1: [("a.JPG", np.array([1000.0, 4.0])), ("a.JPG", np.array([4.0, 4.0]))],
    }
    obs = observations_from_tracks(
        tracks, {"a.JPG": 0}, downsample=1, shape=(10, 10)
    )

    assert len(obs.xy) == 1
    assert obs.xy[0][3] == 4


def test_drops_images_not_in_the_frame_sequence():
    tracks = {1: [("a.JPG", np.array([2.0, 2.0])), ("ghost.JPG", np.array([2.0, 2.0]))]}

    obs = observations_from_tracks(tracks, {"a.JPG": 0}, downsample=1, shape=(10, 10))

    assert [o[0] for o in obs.xy] == [0]


def test_by_frame_groups_observations():
    tracks = {
        1: [("a.JPG", np.array([2.0, 2.0])), ("b.JPG", np.array([3.0, 3.0]))],
        2: [("a.JPG", np.array([4.0, 4.0]))],
    }

    obs = observations_from_tracks(
        tracks, {"a.JPG": 0, "b.JPG": 1}, downsample=1, shape=(10, 10)
    )

    assert set(obs.by_frame[0]) == {1, 2}
    assert set(obs.by_frame[1]) == {1}


def test_frames_of_point_lists_all_observing_frames():
    tracks = {7: [("a.JPG", np.array([1.0, 1.0])), ("b.JPG", np.array([1.0, 1.0]))]}

    obs = observations_from_tracks(
        tracks, {"a.JPG": 0, "b.JPG": 1}, downsample=1, shape=(10, 10)
    )

    assert sorted(obs.frames_of_point[7]) == [0, 1]


def test_rejects_non_positive_downsample():
    with pytest.raises(ValueError, match="downsample"):
        observations_from_tracks({}, {}, downsample=0, shape=(10, 10))
