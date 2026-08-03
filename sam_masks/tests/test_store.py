import json

import numpy as np
import pytest

from sam_masks.store import FrameMasks, load_frame, save_frame, write_meta, read_meta


def circle_mask(h, w, cy, cx, r):
    y, x = np.ogrid[:h, :w]
    return ((y - cy) ** 2 + (x - cx) ** 2) <= r * r


def test_rle_round_trip_preserves_masks_exactly(tmp_path):
    masks = np.stack([circle_mask(40, 60, 10, 15, 6), circle_mask(40, 60, 30, 45, 8)])
    frame = FrameMasks(
        frame_idx=7, masks=masks, obj_ids=[3, 9], scores=[0.9, 0.8], shape=(40, 60)
    )

    save_frame(tmp_path, frame)
    loaded = load_frame(tmp_path, 7)

    np.testing.assert_array_equal(loaded.masks, masks)
    assert loaded.obj_ids == [3, 9]
    assert loaded.frame_idx == 7
    np.testing.assert_allclose(loaded.scores, [0.9, 0.8])


def test_handles_empty_frame(tmp_path):
    frame = FrameMasks(
        frame_idx=0,
        masks=np.zeros((0, 20, 20), dtype=bool),
        obj_ids=[],
        scores=[],
        shape=(20, 20),
    )

    save_frame(tmp_path, frame)
    loaded = load_frame(tmp_path, 0)

    assert loaded.masks.shape == (0, 20, 20)
    assert loaded.obj_ids == []


def test_writes_label_png_painted_largest_first(tmp_path):
    big = np.zeros((10, 10), dtype=bool)
    big[:, :] = True
    small = np.zeros((10, 10), dtype=bool)
    small[2:5, 2:5] = True
    frame = FrameMasks(
        frame_idx=0,
        masks=np.stack([big, small]),
        obj_ids=[1, 2],
        scores=[0.5, 0.6],
        shape=(10, 10),
    )

    save_frame(tmp_path, frame)

    from PIL import Image

    labels = np.array(Image.open(tmp_path / "labels" / "000000.png"))
    assert labels.dtype == np.uint16
    # The small mask is painted after the big one, so it wins where they overlap.
    assert labels[3, 3] == 2
    assert labels[0, 0] == 1


def test_label_png_uses_zero_for_background(tmp_path):
    m = np.zeros((8, 8), dtype=bool)
    m[1:3, 1:3] = True
    frame = FrameMasks(
        frame_idx=0, masks=m[None], obj_ids=[5], scores=[0.5], shape=(8, 8)
    )

    save_frame(tmp_path, frame)

    from PIL import Image

    labels = np.array(Image.open(tmp_path / "labels" / "000000.png"))
    assert labels[7, 7] == 0
    assert labels[1, 1] == 5


def test_rejects_object_id_exceeding_uint16(tmp_path):
    m = np.ones((4, 4), dtype=bool)
    frame = FrameMasks(
        frame_idx=0, masks=m[None], obj_ids=[70000], scores=[0.5], shape=(4, 4)
    )
    with pytest.raises(ValueError, match="uint16"):
        save_frame(tmp_path, frame)


def test_rejects_mismatched_lengths(tmp_path):
    masks = np.zeros((2, 4, 4), dtype=bool)
    frame = FrameMasks(
        frame_idx=0, masks=masks, obj_ids=[1], scores=[0.5, 0.5], shape=(4, 4)
    )
    with pytest.raises(ValueError, match="same length"):
        save_frame(tmp_path, frame)


def test_meta_round_trips(tmp_path):
    meta = {"model": "sam31", "mode": "video", "scene": "garden", "reseed_every": 10}
    write_meta(tmp_path, meta)
    assert read_meta(tmp_path) == meta


def test_meta_is_readable_json(tmp_path):
    write_meta(tmp_path, {"model": "sam21"})
    assert json.loads((tmp_path / "meta.json").read_text())["model"] == "sam21"
