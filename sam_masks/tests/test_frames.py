import json

import numpy as np
import pytest

from sam_masks.frames import build_sequence, load_frame_index, pose_order


def make_images(tmp_path, names):
    src = tmp_path / "images_4"
    src.mkdir(parents=True, exist_ok=True)
    for name in names:
        (src / name).write_bytes(b"fake-jpeg")
    return src


def test_builds_sequentially_numbered_symlinks(tmp_path):
    src = make_images(tmp_path, ["DSC03.JPG", "DSC01.JPG", "DSC02.JPG"])
    out = tmp_path / "frames"

    seq = build_sequence(src, out)

    assert [p.name for p in sorted(out.iterdir())] == [
        "000000.jpg",
        "000001.jpg",
        "000002.jpg",
    ]
    assert (out / "000000.jpg").is_symlink()
    assert (out / "000000.jpg").resolve() == (src / "DSC01.JPG").resolve()
    assert seq.names == ["DSC01.JPG", "DSC02.JPG", "DSC03.JPG"]


def test_writes_frame_index_mapping(tmp_path):
    src = make_images(tmp_path, ["b.JPG", "a.JPG"])
    out = tmp_path / "frames"

    build_sequence(src, out)

    index = json.loads((out.parent / "frame_index.json").read_text())
    assert index["names"] == ["a.JPG", "b.JPG"]
    assert index["order"] == "filename"


def test_load_frame_index_round_trips(tmp_path):
    src = make_images(tmp_path, ["b.JPG", "a.JPG"])
    out = tmp_path / "frames"
    seq = build_sequence(src, out)

    assert load_frame_index(out.parent).names == seq.names


def test_rebuild_is_idempotent(tmp_path):
    src = make_images(tmp_path, ["a.JPG", "b.JPG"])
    out = tmp_path / "frames"

    build_sequence(src, out)
    build_sequence(src, out)

    assert len(list(out.iterdir())) == 2


def test_stale_symlinks_are_removed_on_rebuild(tmp_path):
    src = make_images(tmp_path, ["a.JPG", "b.JPG", "c.JPG"])
    out = tmp_path / "frames"
    build_sequence(src, out)
    assert len(list(out.iterdir())) == 3

    (src / "c.JPG").unlink()
    build_sequence(src, out)

    assert [p.name for p in sorted(out.iterdir())] == ["000000.jpg", "000001.jpg"]


def test_raises_on_empty_source(tmp_path):
    src = tmp_path / "images_4"
    src.mkdir()
    with pytest.raises(FileNotFoundError, match="No images"):
        build_sequence(src, tmp_path / "frames")


def test_pose_order_is_a_permutation_of_all_names():
    names = ["a", "b", "c", "d"]
    centres = np.array([[0.0, 0, 0], [10, 0, 0], [1, 0, 0], [11, 0, 0]])

    order = pose_order(names, centres)

    assert sorted(order) == sorted(names)
    assert len(order) == len(names)


def test_pose_order_follows_nearest_neighbour_chain():
    names = ["a", "b", "c", "d"]
    centres = np.array([[0.0, 0, 0], [10, 0, 0], [1, 0, 0], [11, 0, 0]])

    # Greedy from 'a': nearest is 'c' at distance 1, then 'b' at 9, then 'd'.
    assert pose_order(names, centres) == ["a", "c", "b", "d"]


def test_pose_order_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        pose_order(["a", "b"], np.zeros((3, 3)))
