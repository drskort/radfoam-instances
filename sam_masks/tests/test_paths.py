import pytest

from sam_masks.paths import resolve_output_root, scene_image_dir, DOWNSAMPLE


def test_prefers_local_work_when_present(tmp_path):
    local = tmp_path / "work" / "user"
    local.mkdir(parents=True)
    node = tmp_path / "nodes" / "host" / "work" / "user"
    node.mkdir(parents=True)
    assert resolve_output_root(candidates=[local, node]) == local


def test_falls_back_to_node_path(tmp_path):
    local = tmp_path / "work" / "user"  # deliberately not created
    node = tmp_path / "nodes" / "host" / "work" / "user"
    node.mkdir(parents=True)
    assert resolve_output_root(candidates=[local, node]) == node


def test_raises_when_no_candidate_exists(tmp_path):
    # Two levels deep, so neither the candidate nor its parent exists.
    with pytest.raises(FileNotFoundError, match="No writable output root"):
        resolve_output_root(candidates=[tmp_path / "nope" / "deeper"])


def test_outdoor_scenes_use_images_4():
    assert DOWNSAMPLE["garden"] == 4
    assert DOWNSAMPLE["bicycle"] == 4
    assert DOWNSAMPLE["stump"] == 4


def test_indoor_scenes_use_images_2():
    assert DOWNSAMPLE["room"] == 2
    assert DOWNSAMPLE["bonsai"] == 2
    assert DOWNSAMPLE["counter"] == 2
    assert DOWNSAMPLE["kitchen"] == 2


def test_scene_image_dir_uses_downsample_factor(tmp_path):
    (tmp_path / "garden" / "images_4").mkdir(parents=True)
    assert scene_image_dir("garden", root=tmp_path) == tmp_path / "garden" / "images_4"


def test_scene_image_dir_rejects_unknown_scene(tmp_path):
    with pytest.raises(KeyError, match="atlantis"):
        scene_image_dir("atlantis", root=tmp_path)
