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


class _FakeSession:
    """Records what the runner asks of a session, and forgets tracks over time."""

    def __init__(self, n_frames, lifetime):
        self.n_frames = n_frames
        self.lifetime = lifetime          # frames an object survives after seeding
        self.added, self.removed = [], []
        self._next_id = 1
        self._born = {}

    def add_masks(self, frame_idx, masks):
        ids = list(range(self._next_id, self._next_id + masks.shape[0]))
        self._next_id += masks.shape[0]
        for i in ids:
            self._born[i] = frame_idx
        self.added.append((frame_idx, len(ids)))
        return ids

    def remove_objects(self, obj_ids, frame_idx=0):
        self.removed.append((frame_idx, sorted(obj_ids)))
        for o in obj_ids:
            self._born.pop(o, None)

    def propagate(self, start_frame=None, max_frames=None):
        start = start_frame or 0
        stop = min(self.n_frames, start + (max_frames or self.n_frames))
        for f in range(start, stop):
            alive = {
                o: np.ones((4, 4), dtype=bool)
                for o, born in self._born.items()
                if f - born < self.lifetime
            }
            yield f, alive

    def remove_stale(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _FakeBackend:
    name = "fake"

    def __init__(self, n_frames, lifetime, n_proposals):
        self.session = _FakeSession(n_frames, lifetime)
        self.n_proposals = n_proposals

    def propose_masks(self, image, image_path=None):
        masks = np.zeros((self.n_proposals, 4, 4), dtype=bool)
        for i in range(self.n_proposals):
            masks[i, i % 4, i % 4] = True
        return masks, [0.9] * self.n_proposals

    def start_session(self, frames_dir):
        return self.session


def _fake_scene(tmp_path, n_frames):
    from PIL import Image

    src = tmp_path / "images_4"
    src.mkdir(parents=True)
    for i in range(n_frames):
        Image.new("RGB", (4, 4)).save(src / f"img{i:03d}.jpg")
    return src


def test_runner_prunes_dead_tracks_before_reseeding(tmp_path, monkeypatch):
    """The runner must release tracks that died, not just stop seeding.

    Objects here survive 3 frames after being seeded, so by the next re-seed
    they are long gone. If the runner does not prune, the model's object budget
    fills with dead tracks and seeding stops -- which is exactly what saturated
    SAM 3.1 around frame 20 of 185 before this was added.
    """
    from sam_masks import run_video

    n_frames = 24
    src = _fake_scene(tmp_path, n_frames)
    backend = _FakeBackend(n_frames=n_frames, lifetime=3, n_proposals=5)

    monkeypatch.setattr(run_video, "scene_image_dir", lambda scene: src)
    monkeypatch.setattr(run_video, "get_backend", lambda *a, **k: backend)
    monkeypatch.setitem(run_video.DOWNSAMPLE, "garden", 4)

    out = run_video.run(
        "garden", "fake", output_root=tmp_path / "out",
        reseed_every=6, stale_after=3, max_objects=12, limit=n_frames,
    )

    from sam_masks.store import read_meta

    meta = read_meta(out)
    assert meta["complete"], "run did not finish"
    assert meta["n_pruned_total"] > 0, "no tracks were ever pruned"
    assert backend.session.removed, "session.remove_objects was never called"
    # Seeding must keep happening after the cap would otherwise have filled:
    # 4 seeds x 5 proposals = 20 objects against a cap of 12.
    assert len(backend.session.added) == 4, "seeding stopped early"


def test_runner_does_not_prune_live_tracks(tmp_path, monkeypatch):
    """Objects still being tracked must survive re-seeding untouched."""
    from sam_masks import run_video

    n_frames = 18
    src = _fake_scene(tmp_path, n_frames)
    # lifetime longer than the whole run: nothing ever goes stale.
    backend = _FakeBackend(n_frames=n_frames, lifetime=999, n_proposals=3)

    monkeypatch.setattr(run_video, "scene_image_dir", lambda scene: src)
    monkeypatch.setattr(run_video, "get_backend", lambda *a, **k: backend)
    monkeypatch.setitem(run_video.DOWNSAMPLE, "garden", 4)

    out = run_video.run(
        "garden", "fake", output_root=tmp_path / "out",
        reseed_every=6, stale_after=3, max_objects=100, limit=n_frames,
    )

    from sam_masks.store import read_meta

    assert read_meta(out)["n_pruned_total"] == 0
    assert backend.session.removed == []


def test_completed_short_run_does_not_satisfy_a_longer_request(tmp_path, monkeypatch):
    """A finished 12-frame arm must not let a 24-frame request skip.

    Both carry complete=True; only the frame count distinguishes them, and
    without that check the longer run silently returns a truncated arm.
    """
    from sam_masks import run_video
    from sam_masks.store import read_meta

    src = _fake_scene(tmp_path, 24)
    monkeypatch.setattr(run_video, "scene_image_dir", lambda scene: src)
    monkeypatch.setitem(run_video.DOWNSAMPLE, "garden", 4)

    short = _FakeBackend(n_frames=24, lifetime=999, n_proposals=2)
    monkeypatch.setattr(run_video, "get_backend", lambda *a, **k: short)
    out = run_video.run("garden", "fake", output_root=tmp_path / "out",
                        reseed_every=6, limit=12)
    assert read_meta(out)["n_frames"] == 12

    long = _FakeBackend(n_frames=24, lifetime=999, n_proposals=2)
    monkeypatch.setattr(run_video, "get_backend", lambda *a, **k: long)
    out = run_video.run("garden", "fake", output_root=tmp_path / "out",
                        reseed_every=6, limit=24)

    assert read_meta(out)["n_frames"] == 24, "short run wrongly satisfied a longer request"
    assert long.session.added, "backend was never invoked for the longer run"
