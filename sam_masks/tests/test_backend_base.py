import numpy as np
import pytest

from sam_masks.backends.base import Backend, Session


class FakeSession(Session):
    def __init__(self):
        self.registered = {}
        self._next_id = 1

    def add_masks(self, frame_idx, masks):
        obj_ids = list(range(self._next_id, self._next_id + masks.shape[0]))
        self._next_id += masks.shape[0]
        self.registered[frame_idx] = obj_ids
        return obj_ids

    def propagate(self, start_frame=None, max_frames=None):
        yield 0, {1: np.ones((4, 4), dtype=bool)}

    def remove_objects(self, obj_ids, frame_idx=0):
        self.removed = list(obj_ids)

    def close(self):
        pass


class FakeBackend(Backend):
    name = "fake"

    def propose_masks(self, image, image_path=None):
        return np.ones((1, 4, 4), dtype=bool), [0.9]

    def start_session(self, frames_dir):
        return FakeSession()


def test_backend_exposes_a_name():
    assert FakeBackend().name == "fake"


def test_propose_masks_returns_masks_and_scores():
    masks, scores = FakeBackend().propose_masks(np.zeros((4, 4, 3), dtype=np.uint8))
    assert masks.shape == (1, 4, 4)
    assert scores == [0.9]


def test_add_masks_returns_one_id_per_mask():
    session = FakeBackend().start_session("frames")
    obj_ids = session.add_masks(0, np.ones((3, 4, 4), dtype=bool))
    assert len(obj_ids) == 3


def test_add_masks_ids_are_unique_across_calls():
    session = FakeBackend().start_session("frames")
    first = session.add_masks(0, np.ones((2, 4, 4), dtype=bool))
    second = session.add_masks(5, np.ones((2, 4, 4), dtype=bool))
    assert set(first).isdisjoint(second)


def test_propagate_yields_frame_index_and_id_to_mask_mapping():
    session = FakeBackend().start_session("frames")
    frame_idx, masks_by_id = next(iter(session.propagate()))
    assert frame_idx == 0
    assert list(masks_by_id) == [1]
    assert masks_by_id[1].dtype == bool


def test_base_backend_methods_are_abstract():
    with pytest.raises(TypeError):
        Backend()


class StubSam2Predictor:
    """Mimics SAM 2's padding behaviour: a row per registered object, always."""

    def __init__(self, logits_per_frame):
        self.logits_per_frame = logits_per_frame

    def init_state(self, video_path):
        return {"video_path": video_path}

    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        return None

    def propagate_in_video(self, state, start_frame_idx=None, max_frame_num_to_track=None):
        start = start_frame_idx or 0
        for frame_idx, logits in enumerate(self.logits_per_frame):
            if frame_idx < start:
                continue
            if max_frame_num_to_track is not None and frame_idx - start >= max_frame_num_to_track:
                return
            yield frame_idx, [1, 2], logits

    def reset_state(self, state):
        return None


def test_sam21_session_drops_lost_tracks_instead_of_padding():
    # SAM 2 returns a zero-filled row for a lost track rather than omitting it.
    # Object 2 is alive on frame 0 and lost on frame 1.
    import torch

    from sam_masks.backends.sam21 import Sam21Session

    alive = torch.full((2, 1, 4, 4), 1.0)
    one_lost = torch.stack(
        [torch.full((1, 4, 4), 1.0), torch.full((1, 4, 4), -1.0)]
    )
    session = Sam21Session(StubSam2Predictor([alive, one_lost]), "frames")

    frames = dict(session.propagate())

    assert sorted(frames[0]) == [1, 2], "both objects alive on frame 0"
    assert sorted(frames[1]) == [1], "lost track must be absent, not empty"
    assert all(m.any() for masks in frames.values() for m in masks.values())


def test_sam21_session_propagate_honours_frame_range():
    # The video runner propagates in segments so it can compare new proposals
    # against live masks at the seed viewpoint; the range must be respected.
    import torch

    from sam_masks.backends.sam21 import Sam21Session

    frames = [torch.full((2, 1, 4, 4), 1.0) for _ in range(5)]
    session = Sam21Session(StubSam2Predictor(frames), "frames")

    got = [f for f, _ in session.propagate(start_frame=1, max_frames=2)]

    assert got == [1, 2]
