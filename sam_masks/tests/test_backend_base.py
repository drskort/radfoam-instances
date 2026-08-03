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

    def propagate(self):
        yield 0, {1: np.ones((4, 4), dtype=bool)}

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
