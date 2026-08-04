"""Per-frame mask persistence.

Masks are stored as COCO RLE inside an .npz per frame: everything-mode yields up
to 256 masks per frame over a few hundred frames, and dense boolean arrays would
be about two orders of magnitude larger.

A uint16 label PNG is written alongside, painted largest-area-first so smaller
objects stay visible where masks overlap. It is lossy for overlapping masks and
exists for inspection and downstream radfoam use; the .npz is the source of
truth.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

UINT16_MAX = 65535


@dataclass
class FrameMasks:
    """Masks for one frame.

    masks has shape (N, H, W) and dtype bool. obj_ids are stable across frames in
    the video arms and are per-frame-arbitrary in the image arms.
    """

    frame_idx: int
    masks: np.ndarray
    obj_ids: list
    scores: list
    shape: tuple
    # Optional per-mask granularity level (SAM's decoder emits subpart / part /
    # whole per prompt point). None when the backend has no notion of levels.
    levels: list = None

    def __len__(self):
        return len(self.obj_ids)


def _validate(frame):
    n = len(frame.obj_ids)
    if frame.levels is not None and len(frame.levels) != n:
        raise ValueError(
            f"levels must be the same length as obj_ids, got "
            f"{len(frame.levels)} and {n}"
        )
    if frame.masks.shape[0] != n or len(frame.scores) != n:
        raise ValueError(
            "masks, obj_ids and scores must be the same length, got "
            f"{frame.masks.shape[0]}, {n}, {len(frame.scores)}"
        )
    if any(int(o) > UINT16_MAX or int(o) < 0 for o in frame.obj_ids):
        raise ValueError(
            f"object ids must fit in uint16 (0..{UINT16_MAX}) for the label PNG"
        )


def _encode(masks):
    """COCO RLE-encode an (N, H, W) boolean array."""
    if masks.shape[0] == 0:
        return []
    fortran = np.asfortranarray(masks.transpose(1, 2, 0).astype(np.uint8))
    return mask_utils.encode(fortran)


def _paint_labels(frame):
    """Return a uint16 label map, painted largest-area-first.

    Larger masks are drawn first so that smaller objects sitting on top of them
    remain visible. Background is 0, so object id 0 is not representable; ids
    start at 1 everywhere in this package.
    """
    labels = np.zeros(frame.shape, dtype=np.uint16)
    n = frame.masks.shape[0]
    if n == 0:
        return labels
    areas = frame.masks.reshape(n, -1).sum(axis=1)
    for i in np.argsort(-areas):
        labels[frame.masks[i]] = np.uint16(frame.obj_ids[i])
    return labels


def save_frame(arm_dir, frame):
    """Write masks/NNNNNN.npz and labels/NNNNNN.png under arm_dir."""
    _validate(frame)
    arm_dir = Path(arm_dir)
    (arm_dir / "masks").mkdir(parents=True, exist_ok=True)
    (arm_dir / "labels").mkdir(parents=True, exist_ok=True)

    rles = _encode(frame.masks)
    payload = {
        "counts": np.array([r["counts"] for r in rles], dtype=object),
        "obj_ids": np.array(frame.obj_ids, dtype=np.int64),
        "scores": np.array(frame.scores, dtype=np.float32),
        "shape": np.array(frame.shape, dtype=np.int64),
    }
    if frame.levels is not None:
        payload["levels"] = np.array(frame.levels, dtype=np.int64)
    np.savez_compressed(arm_dir / "masks" / f"{frame.frame_idx:06d}.npz", **payload)

    # Flat label map over every mask, whatever its level.
    Image.fromarray(_paint_labels(frame)).save(
        arm_dir / "labels" / f"{frame.frame_idx:06d}.png"
    )

    # One label map per granularity level. A contrastive instance loss is
    # supervised per view from a label map, and supervising at several
    # granularities lets the learned embedding carry the part/whole hierarchy
    # instead of being forced to commit to one reading of it.
    if frame.levels is not None:
        for level in sorted(set(int(v) for v in frame.levels)):
            keep = [i for i, v in enumerate(frame.levels) if int(v) == level]
            sub = FrameMasks(
                frame_idx=frame.frame_idx,
                masks=frame.masks[keep],
                obj_ids=[frame.obj_ids[i] for i in keep],
                scores=[frame.scores[i] for i in keep],
                shape=frame.shape,
            )
            out = arm_dir / f"labels_l{level}"
            out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_paint_labels(sub)).save(
                out / f"{frame.frame_idx:06d}.png"
            )


def load_frame(arm_dir, frame_idx):
    """Read back a FrameMasks written by save_frame."""
    path = Path(arm_dir) / "masks" / f"{frame_idx:06d}.npz"
    data = np.load(path, allow_pickle=True)
    shape = tuple(int(v) for v in data["shape"])
    obj_ids = [int(v) for v in data["obj_ids"]]

    counts = list(data["counts"])
    if counts:
        rles = [{"size": list(shape), "counts": c} for c in counts]
        decoded = mask_utils.decode(rles)          # (H, W, N) uint8
        masks = decoded.transpose(2, 0, 1).astype(bool)
    else:
        masks = np.zeros((0, *shape), dtype=bool)

    levels = [int(v) for v in data["levels"]] if "levels" in data.files else None
    return FrameMasks(
        frame_idx=frame_idx,
        masks=masks,
        obj_ids=obj_ids,
        scores=[float(v) for v in data["scores"]],
        shape=shape,
        levels=levels,
    )


def write_meta(arm_dir, meta):
    arm_dir = Path(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_meta(arm_dir):
    return json.loads((Path(arm_dir) / "meta.json").read_text())
