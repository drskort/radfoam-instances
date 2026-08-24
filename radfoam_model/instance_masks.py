"""Load precomputed SAM label maps and align them with training rays.

The masks come from `sam_masks` as one uint16
label PNG per frame per granularity level, with 0 meaning background.

Two things have to line up for the loss to mean anything:

* **Resolution.** Masks are stored at the scene's working resolution
  (images_4 outdoor, images_2 indoor) while training renders at whatever the
  downsample schedule currently says. Labels are resized with NEAREST -- any
  interpolation would invent label values that correspond to no mask.
* **Identity.** A mask id is only meaningful within its own view, so ids are
  packed as view_idx * MASK_STRIDE + local_id. Background becomes IGNORE_LABEL
  and is dropped by the loss rather than treated as an object.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from radfoam_model.instance_loss import IGNORE_LABEL, MASK_STRIDE

from radfoam_model.data_paths import SAM_MASK_ROOTS as DEFAULT_MASK_ROOTS
DEFAULT_ARM = "sam21_levels_image_t70"
DEFAULT_LEVELS = (0, 1, 2)


def resolve_mask_dir(scene, arm=DEFAULT_ARM, roots=None):
    """Return the arm directory for a scene, or None if masks are absent."""
    for root in roots if roots is not None else DEFAULT_MASK_ROOTS:
        candidate = Path(root) / scene / arm
        if (candidate / "frame_index.json").exists():
            return candidate
    return None


def load_level_labels(mask_dir, image_names, img_wh, levels=DEFAULT_LEVELS):
    """Return (n, h, w, len(levels)) float32 labels aligned with image_names.

    float32 rather than int: the batch fetchers this feeds are built for float
    tensors, and float32 represents integers exactly up to 2^24 -- far above
    the largest packed id (n_views * MASK_STRIDE).
    """
    mask_dir = Path(mask_dir)
    index = json.loads((mask_dir / "frame_index.json").read_text())
    name_to_frame = {name: i for i, name in enumerate(index["names"])}

    width, height = img_wh
    labels = np.full(
        (len(image_names), height, width, len(levels)),
        IGNORE_LABEL,
        dtype=np.float32,
    )

    missing = []
    for view_idx, name in enumerate(image_names):
        frame_idx = name_to_frame.get(name)
        if frame_idx is None:
            missing.append(name)
            continue
        for slot, level in enumerate(levels):
            path = mask_dir / f"labels_l{level}" / f"{frame_idx:06d}.png"
            if not path.exists():
                continue
            # NEAREST: label maps are categorical, so any smoothing would
            # fabricate ids that name no mask.
            raw = Image.open(path)
            if raw.size != (width, height):
                raw = raw.resize((width, height), Image.NEAREST)
            local = np.asarray(raw).astype(np.int64)
            packed = np.where(
                local == 0, IGNORE_LABEL, view_idx * MASK_STRIDE + local
            )
            labels[view_idx, :, :, slot] = packed.astype(np.float32)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training image(s) have no mask entry, e.g. "
            f"{missing[:3]}. Was this arm generated for a different scene?"
        )

    return torch.from_numpy(labels)


def label_coverage(labels):
    """Fraction of entries carrying a real mask, per level -- a sanity check."""
    valid = (labels >= 0).float()
    return valid.mean(dim=(0, 1, 2)) if labels.dim() == 4 else valid.mean()
