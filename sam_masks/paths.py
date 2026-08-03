"""Filesystem layout for the SAM mask precomputation runs.

/work/user is a node-local ext4 disk on the login node host; the same
disk is visible from compute nodes as /nodes/host/work/user. Callers
should never hardcode either one.

/shared/user holds the datasets but is at its 500 GB quota, so it is
strictly an input path.
"""

from pathlib import Path

DATASET_ROOT = Path("/shared/user/datasets")

OUTPUT_ROOT_CANDIDATES = [
    Path("/work/user/sam_masks"),
    Path("/nodes/host/work/user/sam_masks"),
]

# Match radfoam's training resolutions so masks align with what the model will
# consume: configs/mipnerf360_outdoor.yaml settles at downsample 4,
# configs/mipnerf360_indoor.yaml at downsample 2.
DOWNSAMPLE = {
    "bicycle": 4,
    "flowers": 4,
    "garden": 4,
    "stump": 4,
    "treehill": 4,
    "bonsai": 2,
    "counter": 2,
    "kitchen": 2,
    "room": 2,
}


def resolve_output_root(candidates=None):
    """Return the first candidate output root whose parent directory exists.

    The root itself is created if missing; its parent must already exist, which
    is what distinguishes "we are on host" from "we are on a compute node".
    """
    candidates = list(candidates if candidates is not None else OUTPUT_ROOT_CANDIDATES)
    for candidate in candidates:
        if candidate.exists() or candidate.parent.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No writable output root among: {tried}")


def scene_image_dir(scene, root=DATASET_ROOT):
    """Return the image directory for a scene at its working resolution."""
    if scene not in DOWNSAMPLE:
        raise KeyError(f"Unknown scene {scene!r}; known: {sorted(DOWNSAMPLE)}")
    return Path(root) / scene / f"images_{DOWNSAMPLE[scene]}"


def scene_sparse_dir(scene, root=DATASET_ROOT):
    """Return the COLMAP sparse reconstruction directory for a scene."""
    return Path(root) / scene / "sparse" / "0"


def arm_dir(output_root, scene, model, mode):
    """Return the output directory for one experiment arm.

    model is "sam31" or "sam21"; mode is "video" or "image".
    """
    return Path(output_root) / scene / f"{model}_{mode}"
