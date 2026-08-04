"""Filesystem layout for the SAM mask precomputation runs.

/work/user is a node-local ext4 disk on the login node host; the same
disk is reachable from anywhere as /nodes/host/work/user. Callers
should never hardcode either one, and must not assume /work means host's
disk -- every compute node has its own /work.

/shared/user holds the datasets but is at its 500 GB quota, so it is
strictly an input path.
"""

from pathlib import Path

DATASET_ROOT = Path("/shared/user/datasets")

# The /nodes path first, deliberately. /work is NODE-LOCAL: it exists on every
# compute node as that node's own scratch, so preferring it means the output
# location depends on which node the job landed on -- results scatter across
# several machines' local disks and only some are visible afterwards. The
# /nodes/host view resolves to the same physical disk from the login node and
# from every compute node, so it is the only unambiguous choice.
OUTPUT_ROOT_CANDIDATES = [
    Path("/nodes/host/work/user/sam_masks"),
    Path("/work/user/sam_masks"),
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

    model is "sam31", "sam21" or "sam21_levels"; mode is "video" or "image".
    """
    return Path(output_root) / scene / f"{model}_{mode}"
