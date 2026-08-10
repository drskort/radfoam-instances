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

# LERF-Mask (Gaussian Grouping's annotated LERF scenes), downloaded rather than
# part of the mip-NeRF 360 set, so it lives on the writable disk.
LERF_ROOT = Path("/nodes/host/work/user/lerf_mask")

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
    # LERF-Mask ships a single ~986x728 resolution, so no downsampling.
    # waldo_kitchen is the fourth LERF-OVS scene and matches at 985x725.
    "figurines": 1,
    "ramen": 1,
    "teatime": 1,
    "waldo_kitchen": 1,
}

# Scenes that do not follow mip-NeRF 360's <root>/<scene>/images_<n> layout.
#
# For LERF-Mask the image directory is images_train, NOT images: the benchmark
# holds four views out of images/ as test_N.jpg and grades against test_mask/.
# Generating masks over images/ would put the graded views into supervision and
# quietly inflate every number the benchmark reports.
# waldo_kitchen has no benchmark split of its own, so its images_train is
# symlinks to every frame -- LERF-OVS evaluates on training views. See the
# README in that scene directory.
SCENE_LAYOUT = {
    scene: (LERF_ROOT, "images_train")
    for scene in ("figurines", "ramen", "teatime", "waldo_kitchen")
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


def scene_root(scene, root=None):
    """Dataset root for a scene; an explicit root always wins."""
    if root is not None:
        return Path(root)
    layout = SCENE_LAYOUT.get(scene)
    return layout[0] if layout else DATASET_ROOT


def scene_image_dir(scene, root=None):
    """Return the image directory for a scene at its working resolution."""
    if scene not in DOWNSAMPLE:
        raise KeyError(f"Unknown scene {scene!r}; known: {sorted(DOWNSAMPLE)}")
    layout = SCENE_LAYOUT.get(scene)
    images = layout[1] if layout else f"images_{DOWNSAMPLE[scene]}"
    return scene_root(scene, root) / scene / images


def scene_sparse_dir(scene, root=None):
    """Return the COLMAP sparse reconstruction directory for a scene."""
    return scene_root(scene, root) / scene / "sparse" / "0"


def arm_dir(output_root, scene, model, mode, tag=None):
    """Return the output directory for one experiment arm.

    model is "sam1", "sam21", "sam21_levels" or "sam31"; mode is "video" or
    "image". tag distinguishes runs of the same arm under different settings --
    a denser proposal grid, say -- so they do not overwrite each other.
    """
    name = f"{model}_{mode}" if tag is None else f"{model}_{mode}_{tag}"
    return Path(output_root) / scene / name
