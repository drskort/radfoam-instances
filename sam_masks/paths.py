"""Filesystem layout for the SAM mask precomputation runs.

Roots come from radfoam_model.data_paths, which reads environment variables and
falls back to <repo>/data/<name>; see the README's dataset section.

One caveat specific to mask OUTPUT, kept because it is easy to get wrong on a
cluster: a node-local scratch path such as /work exists separately on every
compute node, so writing there scatters results across whichever machines the
jobs landed on. Point RADFOAM_SAM_MASKS at a path that resolves to the same
physical disk from every node.
"""

from pathlib import Path

from radfoam_model.data_paths import (
    LERF_MASK_ROOT as LERF_ROOT,
    MIPNERF360_ROOT as DATASET_ROOT,
    SAM_MASK_ROOTS as OUTPUT_ROOT_CANDIDATES,
    SCANNETPP_ROOT,
)

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


def is_scannetpp(scene):
    """ScanNet++ scenes are hex ids, and there are 1006 of them.

    Enumerating them into DOWNSAMPLE/SCENE_LAYOUT by hand would be absurd, so
    membership is decided by the release layout being present on disk.
    """
    return (SCANNETPP_ROOT / scene / "dslr" / "resized_images").is_dir()


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
    if is_scannetpp(scene):
        return SCANNETPP_ROOT
    layout = SCENE_LAYOUT.get(scene)
    return layout[0] if layout else DATASET_ROOT


def scene_downsample(scene):
    """Working resolution for a scene, recorded in the arm's metadata.

    ScanNet++ is 1006 scenes keyed by hex id, so they cannot be enumerated in
    DOWNSAMPLE by hand; the release ships one resolution and the masks are
    resized to the training size on load.
    """
    if is_scannetpp(scene):
        return 1
    if scene not in DOWNSAMPLE:
        raise KeyError(f"Unknown scene {scene!r}; known: {sorted(DOWNSAMPLE)}")
    return DOWNSAMPLE[scene]


def scene_image_dir(scene, root=None):
    """Return the image directory for a scene at its working resolution."""
    if is_scannetpp(scene):
        # One resolution on disk, 1752x1168. Masks are resized to the training
        # size with NEAREST when loaded, so generating at native resolution
        # costs SAM time but loses no boundary detail.
        return scene_root(scene, root) / scene / "dslr" / "resized_images"
    if scene not in DOWNSAMPLE:
        raise KeyError(f"Unknown scene {scene!r}; known: {sorted(DOWNSAMPLE)}")
    layout = SCENE_LAYOUT.get(scene)
    images = layout[1] if layout else f"images_{DOWNSAMPLE[scene]}"
    return scene_root(scene, root) / scene / images


def scene_sparse_dir(scene, root=None):
    """Return the COLMAP sparse reconstruction directory for a scene."""
    if is_scannetpp(scene):
        return scene_root(scene, root) / scene / "dslr" / "colmap"
    return scene_root(scene, root) / scene / "sparse" / "0"


def arm_dir(output_root, scene, model, mode, tag=None):
    """Return the output directory for one experiment arm.

    model is "sam1", "sam21", "sam21_levels" or "sam31"; mode is "video" or
    "image". tag distinguishes runs of the same arm under different settings --
    a denser proposal grid, say -- so they do not overwrite each other.
    """
    name = f"{model}_{mode}" if tag is None else f"{model}_{mode}_{tag}"
    return Path(output_root) / scene / name
