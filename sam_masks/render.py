"""Render mask arms as orbit videos with view-stable colours.

The point is to see whether masks flicker. Colouring by object id would defeat
that for the per-image arms, which assign ids arbitrarily per frame -- every
frame would look like fresh noise whether the underlying segmentation was rock
stable or completely unstable.

So colour comes from geometry instead. Each mask is coloured by the mean 3D
position of the COLMAP sparse points that land inside it, mapped through the
scene's bounding box into RGB. A mask covering the same physical object gets the
same colour in every view regardless of what id it was given, so:

  * steady colour  = the segmentation is genuinely consistent across views
  * colour flicker = the same region is being grouped differently frame to frame
  * grey           = too few sparse points inside the mask to place it in 3D

This also makes the arms comparable to each other, since the colour of a region
is a property of the scene rather than of the run.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from sam_masks.colmap_tracks import read_tracks
from sam_masks.frames import load_frame_index
from sam_masks.paths import DOWNSAMPLE, resolve_output_root, scene_image_dir, scene_sparse_dir
from sam_masks.store import load_frame, read_meta

MIN_POINTS_FOR_COLOUR = 2
UNPLACED_COLOUR = np.array([110, 110, 110], dtype=np.uint8)


def read_points_xyz(sparse_dir):
    """Return {point3D_id: xyz} for the scene's sparse reconstruction."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    return {int(pid): np.asarray(p.xyz, dtype=np.float64) for pid, p in rec.points3D.items()}


def build_anchors(positions, n_anchors=96, iterations=25, seed=0):
    """Cluster sparse points into scene anchors, each with a distinct colour.

    A linear position-to-RGB map is view-stable but useless here: neighbouring
    objects land on nearly the same colour, so a merge or split between two
    adjacent regions is invisible -- exactly the failure this video is meant to
    reveal. Assigning each mask the colour of its nearest anchor keeps colours
    view-stable while making adjacent regions strongly distinguishable.

    Nearest-anchor is also robust in a way voxel hashing is not: a centroid that
    drifts slightly between frames keeps the same nearest anchor, so the video
    does not invent flicker that the segmentation did not produce.
    """
    rng = np.random.default_rng(seed)
    sample = positions[rng.choice(len(positions), size=min(len(positions), 20000),
                                  replace=False)]
    centres = sample[rng.choice(len(sample), size=n_anchors, replace=False)]

    for _ in range(iterations):
        distances = ((sample[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        assignment = distances.argmin(axis=1)
        for k in range(n_anchors):
            members = sample[assignment == k]
            if len(members):
                centres[k] = members.mean(axis=0)

    # Golden-ratio hue spacing gives maximally separated hues for any count.
    hues = (np.arange(n_anchors) * 0.61803398875) % 1.0
    hsv = np.stack(
        [hues * 179, np.full(n_anchors, 235.0), np.full(n_anchors, 245.0)], axis=1
    ).astype(np.uint8)[None]
    colours = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]
    return centres, colours


def anchor_colour(xyz, centres, colours):
    """Colour of the anchor nearest this 3D position."""
    return colours[int(((centres - xyz) ** 2).sum(axis=1).argmin())]


def frame_observations(tracks, name_to_frame, downsample, shape):
    """frame_idx -> list of (point_id, row, col) at working resolution."""
    height, width = shape
    by_frame = {}
    for point_id, elements in tracks.items():
        for name, xy in elements:
            frame_idx = name_to_frame.get(name)
            if frame_idx is None:
                continue
            col = int(xy[0] / downsample)
            row = int(xy[1] / downsample)
            if 0 <= row < height and 0 <= col < width:
                by_frame.setdefault(frame_idx, []).append((int(point_id), row, col))
    return by_frame


def colour_masks(frame_masks, observations, xyz_by_point, centres, palette):
    """Return an (N, 3) uint8 colour per mask, from its 3D centroid."""
    colours = np.tile(UNPLACED_COLOUR, (frame_masks.masks.shape[0], 1))
    if not observations or frame_masks.masks.shape[0] == 0:
        return colours

    rows = np.array([r for _, r, _ in observations])
    cols = np.array([c for _, _, c in observations])
    ids = [pid for pid, _, _ in observations]

    for i, mask in enumerate(frame_masks.masks):
        inside = mask[rows, cols]
        if inside.sum() < MIN_POINTS_FOR_COLOUR:
            continue
        positions = [xyz_by_point[ids[j]] for j in np.flatnonzero(inside)
                     if ids[j] in xyz_by_point]
        if len(positions) < MIN_POINTS_FOR_COLOUR:
            continue
        colours[i] = anchor_colour(np.mean(positions, axis=0), centres, palette)
    return colours


def composite(image, frame_masks, colours, alpha=0.65):
    """Alpha-blend coloured masks over the source image, smallest mask on top."""
    canvas = image.copy()
    if frame_masks.masks.shape[0] == 0:
        return canvas
    areas = frame_masks.masks.reshape(frame_masks.masks.shape[0], -1).sum(axis=1)
    for i in np.argsort(-areas):
        mask = frame_masks.masks[i]
        canvas[mask] = (
            alpha * colours[i] + (1 - alpha) * canvas[mask]
        ).astype(np.uint8)
    return canvas


def label_bar(width, text, height=34):
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (240, 240, 240), 1, cv2.LINE_AA)
    return bar


def render_arm(arm_path, scene, out_path, fps=12, scale=0.5, limit=None):
    """Write an orbit video for one arm. Returns the number of frames written."""
    arm_path = Path(arm_path)
    meta = read_meta(arm_path)
    sequence = load_frame_index(arm_path)
    source = scene_image_dir(scene)
    names = sequence.names[: limit or meta["n_frames"]]

    first = np.array(Image.open(arm_path / "labels" / "000000.png"))
    shape = first.shape

    tracks = read_tracks(scene_sparse_dir(scene))
    xyz_by_point = read_points_xyz(scene_sparse_dir(scene))
    name_to_frame = {name: i for i, name in enumerate(sequence.names)}
    observations = frame_observations(tracks, name_to_frame, DOWNSAMPLE[scene], shape)

    positions = np.stack(list(xyz_by_point.values()))
    # Clip far outliers before clustering: COLMAP leaves a few points at huge
    # distance which would otherwise capture anchors that nothing ever uses.
    lo = np.percentile(positions, 1, axis=0)
    hi = np.percentile(positions, 99, axis=0)
    inliers = positions[np.all((positions >= lo) & (positions <= hi), axis=1)]
    centres, palette = build_anchors(inliers)

    out_w = int(shape[1] * scale) // 2 * 2
    out_h = int(shape[0] * scale) // 2 * 2
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h + 34)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {out_path}")

    written = 0
    try:
        for frame_idx, name in enumerate(tqdm(names, desc=out_path.stem)):
            if not (arm_path / "masks" / f"{frame_idx:06d}.npz").exists():
                continue
            image = np.array(Image.open(source / name).convert("RGB"))
            frame_masks = load_frame(arm_path, frame_idx)
            colours = colour_masks(
                frame_masks, observations.get(frame_idx, []), xyz_by_point,
                centres, palette
            )
            canvas = composite(image, frame_masks, colours)
            canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
            caption = (
                f"{arm_path.name}   frame {frame_idx:3d}/{len(names)}   "
                f"{frame_masks.masks.shape[0]:3d} masks"
            )
            stacked = np.vstack([label_bar(out_w, caption), canvas])
            writer.write(cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR))
            written += 1
    finally:
        writer.release()
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Render mask arms as orbit videos with view-stable colours."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--arms", nargs="+", required=True,
                        help="Arm directory names, e.g. sam1_image_g32")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.output_root) if args.output_root else resolve_output_root()
    out_dir = Path(args.out_dir) if args.out_dir else root / args.scene / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arms:
        arm_path = root / args.scene / arm
        if not (arm_path / "meta.json").exists():
            print(f"skipping {arm}: no meta.json")
            continue
        out_path = out_dir / f"{args.scene}_{arm}.mp4"
        n = render_arm(arm_path, args.scene, out_path,
                       fps=args.fps, scale=args.scale, limit=args.limit)
        print(f"{out_path}  ({n} frames)")


if __name__ == "__main__":
    main()
