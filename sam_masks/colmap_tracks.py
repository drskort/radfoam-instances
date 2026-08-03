"""COLMAP sparse-point observations, the correspondence source for the metrics.

There is no ground-truth segmentation for these scenes, so consistency has to be
measured geometrically. COLMAP's reconstruction gives, for each 3D point, the set
of images observing it and where. That is a free and reliable cross-view
correspondence set.

pycolmap has no dependency on radfoam's CUDA extension, so this works inside the
SAM venv.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Observations:
    """Sparse-point observations at working resolution.

    xy is a list of (frame_idx, point_id, row, col) tuples with integer pixel
    coordinates. by_frame and frames_of_point are derived indices.
    """

    xy: list = field(default_factory=list)
    by_frame: dict = field(default_factory=lambda: defaultdict(list))
    frames_of_point: dict = field(default_factory=lambda: defaultdict(list))


def read_tracks(sparse_dir):
    """Read a COLMAP reconstruction into {point3D_id: [(image_name, xy), ...]}.

    Coordinates are full-resolution pixel coordinates, as COLMAP stores them.
    """
    import pycolmap

    rec = pycolmap.Reconstruction(str(Path(sparse_dir)))
    image_names = {image_id: image.name for image_id, image in rec.images.items()}

    tracks = {}
    for point_id, point in rec.points3D.items():
        elements = []
        for element in point.track.elements:
            image = rec.images[element.image_id]
            xy = image.points2D[element.point2D_idx].xy
            elements.append((image_names[element.image_id], xy))
        if elements:
            tracks[int(point_id)] = elements
    return tracks


def observations_from_tracks(tracks, name_to_frame, downsample, shape):
    """Project COLMAP tracks onto frame indices at working resolution.

    Observations whose image is not part of the frame sequence, or which fall
    outside the working-resolution image bounds after scaling, are dropped.
    """
    if downsample <= 0:
        raise ValueError(f"downsample must be positive, got {downsample}")

    height, width = shape
    obs = Observations()

    for point_id, elements in tracks.items():
        for name, xy in elements:
            frame_idx = name_to_frame.get(name)
            if frame_idx is None:
                continue
            col = int(xy[0] / downsample)
            row = int(xy[1] / downsample)
            if not (0 <= row < height and 0 <= col < width):
                continue
            obs.xy.append((frame_idx, int(point_id), row, col))
            obs.by_frame[frame_idx].append(int(point_id))
            obs.frames_of_point[int(point_id)].append(frame_idx)

    return obs


def load_observations(sparse_dir, name_to_frame, downsample, shape):
    """Convenience wrapper: read a reconstruction and project it in one call."""
    return observations_from_tracks(
        read_tracks(sparse_dir), name_to_frame, downsample, shape
    )
