"""Ordered frame sequences and the symlink farm SAM's video predictors expect.

SAM 2 and SAM 3 both accept a directory of sequentially numbered JPEG frames.
mip-NeRF 360 images are named like DSC07956.JPG, and for these captures filename
order is capture order along the orbit, so it is the default. Pose ordering is
available as a diagnostic for whether ordering matters at all.

Symlinks, not copies: /shared is at quota and read-only, and the datasets are
24 GB.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"}


@dataclass
class FrameSequence:
    """An ordered set of frames plus the directory of numbered symlinks."""

    names: list          # original filenames, in frame order
    frames_dir: Path     # directory holding 000000.jpg, 000001.jpg, ...
    source_dir: Path
    order: str           # "filename" or "pose"

    def __len__(self):
        return len(self.names)

    def index_of(self, name):
        return self.names.index(name)


def _list_images(source_dir):
    images = [p for p in Path(source_dir).iterdir() if p.suffix in IMAGE_SUFFIXES]
    if not images:
        raise FileNotFoundError(f"No images found in {source_dir}")
    return sorted(images, key=lambda p: p.name)


def pose_order(names, centres):
    """Order names along a greedy nearest-neighbour path through camera centres.

    Starts at names[0] and repeatedly hops to the nearest unvisited centre. This
    is a diagnostic ordering: if tracking behaves differently under it than under
    filename order, the capture order is not the smoothest available path.
    """
    centres = np.asarray(centres, dtype=np.float64)
    if len(names) != len(centres):
        raise ValueError(
            f"names and centres must be the same length, got {len(names)} and {len(centres)}"
        )

    remaining = set(range(len(names)))
    current = 0
    remaining.discard(current)
    path = [current]

    while remaining:
        candidates = np.fromiter(remaining, dtype=int)
        distances = np.linalg.norm(centres[candidates] - centres[current], axis=1)
        current = int(candidates[int(np.argmin(distances))])
        remaining.discard(current)
        path.append(current)

    return [names[i] for i in path]


def build_sequence(source_dir, frames_dir, order="filename", centres=None):
    """Create the numbered symlink farm and write frame_index.json beside it.

    frame_index.json is written to frames_dir.parent so it sits next to, rather
    than inside, the directory handed to the model.
    """
    source_dir = Path(source_dir)
    frames_dir = Path(frames_dir)
    images = _list_images(source_dir)
    names = [p.name for p in images]

    if order == "pose":
        if centres is None:
            raise ValueError("order='pose' requires camera centres")
        names = pose_order(names, centres)
    elif order != "filename":
        raise ValueError(f"Unknown order {order!r}; expected 'filename' or 'pose'")

    frames_dir.mkdir(parents=True, exist_ok=True)
    # Clear first: a shrunk source would otherwise leave stale symlinks behind,
    # and the model would happily segment a frame that no longer exists.
    for stale in frames_dir.iterdir():
        stale.unlink()

    for i, name in enumerate(names):
        (frames_dir / f"{i:06d}.jpg").symlink_to(source_dir / name)

    index_path = frames_dir.parent / "frame_index.json"
    index_path.write_text(
        json.dumps(
            {
                "names": names,
                "source_dir": str(source_dir),
                "frames_dir": str(frames_dir),
                "order": order,
            },
            indent=2,
        )
    )

    return FrameSequence(
        names=names, frames_dir=frames_dir, source_dir=source_dir, order=order
    )


def load_frame_index(run_dir):
    """Load a previously written FrameSequence from run_dir/frame_index.json."""
    data = json.loads((Path(run_dir) / "frame_index.json").read_text())
    return FrameSequence(
        names=data["names"],
        frames_dir=Path(data["frames_dir"]),
        source_dir=Path(data["source_dir"]),
        order=data["order"],
    )
