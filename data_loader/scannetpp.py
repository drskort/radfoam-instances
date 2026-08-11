"""ScanNet++ DSLR captures, in the layout the official release ships.

Room-scale indoor scans rather than the object-centric captures the other
loaders handle, and the benchmark that matters for us: OpenSplat3D scores 3D
instance segmentation on the 50-scene nvs_sem_val split, against points sampled
from the aligned mesh. That is a real 3D evaluation instead of a rendered mask
compared to a 2D polygon, which is where a space-tiling representation has
something to say -- a point falls inside exactly one Voronoi cell, whereas
"nearest Gaussian mean" is an approximation their pipeline then has to smooth.

Only the paths and the split differ from a mip-NeRF 360 capture. The camera is
OPENCV_FISHEYE, which needs no special handling here because ray directions go
through pycolmap's cam_from_img and it applies the distortion model itself.
"""

import json
import os

from PIL import Image

from .colmap import COLMAPDataset

ROOT = "/shared/scannetpp/data"
SPLITS = "/shared/scannetpp/splits"


class ScanNetPPDataset(COLMAPDataset):
    def colmap_path(self, datadir):
        return os.path.join(datadir, "dslr", "colmap")

    def images_path(self, datadir, downsample):
        # resized_images is the 1752x1168 release; there is no pyramid on disk,
        # so downsample is honoured by the caller's choice of working size
        # rather than by picking a different directory.
        return os.path.join(datadir, "dslr", "resized_images")

    def load_image(self, path):
        """Resize on load: the release ships one resolution, not a pyramid.

        Full 1752x1168 over ~390 frames is roughly 29 GB of rays and colours
        before DataHandler's rearrange copies them again. downsample 2 lands at
        876x584, which is also close to the working size the LERF scenes train
        at, so per-scene cost stays comparable.
        """
        image = Image.open(path)
        if self.downsample == 1:
            return image
        width, height = image.size
        return image.resize(
            (width // self.downsample, height // self.downsample),
            Image.LANCZOS,
        )

    def split_names(self, datadir, names, split):
        """The capture ships its own train/test lists -- use them.

        Falling back to the every-8th rule would hold out frames the benchmark
        expects to be trained on, and ScanNet++'s test frames are deliberately
        chosen to be a novel-view split rather than every eighth frame of the
        trajectory.
        """
        path = os.path.join(datadir, "dslr", "train_test_lists.json")
        if not os.path.exists(path):
            return None
        lists = json.loads(open(path).read())
        wanted = set(lists["train" if split == "train" else "test"])
        chosen = [n for n in names if os.path.basename(n) in wanted]
        if not chosen:
            raise ValueError(
                f"{path} lists no {split} frame present in the reconstruction"
            )
        return chosen


def val_scenes():
    """The 50 scenes OpenSplat3D reports on."""
    with open(os.path.join(SPLITS, "nvs_sem_val.txt")) as handle:
        return [line.strip() for line in handle if line.strip()]
