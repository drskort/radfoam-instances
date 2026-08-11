import os

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import pycolmap


def get_cam_ray_dirs(camera):
    x = np.arange(camera.width, dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate(
        [ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1
    )
    ray_dirs = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)


class COLMAPDataset:
    """mip-NeRF 360 layout: <root>/images[_N] and <root>/sparse/0.

    The three hooks below are what other captures vary. ScanNet++ keeps its
    poses under dslr/colmap, its frames under dslr/resized_images, and ships a
    canonical split file -- but the ray construction, pose handling and
    distortion model are identical, so it subclasses rather than duplicates.
    """

    def colmap_path(self, datadir):
        return os.path.join(datadir, "sparse/0/")

    def images_path(self, datadir, downsample):
        if downsample == 1:
            return os.path.join(datadir, "images")
        return os.path.join(datadir, f"images_{downsample}")

    def split_names(self, datadir, names, split):
        """Return the names for `split`, or None to fall back to the defaults."""
        return None

    def load_image(self, path):
        """Open one frame at the working resolution.

        mip-NeRF 360 ships a pre-downsampled pyramid on disk, so the base class
        just opens what images_path picked. Captures that ship a single
        resolution resize here instead.
        """
        return Image.open(path)

    def __init__(self, datadir, split, downsample):
        assert downsample in [1, 2, 4, 8]

        self.root_dir = datadir
        self.colmap_dir = self.colmap_path(datadir)
        self.split = split
        self.downsample = downsample

        images_dir = self.images_path(datadir, downsample)
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory {images_dir} not found")

        self.reconstruction = pycolmap.Reconstruction()
        self.reconstruction.read(self.colmap_dir)

        if len(self.reconstruction.cameras) > 1:
            raise ValueError("Multiple cameras are not supported")

        names = sorted(im.name for im in self.reconstruction.images.values())
        indices = np.arange(len(names))

        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split}")

        override = self.split_names(datadir, names, split)
        if override is not None:
            names = override
        elif os.path.isdir(os.path.join(datadir, "images_train")):
            # LERF-Mask ships its own split: images_train/ is the training set
            # and the graded views are images/test_*.jpg. Falling back to the
            # every-8th rule here would train on the graded views and score the
            # model on frames it had already fit.
            #
            # Membership in images_train/ decides it, not the test_ prefix. For
            # the three Gaussian Grouping scenes those agree exactly. They do
            # not for waldo_kitchen, which has no benchmark split: its held-out
            # frames keep their original names, and the every-8th rule cannot
            # be used there either because it would drop one of the five
            # LERF-OVS annotated frames out of training.
            train_dir = os.path.join(datadir, "images_train")
            in_train = {
                os.path.basename(n) for n in os.listdir(train_dir)
            }
            names = [
                n for n in names
                if (os.path.basename(n) in in_train) == (split == "train")
            ]
        elif split == "train":
            names = list(np.array(names)[indices % 8 != 0])
        else:
            names = list(np.array(names)[indices % 8 == 0])

        names = list(str(name) for name in names)

        im = self.load_image(os.path.join(images_dir, names[0]))
        self.img_wh = im.size
        im.close()

        self.camera = list(self.reconstruction.cameras.values())[0]
        self.camera.rescale(self.img_wh[0], self.img_wh[1])

        self.fx = self.camera.focal_length_x
        self.fy = self.camera.focal_length_y

        cam_ray_dirs = get_cam_ray_dirs(self.camera)

        self.images = []
        for name in names:
            image = None
            for image_id in self.reconstruction.images:
                image = self.reconstruction.images[image_id]
                if image.name == name:
                    break

            if image is None:
                raise ValueError(
                    f"Image {name} not found in COLMAP reconstruction"
                )

            self.images.append(image)

        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        for image in tqdm(self.images):
            c2w = torch.tensor(
                image.cam_from_world.inverse().matrix(), dtype=torch.float32
            )
            self.poses.append(c2w)
            world_ray_dirs = torch.einsum(
                "ij,kj->ik",
                cam_ray_dirs,
                c2w[:, :3],
            )
            world_ray_origins = c2w[:, 3] + torch.zeros_like(cam_ray_dirs)
            world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
            world_rays = world_rays.reshape(self.img_wh[1], self.img_wh[0], 6)

            im = self.load_image(os.path.join(images_dir, image.name))
            im = im.convert("RGB")
            rgbs = torch.tensor(np.array(im), dtype=torch.float32) / 255.0
            im.close()

            self.all_rays.append(world_rays)
            self.all_rgbs.append(rgbs)

        self.poses = torch.stack(self.poses)
        self.all_rays = torch.stack(self.all_rays)
        self.all_rgbs = torch.stack(self.all_rgbs)

        self.points3D = []
        self.points3D_color = []
        for point in self.reconstruction.points3D.values():
            self.points3D.append(point.xyz)
            self.points3D_color.append(point.color)

        self.points3D = torch.tensor(
            np.array(self.points3D), dtype=torch.float32
        )
        self.points3D_color = torch.tensor(
            np.array(self.points3D_color), dtype=torch.float32
        )
        self.points3D_color = self.points3D_color / 255.0
