"""Render RGB, instance-feature PCA and HDBSCAN clusters for a trained scene.

Everything that determines colour is computed ONCE, from the per-primitive
features of the 3D field, and then reused for every frame:

  * the PCA basis and its normalisation range,
  * the HDBSCAN clustering and its palette.

That is what makes the video readable. Fitting PCA per frame, or clustering the
rendered pixels of each view separately, would recolour the scene every frame
and show nothing about whether the embedding is actually consistent.

Clustering the 3D field rather than the 2D renders is also what OpenSplat3D
does -- instances live in the field, and the 2D views are projections of them.

    python scripts/render_instances.py \
            --checkpoint output/garden_inst_nogeo
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import *  # noqa: F401,F403  (ParamGroup definitions)
from data_loader import DataHandler
from radfoam_model.scene import RadFoamScene

from radfoam_model.instance_cluster import (  # noqa: E402
    NOISE_ID,
    assign,
    fit_clusters,
    to_pca_rgb,
)

NOISE_COLOUR = np.array([90, 90, 90], dtype=np.uint8)


def load_model(checkpoint, device, model_file="model.pt"):
    import configargparse

    config = Path(checkpoint) / "config.yaml"
    parser = configargparse.ArgParser(default_config_files=[str(config)])
    parser.add_argument("-c", "--config", is_config_file=True)
    model_params = ModelParams(parser)  # noqa: F405
    PipelineParams(parser)  # noqa: F405
    OptimizationParams(parser)  # noqa: F405
    dataset_params = DatasetParams(parser)  # noqa: F405
    args = parser.parse_args(["-c", str(config)])

    model = RadFoamScene(args=model_params.extract(args), device=device)
    model.load_pt(str(Path(checkpoint) / model_file))
    return model, args, dataset_params.extract(args)


def feature_to_cluster_rgb(feature_map, clustering):
    ids = assign(feature_map, clustering).cpu().numpy()
    rgb = np.full((*ids.shape, 3), NOISE_COLOUR, dtype=np.uint8)
    valid = ids != NOISE_ID
    if clustering.n_clusters:
        rgb[valid] = clustering.colours[ids[valid]]
    return rgb


def label_bar(width, text, height=30):
    bar = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Output scale per panel; 1.0 keeps render size.")
    parser.add_argument("--downsample", type=int, default=None,
                        help="Image downsample. Lower is higher resolution; "
                             "memory grows ~4x per step since every frame's "
                             "rays are held on the GPU at once.")
    parser.add_argument("--model", default="model.pt",
                        help="Checkpoint file inside --checkpoint. Point at a "
                             "numbered snapshot to avoid racing a live run.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, cfg, dataset_args = load_model(args.checkpoint, device, args.model)
    out_dir = Path(args.out_dir or Path(args.checkpoint) / "instances")
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_dim = getattr(model, "feat_dim", 0)
    has_features = feat_dim > 0 and hasattr(model, "att_feat")
    print(f"{args.checkpoint}: feat_dim={feat_dim}")

    clustering = fit_clusters(model.att_feat) if has_features else None
    if has_features:
        print(f"{clustering.n_clusters} instances "
              f"({100 * clustering.noise_fraction:.1f}% noise)")

    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    downsample = args.downsample or min(dataset_args.downsample)
    data.reload(split=args.split, downsample=downsample)
    width, height = data.img_wh
    n_frames = data.rays.shape[0] if args.limit is None else min(
        args.limit, data.rays.shape[0]
    )

    panels = 3 if has_features else 1
    out_w = int(width * args.scale) // 2 * 2
    out_h = int(height * args.scale) // 2 * 2
    writer = cv2.VideoWriter(
        str(out_dir / f"instances_{Path(args.model).stem}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (out_w * panels, out_h + 30),
    )

    with torch.no_grad():
        for frame in range(n_frames):
            rays = data.rays[frame].to(device).reshape(-1, 6)
            points, _, _, _ = model.get_trace_data()
            start = model.get_starting_point(rays, points, model.aabb_tree)
            rgba, feature, *_ = model(rays, start)

            rgb = (rgba[..., :3].clamp(0, 1) * 255).to(torch.uint8)
            rgb = rgb.reshape(height, width, 3).cpu().numpy()
            tiles = [rgb]

            if has_features:
                fmap = feature.reshape(height, width, feat_dim).float()
                tiles.append(to_pca_rgb(fmap, clustering))
                tiles.append(feature_to_cluster_rgb(fmap, clustering))

            tiles = [cv2.resize(t, (out_w, out_h), interpolation=cv2.INTER_AREA)
                     for t in tiles]
            canvas = np.hstack(tiles)
            caption = (f"{Path(args.checkpoint).name}   frame {frame}/{n_frames}"
                       + ("   |  RGB  |  feature PCA  |  HDBSCAN clusters"
                          if has_features else "   |  RGB (no features)"))
            writer.write(cv2.cvtColor(
                np.vstack([label_bar(out_w * panels, caption), canvas]),
                cv2.COLOR_RGB2BGR))

            if frame % 25 == 0:
                cv2.imwrite(str(out_dir / f"frame_{frame:04d}.png"),
                            cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print(f"\rframe {frame + 1}/{n_frames}", end="", flush=True)

    writer.release()
    video = out_dir / f"instances_{Path(args.model).stem}.mp4"
    print(f"\nwrote {video}  ({out_w * panels}x{out_h + 30}, {n_frames} frames)")
    if has_features:
        (out_dir / "clusters.json").write_text(json.dumps({
            "n_clusters": clustering.n_clusters,
            "noise_fraction": clustering.noise_fraction,
        }, indent=2))


if __name__ == "__main__":
    main()
