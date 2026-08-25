"""Look at what multicut actually produces, next to HDBSCAN, on the same views.

Both are rendered through the argmax readout -- each pixel takes the cluster
contributing the most alpha along its ray -- so the two panels differ only in
how the clusters were formed, not in how they are turned into pixels.

    MODEL=model_020000.pt sbatch scripts/render_multicut_slurm.sh output/ramen_inst_geo
"""


import sys
from pathlib import Path

# Run directly from a clone: this lives in scripts/ but imports configs/,
# radfoam_model/ and data_loader/ from the repo root, which pip does not
# install (setup.cfg packages only src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import *  # noqa: F401,F403
from data_loader import DataHandler
from radfoam_model.instance_cluster import NOISE_ID, assign, fit_clusters
from radfoam_model.instance_graph import (
    _stable_palette,
    clustering_from_labels,
    fit_graph_clusters,
)

from eval_lerf_grounded import render_argmax_labels  # noqa: E402
from radfoam_model.checkpoint import load_model  # noqa: E402

NOISE_COLOUR = np.array([70, 70, 70], dtype=np.uint8)
ALPHA = 0.6


def colourise(id_map, palette):
    rgb = np.full((*id_map.shape, 3), NOISE_COLOUR, dtype=np.uint8)
    valid = id_map != NOISE_ID
    if palette.shape[0]:
        rgb[valid] = palette[id_map[valid] % palette.shape[0]]
    return rgb


def blend(rgb, overlay, alpha=ALPHA):
    return ((1 - alpha) * rgb + alpha * overlay).astype(np.uint8)


def caption(width, text, height=30):
    bar = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def stats(labels, n_clusters):
    if not n_clusters:
        return "no clusters"
    counts = torch.bincount(labels[labels >= 0], minlength=n_clusters).cpu().numpy()
    total = int(labels.shape[0])
    return (f"{n_clusters} clusters | largest {100 * counts.max() / total:.0f}% "
            f"of cells | median {int(np.median(counts))} cells")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--taus", type=float, nargs="*", default=[0.5, 1.0])
    parser.add_argument("--density", action="store_true",
                        help="Also weight edges by min cell density.")
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, _, dataset_args = load_model(args.checkpoint, device, args.model)
    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    data.reload(split=args.split, downsample=min(dataset_args.downsample))
    height, width = data.img_wh[1], data.img_wh[0]
    views = list(data.image_names)[: args.limit]
    index = {n: i for i, n in enumerate(data.image_names)}

    panels = {}

    hdb = fit_clusters(model.att_feat)
    hdb_labels = assign(model.att_feat.detach().float(), hdb)
    print(f"hdbscan: {stats(hdb_labels, hdb.n_clusters)}", flush=True)
    panels[f"HDBSCAN  {stats(hdb_labels, hdb.n_clusters)}"] = (
        render_argmax_labels(model, data, hdb_labels, hdb.n_clusters,
                             views, index, device),
        _stable_palette(hdb.n_clusters),
    )

    variants = [(tau, False) for tau in args.taus]
    if args.density:
        variants += [(tau, True) for tau in args.taus]
    for tau, use_density in variants:
        result = fit_graph_clusters(
            model.att_feat, model.point_adjacency, model.point_adjacency_offsets,
            method="multicut", tau=tau, metric="euclidean",
            min_size=args.min_size,
            density=model.get_primal_density() if use_density else None,
        )
        clustering = clustering_from_labels(model.att_feat, result.labels)
        info = stats(result.labels, clustering.n_clusters)
        tag = f"tau={tau}{' +density' if use_density else ''}"
        print(f"multicut {tag}: {info}", flush=True)
        panels[f"MULTICUT {tag}  {info}"] = (
            render_argmax_labels(model, data, result.labels,
                                 clustering.n_clusters, views, index, device),
            _stable_palette(clustering.n_clusters),
        )

    out_dir = Path(args.checkpoint) / "multicut"
    out_dir.mkdir(parents=True, exist_ok=True)
    for vi, view_name in enumerate(views):
        rgb = (data.rgbs[index[view_name]].reshape(height, width, -1)[..., :3]
               .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        columns = [np.vstack([caption(width, f"image  {view_name}"), rgb])]
        for title, (maps, palette) in panels.items():
            overlay = blend(rgb, colourise(maps[vi], palette))
            columns.append(np.vstack([caption(width, title), overlay]))
        cv2.imwrite(str(out_dir / f"{Path(view_name).stem}.png"),
                    cv2.cvtColor(np.hstack(columns), cv2.COLOR_RGB2BGR))
    print(f"wrote {len(views)} panels to {out_dir}")


if __name__ == "__main__":
    main()
