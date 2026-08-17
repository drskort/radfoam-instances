"""Remove one instance from a trained scene and re-render, by enclosure.

Implements the decode-time removal described in `docs/scene_editing_handoff.md`.
Nothing is retrained and the triangulation is never rebuilt: removal is done by
zeroing the density of the object's cells, which is what makes it safe to run at
inference (`update_triangulation` calls `permute_points`, which touches the
optimizer and therefore only works during training).

The object is taken as a *region*, not as a surface:

    core     = cells labelled c whose density clears the solid threshold
    shell    = core plus its full 1-ring -- all neighbours, not only those
               sharing the label, because cells just outside the surface are
               barely observed and their labels are unreliable; filtering by
               identity punches holes in the barrier the flood fill depends on
    outside  = flood fill from the bounding-box boundary, blocked by shell
    interior = everything in the box that the fill could not reach
    volume   = shell u interior

`interior` is watertight by construction and is derived without consulting the
features or density of the cells inside it -- both are meaningless there, since
the reconstruction stores objects as opaque shells over vacuum (15.3% of cells
sit at density 0.000 at every percentile). That is also why the "after" panel
shows vacuum rather than background: nothing behind the object was ever seen.

The flood fill runs on the sub-graph inside the object's padded bounding box
rather than on all ~4M cells. An interior cell of c is inside that box by
definition, so the restriction is exact, and it keeps the component labelling
to tens of thousands of nodes.

    sbatch -p 3090-lo --mem=48G scripts/render_removal_slurm.sh \\
        output/teatime_var05_geo
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
    fit_clusters_full,
    load_cached_clustering,
)
from radfoam_model.instance_graph import undirected_edges  # noqa: E402

SH_C0 = 0.28209479177387814
HIGHLIGHT = (1.0, 0.18, 0.18)


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


def one_ring(mask, edges):
    """mask plus every cell sharing a Delaunay face with it."""
    u, v = edges[:, 0], edges[:, 1]
    grown = mask.clone()
    grown[v[mask[u]]] = True
    grown[u[mask[v]]] = True
    return grown


def enclose(core, shell, points, edges, pad=0.12):
    """Cells the object encloses: everything in its box the outside cannot reach.

    Returns (volume, interior, leaked) where `leaked` is True if a core cell was
    reachable from the box boundary -- i.e. the shell has a hole and the fill is
    not to be trusted.
    """
    lo = points[core].min(dim=0).values
    hi = points[core].max(dim=0).values
    span = (hi - lo).clamp(min=1e-6)
    lo, hi = lo - pad * span, hi + pad * span

    in_box = ((points >= lo) & (points <= hi)).all(dim=1)
    free = in_box & ~shell                      # nodes the fill may travel through

    idx = torch.nonzero(free, as_tuple=True)[0]
    if idx.numel() == 0:
        return shell.clone(), torch.zeros_like(shell), False

    remap = torch.full((points.shape[0],), -1, dtype=torch.long,
                       device=points.device)
    remap[idx] = torch.arange(idx.numel(), device=points.device)

    keep = free[edges[:, 0]] & free[edges[:, 1]]
    sub = remap[edges[keep]].cpu().numpy()

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = idx.numel()
    graph = coo_matrix(
        (np.ones(sub.shape[0], dtype=np.uint8), (sub[:, 0], sub[:, 1])),
        shape=(n, n),
    )
    _, comp = connected_components(graph, directed=False)
    comp = torch.from_numpy(comp.astype(np.int64)).to(points.device)

    # Seeds: free cells sitting on the padded box boundary are outside by
    # construction -- the box was padded past the object, so its faces are in
    # open space.
    margin = 0.02 * span
    p = points[idx]
    on_face = ((p <= lo + margin) | (p >= hi - margin)).any(dim=1)
    outside_comps = torch.unique(comp[on_face])

    outside = torch.zeros_like(free)
    outside[idx] = torch.isin(comp, outside_comps)

    interior = in_box & ~outside & ~shell
    leaked = bool((core & outside).any())
    return shell | interior, interior, leaked


def render(model, rays, height, width, device):
    with torch.no_grad():
        points, _, _, _ = model.get_trace_data()
        start = model.get_starting_point(rays, points, model.aabb_tree)
        rgba, *_ = model(rays, start)
        rgb = (rgba[..., :3].clamp(0, 1) * 255).to(torch.uint8)
        return rgb.reshape(height, width, 3).cpu().numpy()


def label_bar(width, text, height=30):
    bar = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.putText(bar, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (235, 235, 235), 1, cv2.LINE_AA)
    return bar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--downsample", type=int, default=None)
    parser.add_argument("--top", type=int, default=4,
                        help="Remove each of the N largest instances in turn.")
    parser.add_argument("--instance-ids", default=None,
                        help="Comma-separated instance ids, overrides --top.")
    parser.add_argument("--frames", default="0,40,80")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, cfg, dataset_args = load_model(args.checkpoint, device, args.model)
    out_dir = Path(args.out_dir or Path(args.checkpoint) / "edit")
    out_dir.mkdir(parents=True, exist_ok=True)

    features = model.att_feat.detach()
    _, labels = load_cached_clustering(args.checkpoint, features)
    if labels is None:
        print("no cached per-cell labels; fitting HDBSCAN over every cell")
        labels, _ = fit_clusters_full(features)
    labels = labels.to(device)

    points = model.primal_points.detach()
    density = model.get_primal_density().detach().reshape(-1)
    edges = undirected_edges(model.point_adjacency, model.point_adjacency_offsets)

    observed = density > 1e-3
    sigma_solid = float(density[observed].median())
    print(f"{int(observed.sum())} observed cells of {density.numel()}; "
          f"sigma_solid = {sigma_solid:.3f}")

    if args.instance_ids:
        targets = [int(t) for t in args.instance_ids.split(",")]
    else:
        valid = labels[labels >= 0]
        ids, counts = torch.unique(valid, return_counts=True)
        targets = ids[counts.argsort(descending=True)][: args.top].tolist()
    print(f"targets: {targets}")

    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    downsample = args.downsample or min(dataset_args.downsample)
    data.reload(split=args.split, downsample=downsample)
    width, height = data.img_wh
    frames = [int(f) for f in args.frames.split(",")
              if int(f) < data.rays.shape[0]]

    density_backup = model.density.data.clone()
    dc_backup = model.att_dc.data.clone()
    sh_backup = model.att_sh.data.clone()
    report = []

    for target in targets:
        core = (labels == target) & (density > sigma_solid)
        if int(core.sum()) < 50:
            print(f"instance {target}: only {int(core.sum())} solid cells, skipping")
            continue

        shell = one_ring(core, edges)
        volume, interior, leaked = enclose(core, shell, points, edges)
        stats = {
            "instance": target,
            "core_cells": int(core.sum()),
            "shell_cells": int(shell.sum()),
            "interior_cells": int(interior.sum()),
            "volume_cells": int(volume.sum()),
            "leaked": leaked,
        }
        print(stats)
        report.append(stats)

        for frame in frames:
            rays = data.rays[frame].to(device).reshape(-1, 6)

            model.density.data.copy_(density_backup)
            model.att_dc.data.copy_(dc_backup)
            model.att_sh.data.copy_(sh_backup)
            before = render(model, rays, height, width, device)

            # Panel 2: the region that is about to go, painted flat red.
            for c in range(3):
                model.att_dc.data[volume, c] = (HIGHLIGHT[c] - 0.5) / SH_C0
            model.att_sh.data[volume] = 0.0
            marked = render(model, rays, height, width, device)

            # Panel 3: the region removed. Pre-activation -20 puts
            # softplus(beta=10) at ~1e-87, i.e. exactly transparent.
            model.att_dc.data.copy_(dc_backup)
            model.att_sh.data.copy_(sh_backup)
            model.density.data[volume] = -20.0
            after = render(model, rays, height, width, device)

            canvas = np.hstack([before, marked, after])
            caption = (f"{Path(args.checkpoint).name}  instance {target}  "
                       f"frame {frame}   |  original  |  enclosed volume "
                       f"({stats['volume_cells']} cells, {stats['interior_cells']} "
                       f"interior)  |  removed")
            canvas = np.vstack([label_bar(canvas.shape[1], caption), canvas])
            path = out_dir / f"remove_{target:03d}_frame{frame:04d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            print(f"wrote {path}")

    model.density.data.copy_(density_backup)
    model.att_dc.data.copy_(dc_backup)
    model.att_sh.data.copy_(sh_backup)
    (out_dir / "removal.json").write_text(json.dumps(
        {"sigma_solid": sigma_solid, "instances": report}, indent=2))


if __name__ == "__main__":
    main()
