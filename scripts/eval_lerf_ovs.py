"""Score a trained scene on LERF-OVS.

Different task from LERF-Mask, and worth being clear about how:

* prompts are overwhelmingly bare nouns (67 categories over 4 scenes, only 5
  compositional, none of them containment). LERF-Mask's two catastrophic
  failures -- "wavy noodles in bowl" and "cookies on a plate" -- appear here as
  the separate categories "wavy noodles", "bowl", "three cookies", "plate".
* there is NO detector. Instances are chosen by language-embedding similarity,
  which is how LangSplat and OpenSplat3D evaluate it. So this measures the
  instance embeddings, not the grounding.
* the annotated frames are ordinary TRAINING views, not a held-out split. That
  is the published protocol, but it is not a held-out measurement and should
  not be reported as one.

Annotations are polygons in LangSplat's JSON format and are rasterised here.

    MODEL=model_020000.pt sbatch scripts/eval_lerf_ovs_slurm.sh output/ramen_var05_geo
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import *  # noqa: F401,F403
from data_loader import DataHandler
from radfoam_model.instance_cluster import NOISE_ID, load_cached_clustering

from eval_lerf_grounded import render_argmax_labels  # noqa: E402
from eval_lerf_mask import boundary_iou, iou  # noqa: E402
from radfoam_model.instance_language import (  # noqa: E402
    DYNAMIC_RATIO,
    MIN_PIXELS,
    RATIO,
    SMALL_AREA,
    LanguageEncoder,
    multi_level_boxes,
    rank_views,
    select_instances,
    square_pad_resize,
    surface_cells,
)
from extract_instance_language import load_model  # noqa: E402

LABEL_ROOT = Path("/nodes/host/work/user/lerf_ovs/lerf_ovs/label")


def load_polygons(scene, img_wh):
    """{frame_name: {category: bool mask}} rasterised from LangSplat polygons."""
    width, height = img_wh
    truth = {}
    for jf in sorted((LABEL_ROOT / scene).glob("*.json")):
        blob = json.loads(jf.read_text())
        name = blob["info"]["name"]
        sw, sh = blob["info"]["width"], blob["info"]["height"]
        per_cat = {}
        for obj in blob["objects"]:
            poly = np.asarray(obj["segmentation"], dtype=np.float64)
            if poly.ndim != 2 or poly.shape[0] < 3:
                continue
            # Annotations were drawn at the source resolution; scale if the
            # training images were loaded at a different downsample.
            poly[:, 0] *= width / sw
            poly[:, 1] *= height / sh
            m = np.zeros((height, width), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            cat = obj["category"]
            # A category may be several disjoint polygons in one frame.
            per_cat[cat] = (per_cat.get(cat, 0) | m).astype(np.uint8)
        truth[name] = {c: m.astype(bool) for c, m in per_cat.items()}
    return truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--encoder", default="masqclip",
                        choices=["masqclip", "siglip"],
                        help="masqclip is OpenSplat3D's default and sees each "
                             "crop together with the instance mask inside it; "
                             "siglip-so400m sees the crop alone.")
    parser.add_argument("--pred-threshold", type=float, default=0.85,
                        help="Instances scoring within this fraction of the "
                             "best answer the query too. Their pred_type=max.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--clustering", default="cached",
                        choices=["cached", "multicut", "multicut_logodds",
                                 "multicut_sam"],
                        help="cached = the HDBSCAN-full partition in "
                             "instances/clustering.pt. multicut = GAEC with "
                             "the tau surrogate. multicut_logodds = calibrated "
                             "without labels. multicut_sam = calibrated "
                             "against SAM mask co-occurrence.")
    parser.add_argument("--sam-prior", action="store_true",
                        help="Add the marginal prior log-odds to "
                             "every edge. Off by default: see "
                             "multicut_sam.")
    parser.add_argument("--occupancy", action="store_true",
                        help="Add edge solidity as a third log-odds term. "
                             "multicut_sam only.")
    parser.add_argument("--density-gate", action="store_true",
                        help="Use occupancy as a GATE on multicut edges rather "
                             "than as an additive log-odds term. Delaunay "
                             "adjacency is not contact: an edge crossing "
                             "vacuum should carry no vote however similar the "
                             "two cells look. This is what stops GAEC "
                             "percolating across the scene through air.")
    parser.add_argument("--sam-views", type=int, default=1000,
                        help="Cap on views used for SAM "
                             "co-occurrence; the default takes all.")
    parser.add_argument("--tau", type=float, default=0.15)
    parser.add_argument("--min-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, dataset_args = load_model(args.checkpoint, device, args.model)
    scene = dataset_args.scene


    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    data.reload(split="train", downsample=min(dataset_args.downsample))

    if args.clustering.startswith("multicut"):
        from radfoam_model.instance_graph import (
            clustering_from_labels,
            fit_graph_clusters,
        )

        if args.clustering == "multicut_sam":
            from radfoam_model.instance_graph import (
                multicut_sam,
                sam_edge_counts,
                undirected_edges,
            )
            from radfoam_model.instance_masks import resolve_mask_dir

            mask_dir = resolve_mask_dir(scene)
            if mask_dir is None:
                raise SystemExit(f"no SAM masks for {scene}")
            edges = undirected_edges(model.point_adjacency,
                                     model.point_adjacency_offsets)
            step = max(1, len(data.image_names) // args.sam_views)
            agree, disagree = sam_edge_counts(
                model, data, edges, mask_dir, data.image_names[::step],
                device, report=True)
            labels, _, _ = multicut_sam(
                model.att_feat, edges, agree, disagree,
                min_size=args.min_size, metric="euclidean", report=True,
                use_prior=args.sam_prior,
                occupancy=(model.get_primal_density().detach().float()
                           .reshape(-1)[edges].min(dim=1).values.cpu().numpy()
                           if args.occupancy else None),
            )
        else:
            kwargs = (dict(tau=args.tau) if args.clustering == "multicut"
                      else dict(report=True))
            result = fit_graph_clusters(
                model.att_feat, model.point_adjacency,
                model.point_adjacency_offsets, method=args.clustering,
                min_size=args.min_size, metric="euclidean",
                density=(model.get_primal_density()
                         if args.density_gate else None),
                **kwargs,
            )
            labels = result.labels
        clustering = clustering_from_labels(model.att_feat, labels)
        tag = (f"tau={args.tau}" if args.clustering == "multicut"
               else "calibrated log-odds")
        print(f"{args.clustering} {tag}: {clustering.n_clusters} instances "
              f"({100 * clustering.noise_fraction:.1f}% unassigned)", flush=True)
    else:
        clustering, labels = load_cached_clustering(args.checkpoint,
                                                    model.att_feat)
        if clustering is None or labels is None:
            raise SystemExit("no cached clustering with per-cell labels; run "
                             "`foamviz.py cluster --method full` first")
        print(f"{clustering.n_clusters} instances", flush=True)
    labels = labels.to(device)

    truth = load_polygons(scene, data.img_wh)
    index = {n: i for i, n in enumerate(data.image_names)}
    views = [n for n in sorted(truth) if n in index]
    missing = [n for n in truth if n not in index]
    if missing:
        raise SystemExit(f"annotated frames absent from the split: {missing[:3]}")
    cats = sorted({c for v in truth.values() for c in v})
    print(f"{len(views)} annotated frames, {len(cats)} categories", flush=True)

    # Instance masks over a sample of training views, used both to rank views
    # and to cut the crops.
    sample = list(data.image_names)[::max(1, len(data.image_names) // 30)]
    sindex = {n: i for i, n in enumerate(data.image_names)}
    smaps = render_argmax_labels(model, data, labels, clustering.n_clusters,
                                 sample, sindex, device)
    cells_per_instance = torch.bincount(
        labels[labels >= 0], minlength=clustering.n_clusters).cpu().numpy()

    encoder = LanguageEncoder(args.encoder, device)
    h, w = data.img_wh[1], data.img_wh[0]

    # (area, view_idx, mask, visible_cells) per instance per view
    obs = {}
    for vi, view in enumerate(sample):
        idm = smaps[vi]
        rays = data.rays[sindex[view]].to(device).reshape(-1, 6)
        seen_cells = torch.unique(surface_cells(model, rays, device))
        hit = labels[seen_cells[seen_cells >= 0]]
        seen = torch.bincount(hit[hit >= 0],
                              minlength=clustering.n_clusters).cpu().numpy()
        for k in np.unique(idm):
            if k == NOISE_ID:
                continue
            m = idm == k
            area = int(m.sum())
            if area < MIN_PIXELS:
                continue
            obs.setdefault(int(k), []).append((area, vi, m, int(seen[k])))

    rgb_cache = {}
    crops, crop_masks = {}, {}
    for k, o in obs.items():
        for area, vi, m, vis in rank_views(o, cells_per_instance[k]):
            if vi not in rgb_cache:
                rgb_cache[vi] = torch.from_numpy((
                    data.rgbs[sindex[sample[vi]]].reshape(h, w, -1)[..., :3]
                    .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                ).permute(2, 0, 1).contiguous()
            rgb = rgb_cache[vi]
            mt = torch.from_numpy(m)
            ratio = (RATIO if (area / (h * w)) < SMALL_AREA or
                     not DYNAMIC_RATIO else 0.1)
            for (r0, r1, c0, c1) in multi_level_boxes(m, ratio=ratio):
                if r1 <= r0 or c1 <= c0:
                    continue
                crop, cm = square_pad_resize(
                    rgb[:, r0:r1, c0:c1], mt[r0:r1, c0:c1], encoder.img_size)
                crops.setdefault(k, []).append(crop)
                crop_masks.setdefault(k, []).append(cm)
    print(f"{len(crops)} instances, {sum(len(c) for c in crops.values())} "
          f"crops (top-5 views x 3 levels, {args.encoder})", flush=True)

    ids, embeddings = [], []
    for n, k in enumerate(sorted(crops)):
        feats = []
        for s in range(0, len(crops[k]), args.batch):
            feats.append(encoder.encode_crops(crops[k][s:s + args.batch],
                                              crop_masks[k][s:s + args.batch]))
        pooled = torch.cat(feats).mean(dim=0)
        embeddings.append(torch.nn.functional.normalize(pooled, dim=-1))
        ids.append(k)
        if n % 25 == 0:
            print(f"\r  embedded {n + 1}/{len(crops)}", end="", flush=True)
    print()
    if not ids:
        raise SystemExit("no instance embeddings")
    text = encoder.encode_text(cats)
    scores = (text @ torch.stack(embeddings).to(device).T).cpu().numpy()
    chosen_per_cat = select_instances(scores, args.pred_threshold)

    id_maps = {v: m for v, m in zip(
        views, render_argmax_labels(model, data, labels, clustering.n_clusters,
                                    views, index, device))}

    per_iou, per_biou, oracle = {}, {}, {}
    # Their headline mIoU is the mean over every (frame, category) pair, not
    # the mean of per-category means. With categories annotated in unequal
    # numbers of frames the two differ, so both are reported.
    all_ious = []
    per_query = {}
    for ci, cat in enumerate(cats):
        chosen = [ids[j] for j in chosen_per_cat[ci]]
        per_query[cat] = len(chosen)
        ious, bious, orc = [], [], []
        for v in views:
            if cat not in truth[v]:
                continue
            gt = truth[v][cat]
            pred = np.isin(id_maps[v], chosen) & (id_maps[v] != NOISE_ID)
            ious.append(iou(gt, pred)); bious.append(boundary_iou(gt, pred))
            best = 0.0
            for k in np.unique(id_maps[v]):
                if k != NOISE_ID:
                    best = max(best, iou(gt, id_maps[v] == k))
            orc.append(best)
        all_ious.extend(ious)
        if ious:
            per_iou[cat] = float(np.mean(ious))
            per_biou[cat] = float(np.mean(bious))
            oracle[cat] = float(np.mean(orc))

    print(f"\n{'category':<24}{'IoU':>8}{'BIoU':>8}{'oracle':>9}")
    for c in cats:
        if c in per_iou:
            print(f"{c:<24}{100*per_iou[c]:>8.2f}{100*per_biou[c]:>8.2f}"
                  f"{100*oracle[c]:>9.2f}")
    mi = float(np.mean(list(per_iou.values())))
    mb = float(np.mean(list(per_biou.values())))
    mo = float(np.mean(list(oracle.values())))
    flat = float(np.mean(all_ious))
    print(f"{'-'*24} {'-'*7} {'-'*7} {'-'*8}")
    print(f"{scene + ' Mean':<24}{100*mi:>8.2f}{100*mb:>8.2f}{100*mo:>9.2f}")
    print(f"{scene + ' flat mIoU':<24}{100*flat:>8.2f}   "
          f"(their aggregation, over {len(all_ious)} frame-category pairs)")
    print(f"instances per query: mean "
          f"{np.mean(list(per_query.values())):.2f}, "
          f"max {max(per_query.values())}")

    suffix = {"cached": "", "multicut_logodds": "_logodds",
              "multicut_sam": "_sam" + ("_occ" if args.occupancy else "")}.get(
        args.clustering, f"_multicut_tau{args.tau}")
    if args.density_gate:
        suffix += "_gated"
    out = (Path(args.checkpoint)
           / f"lerf_ovs_{Path(args.model).stem}_{args.encoder}{suffix}.json")
    out.write_text(json.dumps(
        {"scene": scene, "miou": mi, "mbiou": mb, "oracle_miou": mo,
         "per_class_iou": per_iou, "oracle": oracle,
         "n_frames": len(views), "n_categories": len(per_iou),
         "flat_miou": flat, "n_instances_per_query": per_query,
         "encoder": args.encoder, "clustering": args.clustering,
         "n_clusters": clustering.n_clusters,
         "pred_threshold": args.pred_threshold}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
