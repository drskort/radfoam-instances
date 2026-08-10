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
    MODEL as OS3D_MODEL,
    RATIO,
    SMALL_AREA,
    embed_instances,
    encode_text,
    multi_level_boxes,
    rank_views,
    surface_cells,
)
from extract_instance_language import (  # noqa: E402
    DEFAULT_VLM,
    collect_instance_views,
    encode_instances,
    load_model,
)

TOPK_NOTE = "top-5 views x 3 levels"
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
    parser.add_argument("--recipe", default="opensplat3d",
                        choices=["opensplat3d", "legacy"],
                        help="opensplat3d = 3 crop levels, visibility-weighted "
                             "view ranking, siglip-so400m. legacy = the "
                             "original single-crop SigLIP2 path.")
    parser.add_argument("--vlm", default=None)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, dataset_args = load_model(args.checkpoint, device, args.model)
    scene = dataset_args.scene

    clustering, labels = load_cached_clustering(args.checkpoint, model.att_feat)
    if clustering is None or labels is None:
        raise SystemExit("no cached clustering with per-cell labels; run "
                         "`foamviz.py cluster --method full` first")
    print(f"{clustering.n_clusters} instances", flush=True)

    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    data.reload(split="train", downsample=min(dataset_args.downsample))
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

    vlm_name = args.vlm or (OS3D_MODEL if args.recipe == "opensplat3d"
                            else DEFAULT_VLM)
    if args.recipe == "opensplat3d":
        from PIL import Image
        # (area, view_idx, mask, visible_cells) per instance per view
        obs = {}
        for vi, view in enumerate(sample):
            idm = smaps[vi]
            rays = data.rays[sindex[view]].to(device).reshape(-1, 6)
            hit = labels[torch.unique(surface_cells(model, rays, device))]
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

        h, w = data.img_wh[1], data.img_wh[0]
        rgb_cache = {}
        crops = {}
        for k, o in obs.items():
            for area, vi, m, vis in rank_views(o, cells_per_instance[k]):
                if vi not in rgb_cache:
                    rgb_cache[vi] = (
                        data.rgbs[sindex[sample[vi]]].reshape(h, w, -1)[..., :3]
                        .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                rgb = rgb_cache[vi]
                ratio = (RATIO if (area / (h * w)) < SMALL_AREA or
                         not DYNAMIC_RATIO else 0.1)
                for (r0, r1, c0, c1) in multi_level_boxes(m, ratio=ratio):
                    if r1 > r0 and c1 > c0:
                        crops.setdefault(k, []).append(
                            Image.fromarray(rgb[r0:r1, c0:c1]))
        print(f"{len(crops)} instances, "
              f"{sum(len(c) for c in crops.values())} crops "
              f"({TOPK_NOTE})", flush=True)
        embeddings, ids = embed_instances(crops, device, vlm_name)
        text = encode_text(cats, device, vlm_name)
    else:
        best_views = collect_instance_views(model, data, clustering, device)
        embeddings, ids, _ = encode_instances(best_views, data, vlm_name, device)
        text = encode_text(cats, device, vlm_name)
    if not len(ids):
        raise SystemExit("no instance embeddings")
    scores = (text @ embeddings.to(device).T).cpu().numpy()

    id_maps = {v: m for v, m in zip(
        views, render_argmax_labels(model, data, labels, clustering.n_clusters,
                                    views, index, device))}

    per_iou, per_biou, oracle = {}, {}, {}
    for ci, cat in enumerate(cats):
        chosen = [ids[j] for j in np.argsort(-scores[ci])[: args.top_k]]
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
    print(f"{'-'*24} {'-'*7} {'-'*7} {'-'*8}")
    print(f"{scene + ' Mean':<24}{100*mi:>8.2f}{100*mb:>8.2f}{100*mo:>9.2f}")

    out = (Path(args.checkpoint)
           / f"lerf_ovs_{Path(args.model).stem}_{args.recipe}.json")
    out.write_text(json.dumps(
        {"scene": scene, "miou": mi, "mbiou": mb, "oracle_miou": mo,
         "per_class_iou": per_iou, "oracle": oracle,
         "n_frames": len(views), "n_categories": len(per_iou),
         "recipe": args.recipe, "vlm": vlm_name}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
