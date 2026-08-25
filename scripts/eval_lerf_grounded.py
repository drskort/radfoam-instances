"""OpenSplat3D's LERF-Mask protocol, reproduced against a Radiant Foam scene.

Their eval does not use language embeddings at all. A prompt is grounded in
*one* reference frame with GroundingDINO + SAM, every 3D instance whose
projection falls mostly inside that 2D mask is selected, and the union of those
instances is rendered into all graded views. Reproduced from
VisualComputingInstitute/opensplat3d, eval/eval_lerf_mask.py:

  * grounding on the FIRST test view only (`obj_masks[0]`),
  * GroundingDINO SwinB at box 0.3 / text 0.45, all surviving boxes fed to
    SAM ViT-H and unioned (`torch.sum(masks, dim=0).bool()`),
  * instance selection by IoA > 0.7, where the denominator is the *instance's*
    area, so an object split across several instances keeps all of them,
  * mIoU averaged per prompt-class first, then over classes,
  * boundary IoU at dilation_ratio 0.02.

One deviation, forced by the representation. They render each instance as a
soft silhouette and threshold at 0.2; the foam is a partition of space, so an
instance mask here is the set of pixels whose nearest centroid is that
instance. Their masks can overlap, ours cannot.

    MODEL=model_020000.pt sbatch scripts/eval_grounded_slurm.sh output/ramen_inst_geo
"""


import sys
from pathlib import Path

# Run directly from a clone: the eval scripts live in scripts/ but import
# configs/, radfoam_model/ and data_loader/ from the repo root, which pip
# does not install (setup.cfg packages only src/). The Slurm wrappers set
# PYTHONPATH; this makes the plain commands in the README work too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import *  # noqa: F401,F403
from data_loader import DataHandler
from radfoam_model.instance_cluster import NOISE_ID, fit_clusters
from radfoam_model.instance_graph import (
    clustering_from_labels,
    fit_graph_clusters,
)

from eval_lerf_mask import (  # noqa: E402
    boundary_iou,
    iou,
    load_ground_truth,
    predict_masks,
)
from radfoam_model.checkpoint import load_model  # noqa: E402

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.45
IOA_THRESHOLD = 0.7
DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_MODEL = "facebook/sam-vit-huge"


def ground_prompt(image, prompt, dino, dino_proc, sam, sam_proc, device):
    """GroundingDINO boxes -> SAM masks -> their union, as a bool array."""
    from PIL import Image as PILImage

    pil = PILImage.fromarray(image)
    # GroundingDINO wants a lowercase, period-terminated caption.
    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."

    inputs = dino_proc(images=pil, text=caption, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino(**inputs)
    detections = dino_proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[pil.size[::-1]],
    )[0]

    boxes = detections["boxes"].cpu().tolist()
    if not boxes:
        return np.zeros(image.shape[:2], dtype=bool), 0

    sam_inputs = sam_proc(pil, input_boxes=[boxes], return_tensors="pt").to(device)
    with torch.no_grad():
        sam_out = sam(**sam_inputs, multimask_output=False)
    masks = sam_proc.image_processor.post_process_masks(
        sam_out.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]
    # Union over boxes, exactly as the reference does.
    return masks.squeeze(1).any(dim=0).numpy().astype(bool), len(boxes)


def render_silhouettes(model, data, labels, cluster_ids, views, index, device,
                       threshold=0.2):
    """Alpha-composite a per-cluster indicator and threshold it.

    This is OpenSplat3D's instance mask, not a hard partition: they render each
    instance as a white silhouette and keep pixels above 0.2, so masks may
    overlap. Reproducing it matters for any clustering whose clusters are
    spatial rather than feature-space modes -- collapsing those to centroids and
    re-assigning by nearest centroid recovers only ~13% of the partition.

    The tracer already carries feat_dim channels, so feat_dim indicators ride
    along in a single pass instead of one pass per cluster.
    """
    feat_dim = model.att_feat.shape[1]
    height, width = data.img_wh[1], data.img_wh[0]
    original = model.att_feat
    out = np.zeros((len(views), len(cluster_ids), height, width), dtype=bool)
    try:
        for start_idx in range(0, len(cluster_ids), feat_dim):
            block = cluster_ids[start_idx:start_idx + feat_dim]
            indicator = torch.zeros(labels.shape[0], feat_dim,
                                    device=device, dtype=original.dtype)
            for j, k in enumerate(block):
                indicator[labels == k, j] = 1.0
            model.att_feat = torch.nn.Parameter(indicator, requires_grad=False)
            with torch.no_grad():
                points, _, _, _ = model.get_trace_data()
                for vi, view_name in enumerate(views):
                    rays = data.rays[index[view_name]].to(device).reshape(-1, 6)
                    begin = model.get_starting_point(rays, points, model.aabb_tree)
                    _, feature, *_ = model(rays, begin)
                    composited = feature.reshape(height, width, feat_dim).float()
                    for j in range(len(block)):
                        out[vi, start_idx + j] = (
                            composited[..., j] > threshold
                        ).cpu().numpy()
    finally:
        model.att_feat = original
    return out


def render_argmax_labels(model, data, labels, n_clusters, views, index, device,
                         min_weight=1e-3):
    """Give each pixel to whichever cluster contributes most alpha along its ray.

    The natural readout for a space partition, and the one both other paths get
    wrong. Nearest-centroid ignores where a cluster actually is; thresholding
    each silhouette at 0.2 assumes clusters are object-sized, so a fine
    decomposition starves -- every cluster contributes a little and none clears
    the bar. An argmax needs no threshold and stays a partition however fine the
    clustering is.

    Only a running best value and index are kept, so memory does not grow with
    the number of clusters.
    """
    feat_dim = model.att_feat.shape[1]
    height, width = data.img_wh[1], data.img_wh[0]
    original = model.att_feat
    best_value = torch.full((len(views), height, width), -1.0, device=device)
    best_id = torch.full((len(views), height, width), NOISE_ID,
                         dtype=torch.long, device=device)
    try:
        for start_idx in range(0, n_clusters, feat_dim):
            block = list(range(start_idx, min(start_idx + feat_dim, n_clusters)))
            indicator = torch.zeros(labels.shape[0], feat_dim,
                                    device=device, dtype=original.dtype)
            for j, k in enumerate(block):
                indicator[labels == k, j] = 1.0
            model.att_feat = torch.nn.Parameter(indicator, requires_grad=False)
            with torch.no_grad():
                points, _, _, _ = model.get_trace_data()
                for vi, view_name in enumerate(views):
                    rays = data.rays[index[view_name]].to(device).reshape(-1, 6)
                    begin = model.get_starting_point(rays, points, model.aabb_tree)
                    _, feature, *_ = model(rays, begin)
                    composited = feature.reshape(
                        height, width, feat_dim
                    ).float()[..., :len(block)]
                    value, arg = composited.max(dim=-1)
                    better = value > best_value[vi]
                    best_value[vi][better] = value[better]
                    best_id[vi][better] = arg[better] + start_idx
            print(f"  argmax pass {start_idx // feat_dim + 1}/"
                  f"{-(-n_clusters // feat_dim)}", flush=True)
    finally:
        model.att_feat = original
    # A pixel no cluster reaches -- empty space, or cells dropped as too small.
    best_id[best_value <= min_weight] = NOISE_ID
    return best_id.cpu().numpy()


def select_by_ioa_silhouette(silhouettes, grounded, threshold=IOA_THRESHOLD):
    """Instances whose own area lies mostly inside the grounded mask."""
    chosen = []
    for i in range(silhouettes.shape[0]):
        area = silhouettes[i].sum()
        if area and np.logical_and(silhouettes[i], grounded).sum() / area > threshold:
            chosen.append(i)
    return chosen


def render_features(model, data, views, index, device):
    """Raw composited feature map per graded view, plus opacity."""
    height, width = data.img_wh[1], data.img_wh[0]
    out = {}
    with torch.no_grad():
        points, _, _, _ = model.get_trace_data()
        for view_name in views:
            rays = data.rays[index[view_name]].to(device).reshape(-1, 6)
            start = model.get_starting_point(rays, points, model.aabb_tree)
            rgba, feature, *_ = model(rays, start)
            out[view_name] = feature.reshape(height, width, -1).float()
    return out


def fisher_decode(features, reference, grounded, views, n_steps=80):
    """Segment by projecting the rendered features onto a discriminant.

    No clustering at all. The grounded mask on the reference view labels which
    pixels are the object; a diagonal Fisher direction separating those from the
    rest is fitted there, the threshold is chosen on that same view, and both
    transfer unchanged to the graded views.

    Motivated by measurement: every LERF-Mask object is linearly separable in
    this feature space at AUC >= 0.988 -- including ones the instance decoder
    scores 0.00 on. The information survives rendering; clustering loses it.

    Nothing here touches ground truth. The reference view's mask comes from
    GroundingDINO + SAM exactly as in the instance path.
    """
    ref = features[reference]
    inside, outside = ref[grounded], ref[~grounded]
    if inside.shape[0] < 16 or outside.shape[0] < 16:
        return {v: np.zeros(features[v].shape[:2], dtype=bool) for v in views}

    spread = inside.var(0, unbiased=False) + outside.var(0, unbiased=False)
    direction = (inside.mean(0) - outside.mean(0)) / spread.clamp(min=1e-6)

    projected = ref @ direction
    gt_ref = torch.from_numpy(grounded).to(projected.device)
    # Threshold fitted against the grounded mask, on the reference view only.
    lo, hi = torch.quantile(projected.flatten().float(),
                            torch.tensor([0.50, 0.9995], device=projected.device))
    best_t, best_iou = lo, -1.0
    for t in torch.linspace(lo.item(), hi.item(), n_steps, device=projected.device):
        pred = projected > t
        inter = (pred & gt_ref).sum()
        union = (pred | gt_ref).sum()
        s = (inter / union).item() if union > 0 else 0.0
        if s > best_iou:
            best_iou, best_t = s, t
    return {v: ((features[v] @ direction) > best_t).cpu().numpy() for v in views}


def instance_ioa(id_map, grounded):
    """IoA of every instance against the grounded mask, computed once.

    IoA divides by the *instance's* area, so the whole threshold sweep is a
    comparison against this vector -- the rendering and the grounding, which
    are what actually cost anything, are shared across every threshold.
    """
    out = {}
    for k in np.unique(id_map):
        if k == NOISE_ID:
            continue
        inside = id_map == k
        area = inside.sum()
        if area:
            out[int(k)] = float(np.logical_and(inside, grounded).sum() / area)
    return out


def select_by_ioa(id_map, grounded, threshold=IOA_THRESHOLD):
    """Instances whose own area lies mostly inside the grounded mask."""
    chosen = []
    for k in np.unique(id_map):
        if k == NOISE_ID:
            continue
        instance = id_map == k
        area = instance.sum()
        if area and np.logical_and(instance, grounded).sum() / area > threshold:
            chosen.append(int(k))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--ioa-threshold", type=float, default=IOA_THRESHOLD)
    parser.add_argument("--clustering", default="hdbscan",
                        choices=["hdbscan", "hdbscan_full", "hdbscan_aug",
                                 "multicut", "felzenszwalb", "threshold"])
    parser.add_argument("--tau", type=float, default=0.15,
                        help="Multicut/threshold weight offset. Swept on all "
                             "three LERF scenes: the per-scene optima are "
                             "0.30/0.25/0.20 but each sits one 0.05 step from "
                             "a cliff (teatime loses 7.6 mIoU from 0.20->0.25, "
                             "ramen loses 77 from 0.30->0.40). 0.15 is clear of "
                             "every cliff and still +1.0 mIoU over cuML "
                             "HDBSCAN, at ~50x less compute.")
    parser.add_argument("--min-size", type=int, default=64)
    parser.add_argument("--with-position", type=float, default=0.0,
                        help="Weight on xyz when clustering (equal-variance "
                             "with the features near 0.4).")
    parser.add_argument("--with-color", type=float, default=0.0,
                        help="Weight on base colour when clustering "
                             "(equal-variance near 3.5).")
    parser.add_argument("--ioa-sweep", default=None,
                        help="Comma-separated IoA thresholds to score in one "
                             "pass, e.g. 0.3,0.5,0.7,0.9. Reuses the same "
                             "render and grounding for every value.")
    parser.add_argument("--decoder", default="instances",
                        choices=["instances", "fisher"],
                        help="instances = cluster then select by IoA (default). "
                             "fisher = project the rendered features onto a "
                             "discriminant fitted on the reference view; no "
                             "clustering, no tau, no IoA.")
    parser.add_argument("--oracle", action="store_true",
                        help="Also report the best IoU any union of instances "
                             "could reach, which separates a decomposition "
                             "ceiling from a grounding/selection error.")
    parser.add_argument("--refit", action="store_true",
                        help="Ignore the cached clustering and fit again.")
    parser.add_argument("--min-cluster-size", type=int, default=32)
    parser.add_argument("--min-samples", type=int, default=16)
    parser.add_argument("--selection-epsilon", type=float, default=0.0)
    parser.add_argument("--selection-method", default="eom",
                        choices=["eom", "leaf"])
    parser.add_argument("--diagnose", action="store_true",
                        help="Report why the boundary score lags: how much of "
                             "each ground-truth mask lands on cells no cluster "
                             "claims, and whether dilating the prediction "
                             "recovers IoU (i.e. the mask is eroded, not ragged).")
    parser.add_argument("--density-weight", action="store_true",
                        help="Scale multicut edge weights by min cell density, "
                             "so links across empty space cannot merge things.")
    parser.add_argument("--silhouette", action="store_true",
                        help="Render per-instance silhouettes and threshold at "
                             "0.2, as the reference does, instead of using the "
                             "hard nearest-centroid partition.")
    parser.add_argument("--silhouette-threshold", type=float, default=0.2)
    parser.add_argument("--readout", default="auto",
                        choices=["auto", "centroid", "silhouette", "argmax"],
                        help="How a cluster becomes pixels. auto = centroid for "
                             "HDBSCAN, argmax for graph partitions.")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, _, dataset_args = load_model(args.checkpoint, device, args.model)
    scene_dir = Path(dataset_args.data_path) / dataset_args.scene
    truth = load_ground_truth(scene_dir)

    labels = None
    used_cache = False
    if args.clustering == "hdbscan_aug":
        from radfoam_model.instance_cluster import fit_clusters_augmented
        labels, clustering = fit_clusters_augmented(
            model.att_feat, positions=model.primal_points.detach().float(),
            colours=model.att_dc.detach().float(),
            with_position=args.with_position, with_color=args.with_color,
        )
        if args.readout == "auto":
            args.readout = "argmax"
        args.silhouette = args.readout == "silhouette"
    elif args.clustering == "hdbscan_full":
        from radfoam_model.instance_cluster import (
            fit_clusters_full,
            load_cached_clustering,
        )

        # Prefer the clustering cached by `cluster_cells.py`. Two fits of the
        # same checkpoint disagree on both the number of instances and their
        # ids, so an eval that refits is not comparable with the viewer export
        # or the language table built from the cache.
        clustering, labels = (None, None)
        if not args.refit:
            clustering, labels = load_cached_clustering(
                args.checkpoint, model.att_feat
            )
            used_cache = clustering is not None
            if clustering is not None and labels is None:
                print("cached clustering has no per-cell labels (sampled fit); "
                      "the argmax readout needs them -- refitting")
                clustering = None
        if clustering is None:
            used_cache = False
            if not args.refit:
                print("no usable cache; fitting (run `cluster_cells.py "
                      "--checkpoint ... --method full` to avoid this)")
            labels, clustering = fit_clusters_full(
                model.att_feat,
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_epsilon=args.selection_epsilon,
                cluster_selection_method=args.selection_method,
            )
        if args.readout == "auto":
            args.readout = "argmax"
        args.silhouette = args.readout == "silhouette"
    elif args.clustering == "hdbscan":
        clustering = fit_clusters(model.att_feat)
        if args.readout == "auto":
            args.readout = "centroid"
        if args.readout != "centroid":
            # Per-cell labels so HDBSCAN can go through the same silhouette
            # path -- the control that separates "multicut is too fine for the
            # 0.2 threshold" from "the silhouette renderer is wrong".
            from radfoam_model.instance_cluster import assign as assign_cells
            labels = assign_cells(model.att_feat.detach().float(), clustering)
            args.silhouette = args.readout == "silhouette"
    else:
        kwargs = ({} if args.clustering == "felzenszwalb"
                  else dict(tau=args.tau, metric="euclidean"))
        result = fit_graph_clusters(
            model.att_feat, model.point_adjacency, model.point_adjacency_offsets,
            method=args.clustering, min_size=args.min_size,
            density=model.get_primal_density() if args.density_weight else None,
            **kwargs
        )
        labels = result.labels
        clustering = clustering_from_labels(model.att_feat, labels)
        if args.readout == "auto":
            args.readout = "argmax"  # centroids cannot express a spatial partition
        args.silhouette = args.readout == "silhouette"
    print(f"{args.clustering}: {clustering.n_clusters} instances "
          f"({100 * clustering.noise_fraction:.1f}% unassigned)", flush=True)

    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    data.reload(split="test", downsample=min(dataset_args.downsample))
    height, width = data.img_wh[1], data.img_wh[0]
    id_maps = predict_masks(model, data, clustering, device)

    views = sorted(truth)
    index = {n: i for i, n in enumerate(data.image_names)}
    reference = views[0]
    reference_rgb = (
        data.rgbs[index[reference]].reshape(height, width, -1)[..., :3]
        .clamp(0, 1).cpu().numpy() * 255
    ).astype(np.uint8)

    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        SamModel,
        SamProcessor,
    )

    dino_proc = AutoProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(
        DINO_MODEL).to(device).eval()
    sam_proc = SamProcessor.from_pretrained(SAM_MODEL)
    sam = SamModel.from_pretrained(SAM_MODEL).to(device).eval()

    prompts = sorted({p for v in truth.values() for p in v})
    per_class_iou, per_class_biou, selections = {}, {}, {}
    per_class_oracle = {}
    sweep = ([float(x) for x in args.ioa_sweep.split(",")]
             if args.ioa_sweep else None)
    sweep_scores = {}
    diagnostics = []
    grounding_iou = {}

    if args.decoder == "fisher":
        print(f"fisher decoder: rendering features for {len(views)} views",
              flush=True)
        feature_maps = render_features(model, data, views, index, device)

    if labels is not None and args.readout == "argmax":
        print(f"argmax readout over {clustering.n_clusters} clusters", flush=True)
        maps = render_argmax_labels(model, data, labels, clustering.n_clusters,
                                    views, index, device)
        id_maps = {v: maps[i] for i, v in enumerate(views)}
        labels = None            # fall through to the plain partition path
    if labels is not None:
        cluster_ids = list(range(clustering.n_clusters))
        print(f"rendering {len(cluster_ids)} silhouettes in the reference view",
              flush=True)
        ref_sil = render_silhouettes(model, data, labels, cluster_ids,
                                     [reference], index, device,
                                     args.silhouette_threshold)[0]

    for prompt in prompts:
        grounded, n_boxes = ground_prompt(
            reference_rgb, prompt, dino, dino_proc, sam, sam_proc, device
        )
        if args.decoder == "fisher":
            fisher_masks = fisher_decode(feature_maps, reference, grounded, views)
            chosen, ioa_table = [], None
        elif labels is not None:
            chosen = select_by_ioa_silhouette(ref_sil, grounded,
                                              args.ioa_threshold)
            ioa_table = None
        else:
            ioa_table = instance_ioa(id_maps[reference], grounded)
            chosen = [k for k, v in ioa_table.items() if v > args.ioa_threshold]
        if sweep and ioa_table is not None:
            for thr in sweep:
                sel = [k for k, v in ioa_table.items() if v > thr]
                for view_name in views:
                    if prompt not in truth[view_name]:
                        continue
                    g = truth[view_name][prompt]
                    pr = (np.isin(id_maps[view_name], sel) if sel
                          else np.zeros_like(g))
                    sweep_scores.setdefault(thr, {}).setdefault(
                        prompt, []).append((iou(g, pr), boundary_iou(g, pr)))
        selections[prompt] = chosen
        # How good is the grounding itself? The reference view has ground truth
        # for most prompts, so the 2D mask GroundingDINO+SAM produce can be
        # scored directly -- separating "the detector was wrong" from
        # everything the 3D side does afterwards.
        if prompt in truth[reference]:
            grounding_iou[prompt] = iou(truth[reference][prompt], grounded)
        print(f"{prompt:<28} {n_boxes} box(es), grounded "
              f"{100 * grounded.mean():5.2f}% of frame -> "
              f"{len(chosen)} instance(s) {chosen}", flush=True)

        oracle_ious = []
        ious, bious = [], []
        if labels is not None:
            per_view = (render_silhouettes(model, data, labels, chosen, views,
                                           index, device,
                                           args.silhouette_threshold)
                        if chosen else None)
        for view_name in views:
            if prompt not in truth[view_name]:
                continue
            gt = truth[view_name][prompt]
            if args.decoder == "fisher":
                pred = fisher_masks[view_name]
                if pred.shape != gt.shape:
                    pred = cv2.resize(pred.astype(np.uint8),
                                      (gt.shape[1], gt.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            elif labels is not None:
                pred = (per_view[views.index(view_name)].any(axis=0)
                        if per_view is not None else np.zeros_like(gt))
            else:
                pred = (np.isin(id_maps[view_name], chosen) if chosen
                        else np.zeros_like(gt))
            ious.append(iou(gt, pred))
            bious.append(boundary_iou(gt, pred))
            if args.oracle:
                # Greedy best union: add instances in order of overlap with the
                # ground truth while the IoU still improves. This is the
                # ceiling the decomposition permits, whatever the grounding
                # and IoA selection happen to pick.
                idm = id_maps[view_name]
                cand = [k for k in np.unique(idm) if k != NOISE_ID]
                gain = sorted(
                    ((np.logical_and(idm == k, gt).sum(), k) for k in cand),
                    reverse=True,
                )
                chosen_o, best_o, cur = [], 0.0, np.zeros_like(gt)
                for overlap, k in gain:
                    if overlap == 0:
                        break
                    trial = np.logical_or(cur, idm == k)
                    s = iou(gt, trial)
                    if s <= best_o:
                        continue
                    best_o, cur, _ = s, trial, chosen_o.append(int(k))
                oracle_ious.append(best_o)
            if args.diagnose:
                unclaimed = float((id_maps[view_name][gt] == NOISE_ID).mean())
                best_d, best_i = 0, iou(gt, pred)
                for d in (1, 2, 3, 5, 8):
                    grown = cv2.dilate(pred.astype(np.uint8),
                                       np.ones((3, 3), np.uint8),
                                       iterations=d).astype(bool)
                    if iou(gt, grown) > best_i:
                        best_d, best_i = d, iou(gt, grown)
                diagnostics.append({
                    "prompt": prompt, "view": view_name,
                    "gt_px": int(gt.sum()), "pred_px": int(pred.sum()),
                    "ratio": float(pred.sum() / max(gt.sum(), 1)),
                    "gt_on_unclaimed": unclaimed,
                    "iou": iou(gt, pred), "best_dilate": best_d,
                    "iou_dilated": best_i,
                })
        per_class_iou[prompt] = float(np.mean(ious))
        if args.oracle:
            per_class_oracle[prompt] = float(np.mean(oracle_ious))
        per_class_biou[prompt] = float(np.mean(bious))

    # Their averaging: per class first, then over classes.
    miou = float(np.mean(list(per_class_iou.values())))
    mbiou = float(np.mean(list(per_class_biou.values())))

    if args.diagnose and diagnostics:
        print(f"\n{'prompt':<26} {'gt px':>8} {'pred px':>8} {'ratio':>6} "
              f"{'gt on unclaimed':>16} {'IoU':>6} {'+dilate':>8} {'IoU@d':>6}")
        for d in diagnostics:
            print(f"{d['prompt'][:25]:<26} {d['gt_px']:>8} {d['pred_px']:>8} "
                  f"{d['ratio']:>6.2f} {100*d['gt_on_unclaimed']:>15.1f}% "
                  f"{d['iou']:>6.3f} {d['best_dilate']:>8} {d['iou_dilated']:>6.3f}")
        good = [d for d in diagnostics if d["iou"] > 0.3]
        if good:
            print(f"\nover the {len(good)} hits: mean pred/gt area "
                  f"{np.mean([d['ratio'] for d in good]):.3f} | mean GT pixels on "
                  f"unclaimed cells {100*np.mean([d['gt_on_unclaimed'] for d in good]):.1f}%"
                  f" | dilation recovers {100*(np.mean([d['iou_dilated'] for d in good]) - np.mean([d['iou'] for d in good])):+.2f} IoU")

    if grounding_iou:
        import numpy as _np
        print(f"\n{'prompt':<30}{'grounding IoU':>14}{'final IoU':>11}")
        for pr in sorted(grounding_iou):
            print(f"{pr:<30}{100*grounding_iou[pr]:>14.2f}"
                  f"{100*per_class_iou.get(pr, 0):>11.2f}")
        g = _np.mean(list(grounding_iou.values()))
        print(f"{'MEAN grounding IoU':<30}{100*g:>14.2f}"
              f"{100*_np.mean(list(per_class_iou.values())):>11.2f}")

    if sweep_scores:
        print(f"\n{'IoA':>6} {'mIoU':>8} {'mBIoU':>8}   (per-class mean, "
              f"{dataset_args.scene})")
        for thr in sorted(sweep_scores):
            per = sweep_scores[thr]
            mi = float(np.mean([np.mean([a for a, _ in v]) for v in per.values()]))
            mb = float(np.mean([np.mean([b for _, b in v]) for v in per.values()]))
            star = "  <-- current" if abs(thr - args.ioa_threshold) < 1e-9 else ""
            print(f"{thr:>6.2f} {100*mi:>8.2f} {100*mb:>8.2f}{star}")

    if args.oracle and per_class_oracle:
        o = float(np.mean(list(per_class_oracle.values())))
        m = float(np.mean(list(per_class_iou.values())))
        print(f"\n{'Category':<30} {'IoU':>7} {'ORACLE':>8} {'gap':>7}")
        for pr in prompts:
            g = per_class_oracle[pr] - per_class_iou[pr]
            print(f"{pr:<30} {100*per_class_iou[pr]:>7.2f} "
                  f"{100*per_class_oracle[pr]:>8.2f} {100*g:>7.2f}")
        print(f"{'-'*30} {'-'*7} {'-'*8} {'-'*7}")
        print(f"{'MEAN':<30} {100*m:>7.2f} {100*o:>8.2f} {100*(o-m):>7.2f}")
        print(f"\n  decomposition ceiling loss : {100-100*o:6.2f} mIoU")
        print(f"  grounding + selection loss : {100*(o-m):6.2f} mIoU")

    print(f"\n{'Category':<30} {'IoU':>7} {'BIoU':>7}  instances")
    for prompt in prompts:
        print(f"{prompt:<30} {100 * per_class_iou[prompt]:>7.2f} "
              f"{100 * per_class_biou[prompt]:>7.2f}  {selections[prompt]}")
    print(f"{'-' * 30} {'-' * 7} {'-' * 7}")
    print(f"{dataset_args.scene + ' Mean':<30} {100 * miou:>7.2f} "
          f"{100 * mbiou:>7.2f}")

    out = (Path(args.checkpoint)
           / f"lerf_mask_grounded_{Path(args.model).stem}"
             f"_{args.decoder}"
             f"_{args.clustering}"
             f"_{args.readout}"
             f"{f'_p{args.with_position}' if args.with_position else ''}"
             f"{f'_c{args.with_color}' if args.with_color else ''}"
             f"{f'_tau{args.tau}' if args.clustering in ('multicut', 'threshold') else ''}"
             f"_m{args.min_cluster_size}s{args.min_samples}"
             f"{'e%g' % args.selection_epsilon if args.selection_epsilon else ''}"
             f"{'_leaf' if args.selection_method == 'leaf' else ''}.json")
    out.write_text(json.dumps({
        "scene": dataset_args.scene, "protocol": "opensplat3d-grounded",
        "clustering": args.clustering, "tau": args.tau,
        "decoder": args.decoder,
        "density_weight": bool(args.density_weight),
        "with_position": args.with_position, "with_color": args.with_color,
        "cache_requested": not args.refit,
        # whether a cached per-cell clustering was actually read, which
        # only the hdbscan_full path does -- "not args.refit" alone says
        # nothing about it and reads as confirmation when it is not.
        "used_cache": bool(used_cache),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "selection_epsilon": args.selection_epsilon,
        "selection_method": args.selection_method,
        "readout": args.readout,
        "ioa_threshold": args.ioa_threshold,
        "miou": miou, "mbiou": mbiou,
        "miou_per_class": per_class_iou, "mbiou_per_class": per_class_biou,
        "selected_instances": selections,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
