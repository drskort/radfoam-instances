"""Show what a LERF-Mask query actually returns, over the image it returned it in.

The eval prints a number per (view, prompt); this draws the thing the number
scores. Two outputs per scene:

  overview_<view>.png -- the whole decomposition as a transparent overlay, so
      you can see which instances exist at all in a graded view.
  grid.png -- one cell per (prompt, view): the returned instance filled in,
      the ground truth drawn as a contour on top, captioned with its IoU. A
      cell where the fill and the outline disagree is a retrieval failure; a
      cell where they agree but the edges are ragged is a mask-quality
      failure. Those look identical in a table of numbers.

    MODEL=model_020000.pt sbatch scripts/render_lerf_slurm.sh output/ramen_inst_geo
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from configs import *  # noqa: F401,F403
from data_loader import DataHandler
from radfoam_model.instance_cluster import NOISE_ID, fit_clusters

from eval_lerf_mask import (  # noqa: E402
    boundary_iou,
    instance_embeddings,
    iou,
    load_ground_truth,
    predict_masks,
)
from extract_instance_language import DEFAULT_VLM, load_model  # noqa: E402

OVERLAY_ALPHA = 0.55
CELL_WIDTH = 420
GT_CONTOUR = (255, 255, 255)
PRED_COLOUR = (255, 64, 64)


def to_rgb(data, view, height, width):
    return (data.rgbs[view].reshape(height, width, -1)[..., :3]
            .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def blend(rgb, mask, colour, alpha=OVERLAY_ALPHA):
    out = rgb.copy()
    out[mask] = (
        (1 - alpha) * rgb[mask] + alpha * np.array(colour, dtype=np.float32)
    ).astype(np.uint8)
    return out


def draw_contour(image, mask, colour=GT_CONTOUR, thickness=2):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, colour, thickness)
    return image


def caption(width, text, height=26, colour=(235, 235, 235)):
    bar = np.full((height, width, 3), 24, dtype=np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                colour, 1, cv2.LINE_AA)
    return bar


def fit_cell(image, width=CELL_WIDTH):
    scale = width / image.shape[1]
    return cv2.resize(image, (width, int(round(image.shape[0] * scale))),
                      interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--vlm", default=DEFAULT_VLM)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, dataset_args = load_model(args.checkpoint, device, args.model)
    scene_dir = Path(dataset_args.data_path) / dataset_args.scene
    truth = load_ground_truth(scene_dir)

    clustering = fit_clusters(model.att_feat)
    print(f"{clustering.n_clusters} instances "
          f"({100 * clustering.noise_fraction:.1f}% noise)")
    embeddings, ids = instance_embeddings(
        model, args.checkpoint, dataset_args, clustering, args.vlm, device
    )

    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    data.reload(split="test", downsample=min(dataset_args.downsample))
    height, width = data.img_wh[1], data.img_wh[0]
    predicted = predict_masks(model, data, clustering, device)

    from transformers import AutoModel, AutoProcessor

    prompts = sorted({p for v in truth.values() for p in v})
    processor = AutoProcessor.from_pretrained(args.vlm)
    vlm = AutoModel.from_pretrained(args.vlm).to(device).eval()
    with torch.no_grad():
        inputs = processor(text=prompts, padding="max_length",
                           return_tensors="pt").to(device)
        text = torch.nn.functional.normalize(
            vlm.get_text_features(**inputs), dim=-1
        )
        scores = (text @ embeddings.to(device).T).cpu().numpy()
    prompt_scores = dict(zip(prompts, scores))

    out_dir = Path(args.checkpoint) / "lerf_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    views = sorted(truth)
    name_to_index = {n: i for i, n in enumerate(data.image_names)}

    # --- the decomposition itself, in context -----------------------------
    for view_name in views:
        rgb = to_rgb(data, name_to_index[view_name], height, width)
        id_map = predicted[view_name]
        overlay = rgb.copy()
        for k in np.unique(id_map):
            if k == NOISE_ID:
                continue
            overlay = blend(overlay, id_map == k,
                            clustering.colours[k].tolist())
        panel = np.hstack([rgb, overlay])
        stamped = np.vstack([
            caption(panel.shape[1],
                    f"{Path(args.checkpoint).name}  {view_name}   "
                    f"left: image   right: {clustering.n_clusters} instances"),
            panel,
        ])
        cv2.imwrite(str(out_dir / f"overview_{Path(view_name).stem}.png"),
                    cv2.cvtColor(stamped, cv2.COLOR_RGB2BGR))

    # --- what each query returned -----------------------------------------
    rows = []
    for prompt in prompts:
        cells = []
        for view_name in views:
            if prompt not in truth[view_name]:
                blank = np.full((height, width, 3), 18, dtype=np.uint8)
                cells.append(np.vstack([
                    fit_cell(blank),
                    caption(CELL_WIDTH, f"{prompt}  (not annotated)"),
                ]))
                continue

            gt = truth[view_name][prompt]
            id_map = predicted[view_name]
            best = np.argsort(-prompt_scores[prompt])[: args.top_k]
            pred = np.isin(id_map, [ids[j] for j in best]) & (id_map != NOISE_ID)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8),
                                  (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)

            rgb = to_rgb(data, name_to_index[view_name], height, width)
            cell = blend(rgb, pred, PRED_COLOUR)
            cell = draw_contour(cell, gt)
            score = iou(gt, pred)
            cells.append(np.vstack([
                fit_cell(cell),
                caption(CELL_WIDTH,
                        f"{prompt[:26]}  inst {ids[best[0]]}  "
                        f"IoU {score:.2f}  B {boundary_iou(gt, pred):.2f}",
                        colour=(120, 255, 120) if score > 0.5
                        else (255, 120, 120)),
            ]))
        rows.append(np.hstack(cells))

    header = caption(
        rows[0].shape[1],
        f"{Path(args.checkpoint).name}   filled = returned instance, "
        f"white outline = ground truth   (top_k={args.top_k})",
        height=32,
    )
    cv2.imwrite(str(out_dir / "grid.png"),
                cv2.cvtColor(np.vstack([header] + rows), cv2.COLOR_RGB2BGR))
    print(f"wrote {out_dir}/grid.png and {len(views)} overviews")


if __name__ == "__main__":
    main()
