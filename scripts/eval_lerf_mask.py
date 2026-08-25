"""Score a trained scene on the LERF-Mask benchmark.

The benchmark (Gaussian Grouping's annotation of three LERF scenes) grades
open-vocabulary 3D instance segmentation: given a text prompt, produce a binary
mask in a *held-out* view, and compare against a human annotation with IoU and
Boundary IoU.

The pipeline under test is the one in extract_instance_language.py -- cluster
the learned per-primitive features into 3D instances, give each instance a
SigLIP embedding built from crops of the *training* views, then answer a prompt
by picking the instance whose embedding scores highest. The only thing added
here is rendering that instance into the graded view and scoring it.

Boundary IoU follows Cheng et al. 2021: intersect each mask with the band of
width 2% of the image diagonal inside its own boundary, then take IoU of the
two bands. It is the metric that punishes a mask which is roughly in the right
place but ragged at the edges, which is exactly where a soft blend of
overlapping primitives is expected to lose to a hard partition of space.

    python scripts/eval_lerf_mask.py \
            --checkpoint output/figurines_inst_geo --model model.pt
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
from PIL import Image

from configs import *  # noqa: F401,F403
from data_loader import DataHandler
from radfoam_model.instance_cluster import NOISE_ID, assign, fit_clusters

# Same directory, and this file is normally run as a script, so scripts/ is
# already on sys.path -- import it as a bare module rather than a package.
from extract_instance_language import (  # noqa: E402
    DEFAULT_VLM,
    collect_instance_views,
    encode_instances,
    load_model,
)

BOUNDARY_DILATION_RATIO = 0.02


def mask_to_boundary(mask, dilation_ratio=BOUNDARY_DILATION_RATIO):
    """The band of width `dilation_ratio * diagonal` just inside the mask."""
    height, width = mask.shape
    dilation = max(
        1, int(round(dilation_ratio * np.sqrt(height ** 2 + width ** 2)))
    )
    binary = mask.astype(np.uint8)
    # Pad with background first, as the reference implementation does: cv2.erode
    # otherwise leaves the image border untouched, so an object running off the
    # edge of the frame would have no boundary along that edge.
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=dilation)
    return binary - eroded[1:height + 1, 1:width + 1]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def boundary_iou(gt, pred):
    return iou(mask_to_boundary(gt), mask_to_boundary(pred))


def load_ground_truth(scene_dir):
    """{view_name: {prompt: bool mask}} from test_mask/<n>/<prompt>.png."""
    truth = {}
    for view_dir in sorted((Path(scene_dir) / "test_mask").iterdir()):
        if not view_dir.is_dir():
            continue
        # test_mask/3/ annotates images/test_3.jpg.
        prompts = {}
        for path in sorted(view_dir.glob("*.png")):
            prompts[path.stem] = np.asarray(Image.open(path).convert("L")) > 127
        truth[f"test_{view_dir.name}.jpg"] = prompts
    return truth


def instance_embeddings(model, checkpoint, dataset_args, clustering, vlm,
                        device, cache=True):
    """Per-instance language embeddings, reusing a stored table when present."""
    store_path = Path(checkpoint) / "instance_language_eval.pt"
    if cache and store_path.exists():
        store = torch.load(store_path)
        if store.get("n_clusters") == clustering.n_clusters:
            return store["embeddings"], store["instance_ids"]

    train = DataHandler(dataset_args, rays_per_batch=0, device=device)
    train.reload(split="train", downsample=min(dataset_args.downsample))
    best_views = collect_instance_views(model, train, clustering, device)
    embeddings, ids, _ = encode_instances(best_views, train, vlm, device)
    torch.save(
        {"embeddings": embeddings, "instance_ids": ids,
         "n_clusters": clustering.n_clusters, "vlm": vlm},
        store_path,
    )
    return embeddings, ids


def predict_masks(model, data, clustering, device):
    """Instance id per pixel, for every view in `data`."""
    predictions = {}
    height, width = data.img_wh[1], data.img_wh[0]
    with torch.no_grad():
        points, _, _, _ = model.get_trace_data()
        for view, name in enumerate(data.image_names):
            rays = data.rays[view].to(device).reshape(-1, 6)
            start = model.get_starting_point(rays, points, model.aabb_tree)
            _, feature, *_ = model(rays, start)
            predictions[name] = assign(
                feature.reshape(height, width, -1).float(), clustering
            ).cpu().numpy()
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--vlm", default=DEFAULT_VLM)
    parser.add_argument("--top-k", type=int, default=1,
                        help="Union this many best-scoring instances. 1 is the "
                             "strict reading; higher forgives an object that "
                             "the clustering split in two.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dump", action="store_true",
                        help="Write predicted-vs-GT overlays for inspection.")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, dataset_args = load_model(args.checkpoint, device, args.model)
    if getattr(model, "feat_dim", 0) == 0:
        raise SystemExit("this checkpoint has no instance features")

    scene_dir = Path(dataset_args.data_path) / dataset_args.scene
    truth = load_ground_truth(scene_dir)
    if not truth:
        raise SystemExit(f"no test_mask/ annotations under {scene_dir}")

    clustering = fit_clusters(model.att_feat)
    print(f"{clustering.n_clusters} instances "
          f"({100 * clustering.noise_fraction:.1f}% noise)")

    embeddings, ids = instance_embeddings(
        model, args.checkpoint, dataset_args, clustering, args.vlm, device,
        cache=not args.no_cache,
    )
    if not len(ids):
        raise SystemExit("no instance embeddings -- nothing to query")

    test = DataHandler(dataset_args, rays_per_batch=0, device=device)
    test.reload(split="test", downsample=min(dataset_args.downsample))
    predicted_ids = predict_masks(model, test, clustering, device)

    # One text encode for every prompt in the scene.
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

    dump_dir = Path(args.checkpoint) / "lerf_mask_eval"
    if args.dump:
        dump_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for view_name, per_prompt in sorted(truth.items()):
        if view_name not in predicted_ids:
            raise SystemExit(
                f"{view_name} is annotated but not in the test split; the "
                "loader and the benchmark disagree about the holdout"
            )
        id_map = predicted_ids[view_name]
        for prompt, gt in sorted(per_prompt.items()):
            best = np.argsort(-prompt_scores[prompt])[: args.top_k]
            pred = np.isin(id_map, [ids[j] for j in best]) & (id_map != NOISE_ID)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred.astype(np.uint8),
                                  (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            # Ceiling: the best single instance in this view, whether or not
            # language found it. Separates "the decomposition has no such
            # object" from "language picked the wrong object".
            oracle_iou, oracle_id = 0.0, -1
            for k in np.unique(id_map):
                if k == NOISE_ID:
                    continue
                score = iou(gt, id_map == k)
                if score > oracle_iou:
                    oracle_iou, oracle_id = score, int(k)

            rows.append({
                "view": view_name, "prompt": prompt,
                "instance": int(ids[best[0]]),
                "iou": iou(gt, pred), "biou": boundary_iou(gt, pred),
                "oracle_iou": oracle_iou, "oracle_instance": oracle_id,
            })
            if args.dump:
                overlay = np.zeros((*gt.shape, 3), np.uint8)
                overlay[..., 1] = gt * 255            # green: ground truth
                overlay[..., 0] = pred * 255          # red: prediction
                cv2.imwrite(
                    str(dump_dir / f"{Path(view_name).stem}_{prompt}.png"),
                    overlay,
                )

    miou = float(np.mean([r["iou"] for r in rows]))
    mbiou = float(np.mean([r["biou"] for r in rows]))

    print(f"\n{'view':>12} {'prompt':<28} {'inst':>5} {'IoU':>7} {'BIoU':>7}"
          f" {'best':>5} {'oracle':>7}")
    for r in rows:
        print(f"{r['view']:>12} {r['prompt']:<28} {r['instance']:>5} "
              f"{r['iou']:>7.3f} {r['biou']:>7.3f} "
              f"{r['oracle_instance']:>5} {r['oracle_iou']:>7.3f}")
    oracle = float(np.mean([r["oracle_iou"] for r in rows]))
    matched = sum(r["instance"] == r["oracle_instance"] for r in rows)
    print(f"\n{dataset_args.scene}  mIoU {100 * miou:.1f}  "
          f"mBIoU {100 * mbiou:.1f}  ({len(rows)} prompt-view pairs, "
          f"top_k={args.top_k})")
    print(f"{dataset_args.scene}  ORACLE mIoU {100 * oracle:.1f}  "
          f"(best single instance per prompt; language picked it "
          f"{matched}/{len(rows)} times)")

    out = (Path(args.checkpoint)
       / f"lerf_mask_{Path(args.model).stem}_top{args.top_k}.json")
    out.write_text(json.dumps({
        "scene": dataset_args.scene, "model": args.model,
        "top_k": args.top_k, "n_clusters": clustering.n_clusters,
        "mIoU": miou, "mBIoU": mbiou, "rows": rows,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
