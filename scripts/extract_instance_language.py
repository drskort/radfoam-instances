"""Attach a language embedding to every 3D instance, OpenSplat3D style.

Language features are NOT lifted into the field. Instead:

  1. cluster the learned instance features into 3D instances,
  2. project each instance into the training views,
  3. keep the top-k views where it is largest and least occluded,
  4. crop the *image* around that projection and encode the crop with a VLM,
  5. average over those views -> one embedding per instance.

This avoids the two problems with lifting CLIP/SigLIP per primitive: the storage
(1152 dims x millions of primitives) and, more fundamentally, that alpha
compositing a semantic embedding is not well defined -- the weighted mean of two
CLIP vectors is not a CLIP vector.

    python scripts/extract_instance_language.py \
            --checkpoint output/garden_inst_nogeo

Then query it:

    .venv/bin/python scripts/extract_instance_language.py \
        --checkpoint output/garden_inst_nogeo --query "a wooden table" "a plant"
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

import numpy as np
import torch
from PIL import Image

from configs import *  # noqa: F401,F403
from data_loader import DataHandler
from radfoam_model.instance_cluster import (
    NOISE_ID,
    assign,
    fit_clusters,
    load_cached_clustering,
)
from radfoam_model.scene import RadFoamScene

# OpenSplat3D's LERF/ScanNet++ settings.
TOP_K_VIEWS = 5
EXPANSION_SMALL = 0.3     # crop padding for small instances
EXPANSION_LARGE = 0.1     # ...and for large ones
SMALL_AREA_RATIO = 0.0075
MIN_INSTANCE_PIXELS = 64
DEFAULT_VLM = "google/siglip2-base-patch16-384"




def crop_box(mask, expansion, shape):
    """Bounding box of a mask, padded, clipped to the image."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        return None
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    pad_r = int((r1 - r0) * expansion)
    pad_c = int((c1 - c0) * expansion)
    return (
        max(0, r0 - pad_r), min(shape[0], r1 + pad_r),
        max(0, c0 - pad_c), min(shape[1], c1 + pad_c),
    )


def collect_instance_views(model, data, clustering, device, limit=None):
    """For each instance, the views where it is largest: {id: [(area, view, mask)]}."""
    n_views = data.rays.shape[0] if limit is None else min(limit, data.rays.shape[0])
    height, width = data.img_wh[1], data.img_wh[0]
    best = {k: [] for k in range(clustering.n_clusters)}

    with torch.no_grad():
        points, _, _, _ = model.get_trace_data()
        for view in range(n_views):
            rays = data.rays[view].to(device).reshape(-1, 6)
            start = model.get_starting_point(rays, points, model.aabb_tree)
            _, feature, *_ = model(rays, start)
            ids = assign(
                feature.reshape(height, width, -1).float(), clustering
            ).cpu().numpy()

            for k in np.unique(ids):
                if k == NOISE_ID:
                    continue
                mask = ids == k
                area = int(mask.sum())
                if area >= MIN_INSTANCE_PIXELS:
                    best[int(k)].append((area, view, mask))
            print(f"\rscanning views {view + 1}/{n_views}", end="", flush=True)
    print()

    for k in best:
        best[k].sort(key=lambda item: -item[0])
        best[k] = best[k][:TOP_K_VIEWS]
    return best


def encode_instances(best_views, data, vlm_name, device):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(vlm_name)
    vlm = AutoModel.from_pretrained(vlm_name).to(device).eval()

    height, width = data.img_wh[1], data.img_wh[0]
    embeddings, kept, boxes = [], [], []
    with torch.no_grad():
        for instance_id, views in sorted(best_views.items()):
            crops, used = [], []
            for area, view, mask in views:
                # Larger instances need less context; this mirrors the
                # reference's dynamic expansion ratio.
                ratio = area / (height * width)
                expansion = (EXPANSION_SMALL if ratio < SMALL_AREA_RATIO
                             else EXPANSION_LARGE)
                box = crop_box(mask, expansion, (height, width))
                if box is None:
                    continue
                r0, r1, c0, c1 = box
                rgb = (data.rgbs[view].reshape(height, width, -1)[r0:r1, c0:c1]
                       .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                crops.append(Image.fromarray(rgb[..., :3]))
                used.append((int(view), int(r0), int(r1), int(c0), int(c1)))

            if not crops:
                continue
            inputs = processor(images=crops, return_tensors="pt").to(device)
            features = vlm.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, dim=-1)
            # Mean over the best views, renormalised: one embedding per instance.
            embeddings.append(
                torch.nn.functional.normalize(features.mean(dim=0), dim=-1)
            )
            kept.append(instance_id)
            boxes.append(used)
            print(f"\rencoded instance {len(kept)}/{len(best_views)}",
                  end="", flush=True)
    print()
    if not embeddings:
        return torch.zeros(0), [], []
    return torch.stack(embeddings).cpu(), kept, boxes


def save_matches(store, query, ranked, data, out_dir, top=3):
    """Write out the crops the winning instances were encoded from.

    A score table alone cannot distinguish "found the table" from "picked an
    arbitrary instance"; the crop shows which it was.
    """
    boxes = store.get("boxes")
    if not boxes:
        print("  (store predates crop saving -- re-run the extraction to "
              "visualise matches)")
        return

    height, width = data.img_wh[1], data.img_wh[0]

    slug = "".join(c if c.isalnum() else "_" for c in query).strip("_")
    for rank, j in enumerate(ranked[:top], 1):
        for view, r0, r1, c0, c1 in boxes[j][:1]:
            rgb = (data.rgbs[view].reshape(height, width, -1)[r0:r1, c0:c1]
                   .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            path = out_dir / f"{slug}_{rank}_instance{store['instance_ids'][j]}.png"
            Image.fromarray(rgb[..., :3]).save(path)
            print(f"     -> {path}")


def run_query(store, queries, vlm_name, device, top=5,
              visualise=None, dataset_args=None):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(vlm_name)
    vlm = AutoModel.from_pretrained(vlm_name).to(device).eval()

    embeddings = store["embeddings"].to(device)
    ids = store["instance_ids"]
    data = None
    if visualise is not None:
        visualise.mkdir(parents=True, exist_ok=True)
        data = DataHandler(dataset_args, rays_per_batch=0, device=device)
        data.reload(split="train", downsample=min(dataset_args.downsample))
    with torch.no_grad():
        inputs = processor(text=list(queries), padding="max_length",
                           return_tensors="pt").to(device)
        text = torch.nn.functional.normalize(
            vlm.get_text_features(**inputs), dim=-1
        )
        scores = text @ embeddings.T

    for query, row in zip(queries, scores):
        best = row.argsort(descending=True)[:top]
        print(f"\n{query!r}")
        for rank, j in enumerate(best.tolist(), 1):
            print(f"  {rank}. instance {ids[j]:4d}   score {row[j].item():.4f}")
        if data is not None:
            save_matches(store, query, best.tolist(), data, visualise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vlm", default=DEFAULT_VLM)
    parser.add_argument("--limit-views", type=int, default=None)
    parser.add_argument("--model", default="model.pt",
                        help="Checkpoint file; point at a numbered snapshot to "
                             "avoid racing a live training run.")
    parser.add_argument("--query", nargs="*", default=None)
    parser.add_argument("--visualise", action="store_true",
                        help="With --query, also write the crops that each "
                             "top match was encoded from.")
    args = parser.parse_args()

    device = torch.device("cuda")
    store_path = (Path(args.checkpoint)
                  / f"instance_language_{Path(args.model).stem}.pt")

    if args.query:
        if not store_path.exists():
            raise SystemExit(f"{store_path} not found -- run without --query first")
        dataset_args = None
        out_dir = None
        if args.visualise:
            _, _, dataset_args = load_model(args.checkpoint, device, args.model)
            out_dir = Path(args.checkpoint) / "language_matches"
        run_query(torch.load(store_path), args.query, args.vlm, device,
                  visualise=out_dir, dataset_args=dataset_args)
        return

    model, _, dataset_args = load_model(args.checkpoint, device, args.model)
    if getattr(model, "feat_dim", 0) == 0:
        raise SystemExit("this checkpoint has no instance features")

    # Prefer the shared cache. Without it this re-fits a 60k-sample HDBSCAN,
    # which disagrees with the full fit on both instance count and ids -- and
    # then every embedding is attached to the wrong object.
    clustering, _cached_labels = load_cached_clustering(
        args.checkpoint, model.att_feat
    )
    if clustering is None:
        print("no usable clustering cache; fitting a fresh sampled one")
        clustering = fit_clusters(model.att_feat)
    # NOTE: view selection below assigns composited features by nearest
    # centroid. For a full-cloud clustering the exact readout is argmax over
    # per-cell labels, but that costs ceil(n_clusters/feat_dim) render passes
    # per view instead of one. Centroids are adequate here because this stage
    # only has to find views where an instance is large enough to crop.
    print(f"{clustering.n_clusters} instances "
          f"({100 * clustering.noise_fraction:.1f}% noise)")

    data = DataHandler(dataset_args, rays_per_batch=0, device=device)
    data.reload(split="train", downsample=min(dataset_args.downsample))

    best_views = collect_instance_views(
        model, data, clustering, device, limit=args.limit_views
    )
    embeddings, ids, boxes = encode_instances(best_views, data, args.vlm, device)
    torch.save(
        {"embeddings": embeddings, "instance_ids": ids, "boxes": boxes,
         "vlm": args.vlm, "n_clusters": clustering.n_clusters},
        store_path,
    )
    print(f"wrote {store_path}: {len(ids)} instances embedded "
          f"({embeddings.shape[-1] if len(ids) else 0}-dim)")


if __name__ == "__main__":
    main()
