"""Per-instance language embeddings, following OpenSplat3D's recipe exactly.

Reproduced from VisualComputingInstitute/opensplat3d (language/embed.py,
language/utils.py) and their configs/lerf.yaml. The differences from the first
implementation in extract_instance_language.py are the whole point -- measured
on LERF-OVS, retrieval was costing 29 mIoU against an oracle of 78.4, and these
are the four things they do differently:

  * THREE crop levels per view, not one. Level 0 is the tight box; level i
    expands each side by width * ratio * (i + 1), so at ratio 0.3 the crops are
    [tight, +0.6w, +0.9w]. Their answer to a crop being dominated by its
    surroundings is multi-scale, not masking -- `masked` is off in lerf.yaml.
  * view ranking by (visible primitives / instance primitives) * (area / max
    area), not by projected area alone. The first term is what stops a mostly
    occluded instance winning on silhouette size.
  * mean over all top-k crops AND all levels, after L2 normalising each.
  * siglip-so400m-patch14-384, which is their `siglip` option. MasQCLIP is
    their default but needs ckpts/MasQCLIP/base_novel.pth, which is not
    published.
"""

import numpy as np
import torch

MODEL = "google/siglip-so400m-patch14-384"
TOPK = 5
LEVELS = 3
RATIO = 0.3
DYNAMIC_RATIO = True
SMALL_AREA = 0.0075          # below this the full ratio is used, else 0.1
MIN_PIXELS = 64


def multi_level_boxes(mask, levels=LEVELS, ratio=RATIO):
    """Tight box plus `levels - 1` progressively wider ones.

    Follows OpenMask3D via OpenSplat3D: expansion for level i is
    width * ratio * (i + 1), applied to each side.
    """
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if rows.size == 0:
        return []
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    h, w = mask.shape
    x_exp, y_exp = int((c1 - c0) * ratio), int((r1 - r0) * ratio)
    boxes = [(r0, r1, c0, c1)]
    for i in range(1, levels):
        level = i + 1
        boxes.append((
            max(0, r0 - y_exp * level), min(h, r1 + y_exp * level),
            max(0, c0 - x_exp * level), min(w, c1 + x_exp * level),
        ))
    return boxes


def surface_cells(model, rays, device):
    """Index of the cell at the median-transmittance depth, one per ray.

    OpenSplat3D counts how many of an instance's Gaussians a view actually
    sees. The foam equivalent is which cells terminate a ray, which the tracer
    already computes for the quantile-depth loss -- ask it for the 0.5 quantile
    and read depth_indices. Rendered area alone is not a substitute: it is
    monotone in the silhouette, so pairing it with itself would leave the
    ranking identical to sorting by area.
    """
    points, attributes, adjacency, offsets = model.get_trace_data()
    start = model.get_starting_point(rays, points, model.aabb_tree)
    quantiles = torch.full((rays.shape[0], 2), 0.5, device=device)
    with torch.no_grad():
        out = model.pipeline.trace_forward(
            points, attributes, adjacency, offsets, rays, start,
            depth_quantiles=quantiles, return_contribution=False,
        )
    return out["depth_indices"][:, 0].long()


def rank_views(observations, total_cells):
    """OpenSplat3D's visibility score: coverage x normalised area.

    observations: list of (area, view, mask, visible_cells).
    """
    if not observations:
        return []
    max_area = max(o[0] for o in observations) or 1
    return sorted(
        observations,
        key=lambda o: (o[3] / max(total_cells, 1)) * (o[0] / max_area),
        reverse=True,
    )[:TOPK]


def embed_instances(crops_per_instance, device, model_name=MODEL, batch=32):
    """One L2-normalised embedding per instance: mean over crops and levels."""
    from PIL import Image  # noqa: F401  (callers pass PIL images)
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    vlm = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings, kept = [], []
    for n, (instance, crops) in enumerate(sorted(crops_per_instance.items())):
        if not crops:
            continue
        feats = []
        for start in range(0, len(crops), batch):
            inputs = processor(images=crops[start:start + batch],
                               return_tensors="pt").to(device)
            with torch.no_grad():
                f = vlm.get_image_features(**inputs)
            feats.append(torch.nn.functional.normalize(f, dim=-1))
        pooled = torch.cat(feats).mean(dim=0)
        embeddings.append(torch.nn.functional.normalize(pooled, dim=-1))
        kept.append(instance)
        if n % 25 == 0:
            print(f"\r  embedded {n + 1}/{len(crops_per_instance)}",
                  end="", flush=True)
    print()
    if not embeddings:
        return torch.zeros(0), []
    return torch.stack(embeddings).cpu(), kept


def encode_text(queries, device, model_name=MODEL):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    vlm = AutoModel.from_pretrained(model_name).to(device).eval()
    with torch.no_grad():
        inputs = processor(text=list(queries), padding="max_length",
                           return_tensors="pt").to(device)
        text = vlm.get_text_features(**inputs)
    return torch.nn.functional.normalize(text, dim=-1)
