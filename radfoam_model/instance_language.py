"""Per-instance language embeddings, following OpenSplat3D's recipe exactly.

Reproduced from VisualComputingInstitute/opensplat3d (language/embed.py,
language/utils.py, language/lang_model.py) and their configs/lerf.yaml. The
differences from the first implementation in extract_instance_language.py are
the whole point -- measured on LERF-OVS, retrieval was costing 29 mIoU against
an oracle of 78.4, and these are what they do differently:

  * THREE crop levels per view, not one. Level 0 is the tight box; level i
    expands each side by width * ratio * (i + 1), so at ratio 0.3 the crops are
    [tight, +0.6w, +0.9w]. Their answer to a crop being dominated by its
    surroundings is multi-scale, not masking -- `masked` is off in lerf.yaml.
  * view ranking by (visible primitives / instance primitives) * (area / max
    area), not by projected area alone. The first term is what stops a mostly
    occluded instance winning on silhouette size.
  * each crop is padded to a SQUARE before the resize, so the encoder never
    sees a stretched object. Handing a rectangular crop straight to a
    processor resizes anisotropically instead, which is what the first version
    here did.
  * queries are wrapped in "an image of {}", not fed bare.
  * mean over all top-k crops AND all levels, after L2 normalising each.

Two encoders. `masqclip` is their default and needs the published
base_novel.pth; it takes the crop AND the instance's mask inside that crop, so
the object is identified to the encoder rather than merely centred. `siglip`
(siglip-so400m-patch14-384) is their fallback and sees the crop alone.
"""

import math

import numpy as np
import torch
import torchvision.transforms.functional as TF

SIGLIP = "google/siglip-so400m-patch14-384"
MASQCLIP_CKPT = "ckpts/MasQCLIP/base_novel.pth"
IMG_SIZE = {"siglip": (384, 384), "masqclip": (336, 336)}
PROMPT = "an image of {}"
TOPK = 5
LEVELS = 3
RATIO = 0.3
DYNAMIC_RATIO = True
SMALL_AREA = 0.0075          # below this the full ratio is used, else 0.1
MIN_PIXELS = 64


def multi_level_boxes(mask, levels=LEVELS, ratio=RATIO):
    """Tight box plus `levels - 1` progressively wider ones, as (r0, r1, c0, c1).

    Follows OpenMask3D via OpenSplat3D: expansion for level i is
    width * ratio * (i + 1), applied to each side. Bounds are half-open here;
    theirs are inclusive pixel coordinates that they slice with y2 + 1, which
    is the same extent.
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


def square_pad_resize(crop, crop_mask, img_size):
    """Zero-pad to a square, centred, then resize crop and mask together.

    crop is uint8 (3, h, w), crop_mask bool (h, w). Without the pad an object
    in a wide box reaches the encoder horizontally squashed, and the mask no
    longer lines up with the patch grid the attention mask is built from.
    """
    _, h, w = crop.shape
    side = max(h, w)
    pt, pl = (side - h) // 2, (side - w) // 2
    pad = [pl, pt, math.ceil((side - w) / 2), math.ceil((side - h) / 2)]
    crop = TF.resize(TF.pad(crop, pad), list(img_size))
    crop_mask = TF.resize(
        TF.pad(crop_mask.unsqueeze(0), pad), list(img_size),
        interpolation=TF.InterpolationMode.NEAREST,
    )
    return crop, crop_mask.squeeze(0)


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


class LanguageEncoder:
    """Whichever of OpenSplat3D's two published encoders is asked for."""

    def __init__(self, kind, device, ckpt=MASQCLIP_CKPT):
        self.kind = kind
        self.device = device
        self.img_size = IMG_SIZE[kind]
        if kind == "siglip":
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(SIGLIP)
            self.model = AutoModel.from_pretrained(SIGLIP).to(device).eval()
        elif kind == "masqclip":
            from third_party.masqclip import MasQCLIP

            self.model = MasQCLIP(["ViT-L/14@336px"])
            missing = self.model.from_pretrained(ckpt)
            # Only the `masqclip.` subtree is in the checkpoint, so the stock
            # CLIP text tower shows up as "missing" and must stay that way.
            loaded = [k for k in missing.missing_keys if k.startswith("visual.")]
            if loaded:
                raise RuntimeError(
                    f"{ckpt} did not supply {len(loaded)} visual weights, "
                    f"e.g. {loaded[:3]} -- wrong checkpoint?"
                )
            self.model = self.model.to(device).eval()
        else:
            raise ValueError(kind)

    @torch.no_grad()
    def encode_crops(self, crops, masks):
        """crops: list of uint8 (3, H, W); masks: list of bool (H, W)."""
        batch = torch.stack(crops).to(self.device)
        if self.kind == "siglip":
            images = [c.permute(1, 2, 0).cpu().numpy() for c in batch]
            inputs = self.processor(images=images, return_tensors="pt")
            feats = self.model.get_image_features(
                **{k: v.to(self.device) for k, v in inputs.items()})
        else:
            # One mask per crop, so each image carries a single query token.
            pixels = self.model.preprocess_images(batch.float().div(255.0))
            m = torch.stack(masks).to(self.device).unsqueeze(1).float()
            feats = self.model.get_image_embedding(pixels, m).squeeze(1)
        return torch.nn.functional.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, queries):
        prompts = [PROMPT.format(q) for q in queries]
        if self.kind == "siglip":
            inputs = self.processor(text=prompts, padding="max_length",
                                    return_tensors="pt").to(self.device)
            text = self.model.get_text_features(**inputs)
        else:
            text = self.model.get_text_embedding(prompts)
        return torch.nn.functional.normalize(text.float(), dim=-1)


def select_instances(scores, threshold=0.85):
    """Which instances answer each query. Multi-instance, as OpenSplat3D does.

    Their default is pred_type="max": shift each query's scores so the worst
    instance sits at zero, then keep everything within `threshold` of the best.
    A query is allowed to claim several instances, which is what lets a
    category that the clustering split across parts still be recovered whole.
    Rows of `scores` are queries, columns instances.

    They rescale by the model's logit scale first. Skipped here: it is a
    positive affine map applied to a whole row, and both the shift by the row
    minimum and the comparison against threshold * row maximum are invariant
    to it.
    """
    shifted = scores - scores.min(axis=1, keepdims=True)
    keep = shifted >= shifted.max(axis=1, keepdims=True) * threshold
    return [np.flatnonzero(row) for row in keep]
