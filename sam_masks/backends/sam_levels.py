"""Level-aware everything-mode generation for SAM 2.

SAM's mask decoder emits three masks per prompt point -- roughly subpart, part
and whole-object -- and `SAM2AutomaticMaskGenerator` flattens them into one
undifferentiated pile before filtering, losing the granularity hierarchy.

OpenSplat3D's contrastive instance loss is supervised per view from a label map,
and its reference implementation keeps SAM's levels as *separate* supervision
levels rather than flattening. That matters: supervised at a single granularity
the field is told once and for all whether a chair and its legs are the same
thing; supervised at several, it sees both relations and the embedding can carry
the hierarchy.

`_process_batch` below is upstream's, with a `levels` field added at MaskData
construction -- the only point where the level dimension still exists, since the
flatten immediately after it is what destroys the information. Everything
downstream (predicted-IoU filter, stability filter, crop-edge filter, NMS) then
carries the field along for free, because MaskData.filter handles tensors
generically.

This is pinned to the sam2 checkout under external/sam_backends/sam2. If that is
bumped, diff SAM2AutomaticMaskGenerator._process_batch against this copy.
"""

import numpy as np
import torch


def build_level_aware_generator(model, config):
    """Return a SAM 2 mask generator that records each mask's decoder level."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.utils.amg import (
        MaskData,
        batched_mask_to_box,
        calculate_stability_score,
        is_box_near_crop_edge,
        mask_to_rle_pytorch,
        uncrop_masks,
    )

    class LevelAwareGenerator(SAM2AutomaticMaskGenerator):
        def _process_batch(
            self, points, im_size, crop_box, orig_size, normalize=False
        ):
            orig_h, orig_w = orig_size

            points = torch.as_tensor(
                points, dtype=torch.float32, device=self.predictor.device
            )
            in_points = self.predictor._transforms.transform_coords(
                points, normalize=normalize, orig_hw=im_size
            )
            in_labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, iou_preds, low_res_masks = self.predictor._predict(
                in_points[:, None, :],
                in_labels[:, None],
                multimask_output=self.multimask_output,
                return_logits=True,
            )

            n_levels = masks.shape[1]
            # flatten(0, 1) is point-major, so the level index of row i is
            # i % n_levels. Attaching it here is the whole point of the override.
            levels = (
                torch.arange(n_levels, device=masks.device)
                .repeat(masks.shape[0])
                .to(torch.int64)
            )

            data = MaskData(
                masks=masks.flatten(0, 1),
                iou_preds=iou_preds.flatten(0, 1),
                points=points.repeat_interleave(n_levels, dim=0),
                low_res_masks=low_res_masks.flatten(0, 1),
                levels=levels,
            )
            del masks

            if not self.use_m2m:
                if self.pred_iou_thresh > 0.0:
                    data.filter(data["iou_preds"] > self.pred_iou_thresh)
                data["stability_score"] = calculate_stability_score(
                    data["masks"], self.mask_threshold, self.stability_score_offset
                )
                if self.stability_score_thresh > 0.0:
                    data.filter(
                        data["stability_score"] >= self.stability_score_thresh
                    )
            else:
                in_points = self.predictor._transforms.transform_coords(
                    data["points"], normalize=normalize, orig_hw=im_size
                )
                labels = torch.ones(
                    in_points.shape[0], dtype=torch.int, device=in_points.device
                )
                refined, ious = self.refine_with_m2m(
                    in_points, labels, data["low_res_masks"], self.points_per_batch
                )
                data["masks"] = refined.squeeze(1)
                data["iou_preds"] = ious.squeeze(1)

                if self.pred_iou_thresh > 0.0:
                    data.filter(data["iou_preds"] > self.pred_iou_thresh)
                data["stability_score"] = calculate_stability_score(
                    data["masks"], self.mask_threshold, self.stability_score_offset
                )
                if self.stability_score_thresh > 0.0:
                    data.filter(
                        data["stability_score"] >= self.stability_score_thresh
                    )

            data["masks"] = data["masks"] > self.mask_threshold
            data["boxes"] = batched_mask_to_box(data["masks"])

            keep = ~is_box_near_crop_edge(
                data["boxes"], crop_box, [0, 0, orig_w, orig_h]
            )
            if not torch.all(keep):
                data.filter(keep)

            data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
            data["rles"] = mask_to_rle_pytorch(data["masks"])
            del data["masks"]

            return data

        def generate(self, image):
            """Upstream's generate, with the level index carried into each record."""
            anns = super().generate(image)
            return anns

    generator = LevelAwareGenerator(
        model,
        points_per_side=config.points_per_side,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_thresh,
        box_nms_thresh=config.nms_iou_thresh,
        multimask_output=True,
    )
    return generator


def generate_with_levels(generator, image):
    """Run the generator and return (masks, scores, levels) as numpy arrays.

    Upstream's `generate` drops unknown MaskData fields when it builds its
    per-mask dicts, so the level has to be read off the MaskData directly.
    """
    mask_data = generator._generate_masks(image)
    from sam2.utils.amg import rle_to_mask

    n = len(mask_data["rles"])
    if n == 0:
        return (
            np.zeros((0, *image.shape[:2]), dtype=bool),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.int64),
        )

    def to_numpy(value):
        # _generate_masks moves some fields to CPU numpy during NMS and
        # post-processing, so the type here depends on which path ran.
        return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)

    masks = np.stack([rle_to_mask(rle) for rle in mask_data["rles"]]).astype(bool)
    scores = to_numpy(mask_data["iou_preds"]).astype(np.float64)
    levels = to_numpy(mask_data["levels"]).astype(np.int64)
    return masks, scores, levels
