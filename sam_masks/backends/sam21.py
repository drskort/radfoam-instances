"""SAM 2.1 adapter — the off-the-shelf arm.

Both halves exist upstream: SAM2AutomaticMaskGenerator for everything-mode and
build_sam2_video_predictor + propagate_in_video for tracking. This is also the
fallback proposal source for SAM 3.1 if Task 1's decision gate had come back
NO-GO (it did not).
"""

import numpy as np
import torch

from sam_masks.automask import AutomaskConfig, select_masks
from sam_masks.backends.base import Backend, Session

DEFAULT_CHECKPOINT = "facebook/sam2.1-hiera-large"


class Sam21Session(Session):
    def __init__(self, predictor, frames_dir, device="cuda", offload_video=True):
        self.predictor = predictor
        # SAM 2 caches memory features in bfloat16 when prompts are added, then
        # reads them back in memory attention during propagation. Without a
        # consistent autocast across the session's whole life those two halves
        # disagree and propagate_in_video dies with "mat1 and mat2 must have the
        # same dtype". Upstream's own notebooks wrap all predictor use this way.
        self._autocast = torch.autocast(device, dtype=torch.bfloat16)
        self._autocast.__enter__()
        try:
            # See sam31.py: decoded frames stay on the CPU so a long, large scene
            # does not exhaust the card before tracking state is allocated.
            self.state = predictor.init_state(
                video_path=str(frames_dir), offload_video_to_cpu=offload_video
            )
        except Exception:
            self._autocast.__exit__(None, None, None)
            raise
        self._next_id = 1

    def add_masks(self, frame_idx, masks):
        obj_ids = []
        for mask in masks:
            obj_id = self._next_id
            self._next_id += 1
            self.predictor.add_new_mask(
                inference_state=self.state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask,
            )
            obj_ids.append(obj_id)
        return obj_ids

    def propagate(self, start_frame=None, max_frames=None):
        for frame_idx, obj_ids, logits in self.predictor.propagate_in_video(
            self.state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=max_frames,
        ):
            masks = (logits > 0.0).cpu().numpy()
            # Empty masks are dropped to match SAM 3.1, which discards zero-area
            # masks internally. SAM 2 instead returns a zero-filled row for every
            # registered object on every frame, so without this filter its object
            # count would be constant by construction -- making a model that had
            # lost every track look like one with perfect retention, which is
            # precisely the quantity this experiment compares.
            yield frame_idx, {
                int(obj_id): masks[i, 0]
                for i, obj_id in enumerate(obj_ids)
                if masks[i, 0].any()
            }

    def remove_objects(self, obj_ids, frame_idx=0):
        for obj_id in obj_ids:
            # strict=False: an id the tracker has already dropped is not an error.
            self.predictor.remove_object(
                self.state, int(obj_id), strict=False, need_output=False
            )

    def close(self):
        try:
            self.predictor.reset_state(self.state)
            del self.state
        finally:
            self._autocast.__exit__(None, None, None)
        torch.cuda.empty_cache()


class Sam21Backend(Backend):
    name = "sam21"

    def __init__(
        self, checkpoint=DEFAULT_CHECKPOINT, config=None, device="cuda",
        max_objects=None, offload_video=True, levels=False,
    ):
        # max_objects is accepted and ignored: SAM 2 has no equivalent cap. The
        # runners pass it uniformly so SAM 3.1's builder default of 16 gets raised.
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf, build_sam2_video_predictor_hf

        self.config = config or AutomaskConfig()
        self.device = device
        self.offload_video = offload_video
        self.levels = levels
        self._image_model = build_sam2_hf(checkpoint, device=device)
        if levels:
            # Keeps SAM's subpart/part/whole decoder outputs distinct instead of
            # flattening them, so downstream supervision can use the hierarchy.
            from sam_masks.backends.sam_levels import build_level_aware_generator

            self._generator = build_level_aware_generator(self._image_model, self.config)
        else:
            self._generator = SAM2AutomaticMaskGenerator(
                self._image_model,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_thresh,
                box_nms_thresh=self.config.nms_iou_thresh,
            )
        self._video_predictor = build_sam2_video_predictor_hf(checkpoint, device=device)

    def propose_masks(self, image, image_path=None):
        # image_path is unused: SAM 2's generator works on in-memory arrays.
        if self.levels:
            from sam_masks.backends.sam_levels import generate_with_levels

            masks, scores, levels = generate_with_levels(self._generator, image)
        else:
            records = self._generator.generate(image)
            if not records:
                return np.zeros((0, *image.shape[:2]), dtype=bool), [], None
            masks = np.stack([r["segmentation"] for r in records]).astype(bool)
            scores = np.array([r["predicted_iou"] for r in records])
            levels = None

        if masks.shape[0] == 0:
            return np.zeros((0, *image.shape[:2]), dtype=bool), [], None

        # Re-run our own filtering so both arms share identical post-processing,
        # including the top_k cap the upstream generator does not apply. Levels
        # ride along on the same selection so they cannot drift out of step.
        selected = select_masks(masks, scores, self.config)
        if not selected:
            return np.zeros((0, *image.shape[:2]), dtype=bool), [], None
        kept_levels = [int(levels[i]) for i in selected] if levels is not None else None
        return masks[selected], [float(scores[i]) for i in selected], kept_levels

    def start_session(self, frames_dir):
        return Sam21Session(
            self._video_predictor, frames_dir, device=self.device,
            offload_video=self.offload_video,
        )
