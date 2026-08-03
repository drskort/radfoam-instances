"""SAM 2.1 adapter — the off-the-shelf arm.

Both halves exist upstream: SAM2AutomaticMaskGenerator for everything-mode and
build_sam2_video_predictor + propagate_in_video for tracking. This is also the
fallback proposal source for SAM 3.1 if Task 1's decision gate had come back
NO-GO (it did not).
"""

import numpy as np
import torch

from sam_masks.automask import AutomaskConfig, filter_and_dedupe
from sam_masks.backends.base import Backend, Session

DEFAULT_CHECKPOINT = "facebook/sam2.1-hiera-large"


class Sam21Session(Session):
    def __init__(self, predictor, frames_dir, device="cuda"):
        self.predictor = predictor
        # SAM 2 caches memory features in bfloat16 when prompts are added, then
        # reads them back in memory attention during propagation. Without a
        # consistent autocast across the session's whole life those two halves
        # disagree and propagate_in_video dies with "mat1 and mat2 must have the
        # same dtype". Upstream's own notebooks wrap all predictor use this way.
        self._autocast = torch.autocast(device, dtype=torch.bfloat16)
        self._autocast.__enter__()
        try:
            self.state = predictor.init_state(video_path=str(frames_dir))
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
        self, checkpoint=DEFAULT_CHECKPOINT, config=None, device="cuda", max_objects=None
    ):
        # max_objects is accepted and ignored: SAM 2 has no equivalent cap. The
        # runners pass it uniformly so SAM 3.1's builder default of 16 gets raised.
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_hf, build_sam2_video_predictor_hf

        self.config = config or AutomaskConfig()
        self.device = device
        self._image_model = build_sam2_hf(checkpoint, device=device)
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
        records = self._generator.generate(image)
        if not records:
            return np.zeros((0, *image.shape[:2]), dtype=bool), []
        masks = np.stack([r["segmentation"] for r in records]).astype(bool)
        scores = np.array([r["predicted_iou"] for r in records])
        # Re-run our own filtering so both arms share identical post-processing,
        # including the top_k cap the upstream generator does not apply.
        return filter_and_dedupe(masks, scores, self.config)

    def start_session(self, frames_dir):
        return Sam21Session(self._video_predictor, frames_dir, device=self.device)
