"""SAM 1 adapter — the model OpenSplat3D actually uses.

Included as a baseline for the per-image arm because OpenSplat3D's reference
implementation imports `segment_anything.SamAutomaticMaskGenerator` directly. If
the newer models do not beat it at producing stable per-view supervision, there
is no reason to swap it out.

Image-only by design. SAM 1 has no video predictor and no memory across frames,
so `start_session` raises rather than pretending otherwise — the video arms are
not defined for this backend.
"""

import os
from pathlib import Path

import numpy as np

from sam_masks.automask import AutomaskConfig, select_masks
from sam_masks.backends.base import Backend, Session

# Meta's published ViT-H weights. Overridable so a shared copy can be used.
DEFAULT_CHECKPOINT = Path(
    os.environ.get("SAM1_CHECKPOINT", Path.home() / ".cache/sam1/sam_vit_h_4b8939.pth")
)
DEFAULT_MODEL_TYPE = "vit_h"


class Sam1Backend(Backend):
    name = "sam1"

    def __init__(
        self,
        checkpoint=DEFAULT_CHECKPOINT,
        model_type=DEFAULT_MODEL_TYPE,
        config=None,
        device="cuda",
        levels=False,
        **ignored,
    ):
        # ignored absorbs max_objects / offload_video, which the runners pass
        # uniformly and SAM 1 has no equivalent of.
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"SAM 1 checkpoint not found at {checkpoint}. Download it with:\n"
                "  curl -L -o ~/.cache/sam1/sam_vit_h_4b8939.pth \\\n"
                "    https://dl.fbaipublicfiles.com/segment_anything/"
                "sam_vit_h_4b8939.pth\n"
                "or point SAM1_CHECKPOINT at an existing copy."
            )

        self.config = config or AutomaskConfig()
        self.device = device
        self.levels = levels

        model = sam_model_registry[model_type](checkpoint=str(checkpoint))
        model.to(device)
        self._generator = SamAutomaticMaskGenerator(
            model,
            points_per_side=self.config.points_per_side,
            pred_iou_thresh=self.config.pred_iou_thresh,
            stability_score_thresh=self.config.stability_thresh,
            box_nms_thresh=self.config.nms_iou_thresh,
        )

    def propose_masks(self, image, image_path=None):
        # image_path is unused: SAM 1's generator works on in-memory arrays.
        records = self._generator.generate(image)
        if not records:
            return np.zeros((0, *image.shape[:2]), dtype=bool), [], None

        masks = np.stack([r["segmentation"] for r in records]).astype(bool)
        scores = np.array([r["predicted_iou"] for r in records])

        # Same post-processing as every other backend, so differences between
        # them are the models rather than the filtering.
        selected = select_masks(masks, scores, self.config)
        if not selected:
            return np.zeros((0, *image.shape[:2]), dtype=bool), [], None
        return masks[selected], [float(scores[i]) for i in selected], None

    def start_session(self, frames_dir):
        raise NotImplementedError(
            "SAM 1 has no video predictor; only the per-image arm is defined "
            "for this backend."
        )
