"""SAM 3.1 adapter.

SAM 3.1 ships no automatic mask generator, so everything-mode is assembled here
from a point grid fed to its tracker. Object Multiplex is what makes tracking a
few hundred grid-seeded objects affordable.

The video API is undocumented and two upstream defects must be worked around.
Both are explained where they occur below and recorded in full, with source
citations, in docs/plans/sam3-api-notes.md. Both poke at private
state and are pinned to the sam3 checkout under external/sam_backends/sam3; if
that checkout is bumped, re-verify them.
"""

import dataclasses
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from sam_masks.automask import AutomaskConfig, filter_and_dedupe, mask_to_interior_point
from sam_masks.backends.base import Backend, Session

VERSION = "sam3.1"


def masks_to_normalized_points(masks):
    """Convert masks to one normalized (x, y) point each, for point prompting.

    SAM 3.1 registers objects from point prompts; add_new_masks is private and
    unrouted. Returns (points, kept_indices) where points are in [0, 1] with x
    first, and kept_indices names the masks that produced a point — empty masks
    are dropped, so the caller must not assume positional alignment.
    """
    points, kept = [], []
    if masks.shape[0] == 0:
        return points, kept

    height, width = masks.shape[1:]
    for i, mask in enumerate(masks):
        interior = mask_to_interior_point(mask)
        if interior is None:
            continue
        row, col = interior
        points.append([(col + 0.5) / width, (row + 0.5) / height])
        kept.append(i)
    return points, kept


class Sam31Session(Session):
    def __init__(self, predictor, frames_dir, session_id="s0"):
        self.predictor = predictor
        self.session_id = session_id

        # Workaround (a): handle_request({"type": "start_session"}) raises
        # TypeError for SAM 3.1 -- Sam3BasePredictor.start_session passes
        # offload_state_to_cpu, which the 3.1 tracker's init_state rejects.
        # Replicate what start_session would have done, minus that argument.
        self.state = predictor.model.init_state(
            resource_path=str(frames_dir),
            offload_video_to_cpu=False,
            async_loading_frames=predictor.async_loading_frames,
        )
        now = time.time()
        predictor._all_inference_states[session_id] = {
            "state": self.state,
            "session_id": session_id,
            "start_time": now,
            "last_use_time": now,
        }

        # Workaround (b): _build_sam2_output early-returns {} for any frame not
        # in cached_frame_outputs, discarding the propagated masks *before* they
        # are merged. Point-driven prompting never populates that dict, so
        # without this every frame after the prompt frame comes back empty --
        # silently, with no error.
        self.state["cached_frame_outputs"] = {
            f: {} for f in range(self.state["num_frames"])
        }

        self._next_id = 1
        self.rejected = []

    def add_masks(self, frame_idx, masks):
        points, kept = masks_to_normalized_points(masks)
        obj_ids = []
        for point in points:
            obj_id = self._next_id
            response = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": self.session_id,
                    "frame_index": int(frame_idx),
                    "obj_id": obj_id,
                    "points": [point],
                    "point_labels": [1],
                    "rel_coordinates": True,
                    "clear_old_points": True,
                }
            )
            # Over the max_num_objects cap the model logs a warning and returns
            # outputs=None rather than raising. Record it; never silently skip.
            if response.get("outputs") is None:
                self.rejected.append({"frame_idx": int(frame_idx), "obj_id": obj_id})
                continue
            self._next_id += 1
            obj_ids.append(obj_id)
        return obj_ids

    def propagate(self):
        stream = self.predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": self.session_id,
                "propagation_direction": "forward",
                "start_frame_index": 0,
            }
        )
        for item in stream:
            outputs = item["outputs"]
            if not outputs:
                yield int(item["frame_index"]), {}
                continue
            obj_ids = [int(o) for o in outputs["out_obj_ids"]]
            masks = outputs["out_binary_masks"]
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()
            masks = np.asarray(masks).astype(bool)
            # Zero-area masks are dropped, matching sam21.py, so both arms count
            # live objects the same way. SAM 3.1 does its own zero-area drop in
            # _postprocess_output, but it runs *before*
            # _apply_object_wise_non_overlapping_constraints, which reassigns
            # every contested pixel to a single winner -- an object entirely
            # covered by another can come back all-False. Rare, but it would
            # inflate this backend's object counts relative to SAM 2's.
            yield int(item["frame_index"]), {
                obj_id: masks[i]
                for i, obj_id in enumerate(obj_ids)
                if masks[i].any()
            }

    def close(self):
        self.predictor.handle_request(
            {"type": "close_session", "session_id": self.session_id}
        )
        torch.cuda.empty_cache()


class Sam31Backend(Backend):
    name = "sam31"

    # 1024 so a full 32x32 grid fits; the builder's own default is 16. Raising
    # it is safe: buckets are allocated dynamically and the row-padding path that
    # would penalise large values is in the text-driven flow, which the
    # point-grid path bypasses.
    def __init__(self, config=None, device="cuda", max_objects=1024):
        from sam3.model_builder import build_sam3_predictor

        self.config = config or AutomaskConfig()
        self.device = device

        # Over the cap, add_prompt returns outputs=None with only a log warning,
        # so an undersized max_objects silently discards part of the grid rather
        # than failing. Refuse that configuration outright.
        grid_points = self.config.points_per_side ** 2
        if grid_points > max_objects:
            raise ValueError(
                f"points_per_side={self.config.points_per_side} needs "
                f"{grid_points} objects but max_objects={max_objects}; "
                "raise max_objects or lower points_per_side"
            )

        self._predictor = build_sam3_predictor(
            version=VERSION,
            max_num_objects=max_objects,   # builder default is 16
            use_fa3=False,                 # FA3 needs Hopper; A40 and 3090 are not
            use_rope_real=True,
            compile=False,
        )

    def propose_masks(self, image, image_path=None):
        """Everything-mode proposals via a point grid on a one-frame session.

        build_sam3_image_model cannot load the 3.1 checkpoint, so the per-image
        arm runs the video predictor over a directory holding a single frame.
        """
        height, width = image.shape[:2]
        staging = Path(tempfile.mkdtemp(prefix="sam31_frame_"))
        try:
            target = staging / "000000.jpg"
            if image_path is not None:
                target.symlink_to(Path(image_path).resolve())
            else:
                from PIL import Image as PILImage

                PILImage.fromarray(image).save(target, quality=95)

            with Sam31Session(
                self._predictor, staging, session_id="propose"
            ) as session:
                grid = self._normalized_grid()
                for x, y in grid:
                    obj_id = session._next_id
                    response = self._predictor.handle_request(
                        {
                            "type": "add_prompt",
                            "session_id": session.session_id,
                            "frame_index": 0,
                            "obj_id": obj_id,
                            "points": [[float(x), float(y)]],
                            "point_labels": [1],
                            "rel_coordinates": True,
                            "clear_old_points": True,
                        }
                    )
                    if response.get("outputs") is not None:
                        session._next_id += 1

                for _, masks_by_id in session.propagate():
                    if not masks_by_id:
                        break
                    masks = np.stack([masks_by_id[o] for o in sorted(masks_by_id)])
                    # SAM 3.1 returns no per-mask quality score on this path --
                    # add_sam2_new_points pins obj_id_to_score to 1.0 for every
                    # point-prompted object -- so rank by area: filter_and_dedupe
                    # needs an ordering, and larger regions are the more reliable
                    # proposals. Area fractions are not IoU predictions, so the
                    # pred_iou_thresh gate is disabled for this path; leaving it
                    # at 0.8 would reject every mask covering under 80% of the
                    # image, i.e. all of them. NMS, min area and top_k still run.
                    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
                    scores = areas / float(height * width)
                    config = dataclasses.replace(self.config, pred_iou_thresh=0.0)
                    return filter_and_dedupe(masks, scores, config)
                return np.zeros((0, height, width), dtype=bool), []
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def _normalized_grid(self):
        offsets = (np.arange(self.config.points_per_side) + 0.5) / self.config.points_per_side
        grid_x, grid_y = np.meshgrid(offsets, offsets)
        return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    def start_session(self, frames_dir):
        return Sam31Session(self._predictor, frames_dir)
