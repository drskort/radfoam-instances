"""Per-image arm: independent everything-mode segmentation on every frame.

No object ids and no memory between frames, which is the point — this is the
baseline the video arms are compared against. Object ids are assigned
per-frame-arbitrarily (1..N), so only the id-free co-segmentation metric is
meaningful for this arm.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam_masks.automask import AutomaskConfig
from sam_masks.backends import get_backend
from sam_masks.frames import build_sequence
from sam_masks.paths import (
    arm_dir,
    resolve_output_root,
    scene_downsample,
    scene_image_dir,
)
from sam_masks.store import FrameMasks, read_meta, save_frame, write_meta


def run(scene, model, output_root=None, config=None, limit=None, force=False,
        tag=None):
    """Segment every frame independently.

    Frames whose .npz already exists are skipped unless force is set. This arm
    is embarrassingly parallel across frames -- nothing carries between them --
    so a run that dies partway can simply be resubmitted. That matters at the
    32x32 grid: SAM 3.1 has no batched prompting, so a 311-image scene is around
    eight hours, and losing all of it to a preemption at hour seven is not an
    acceptable failure mode.
    """
    config = config or AutomaskConfig()
    output_root = Path(output_root) if output_root else resolve_output_root()
    out = arm_dir(output_root, scene, model, "image", tag=tag)
    out.mkdir(parents=True, exist_ok=True)

    source = scene_image_dir(scene)
    # limit is applied when building the farm, not after: the video predictors
    # propagate over every frame present in that directory.
    sequence = build_sequence(source, out / "frames", limit=limit)
    names = sequence.names

    todo = [
        (i, name)
        for i, name in enumerate(names)
        if force or not (out / "masks" / f"{i:06d}.npz").exists()
    ]
    skipped = len(names) - len(todo)
    if skipped:
        print(f"resuming: {skipped}/{len(names)} frame(s) already done")
    if not todo:
        print("nothing to do; pass --force to recompute")

    # Loading the checkpoint costs ~20 s, so do not pay it for a no-op run.
    backend = get_backend(model, config=config) if todo else None
    failures = []
    started = time.time()

    for frame_idx, name in tqdm(todo, desc=f"{scene}/{model}/image"):
        try:
            image = np.array(Image.open(source / name).convert("RGB"))
            masks, scores, levels = backend.propose_masks(
                image, image_path=source / name
            )
            save_frame(
                out,
                FrameMasks(
                    frame_idx=frame_idx,
                    masks=masks,
                    obj_ids=list(range(1, masks.shape[0] + 1)),
                    scores=scores,
                    shape=image.shape[:2],
                    levels=levels,
                ),
            )
        except Exception as exc:  # keep a multi-hour job alive
            failures.append({"frame_idx": frame_idx, "name": name, "error": repr(exc)})

    # Carry forward the previous invocation's record. A resumed run recomputes
    # nothing, so writing a fresh meta would erase the failure list from the pass
    # that actually did the work -- and Task 14 reads `failures` to decide
    # whether an arm is trustworthy.
    previous = {}
    if (out / "meta.json").exists():
        try:
            previous = read_meta(out)
        except (OSError, ValueError):
            previous = {}

    write_meta(
        out,
        {
            "scene": scene,
            "model": model,
            "mode": "image",
            "downsample": scene_downsample(scene),
            "n_frames": len(names),
            "n_computed": len(todo),
            "n_skipped": skipped,
            "config": vars(config),
            "tag": tag,
            "failures": previous.get("failures", []) + failures,
            "elapsed_s": round(
                previous.get("elapsed_s", 0.0) + time.time() - started, 1
            ),
            "invocations": previous.get("invocations", 0) + 1,
        },
    )
    if failures:
        print(f"{len(failures)} frame(s) failed; see meta.json")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run the per-image segmentation arm.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", required=True, choices=["sam1", "sam21", "sam21_levels", "sam31"])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N frames (for smoke tests).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute frames that already have output.")
    parser.add_argument("--points-per-side", type=int, default=None,
                        help="Override the proposal grid density.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Override the cap on kept masks per frame.")
    parser.add_argument("--pred-iou-thresh", type=float, default=None,
                        help="Override the predicted-IoU filter.")
    parser.add_argument("--stability-thresh", type=float, default=None,
                        help="Override the stability-score filter.")
    parser.add_argument("--tag", default=None,
                        help="Suffix the output directory, to keep runs at "
                             "different settings side by side.")
    args = parser.parse_args()

    config = AutomaskConfig()
    if args.points_per_side is not None:
        config.points_per_side = args.points_per_side
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.pred_iou_thresh is not None:
        config.pred_iou_thresh = args.pred_iou_thresh
    if args.stability_thresh is not None:
        config.stability_thresh = args.stability_thresh

    out = run(args.scene, args.model, args.output_root, config=config,
              limit=args.limit, force=args.force, tag=args.tag)
    print(json.dumps({"output": str(out)}))


if __name__ == "__main__":
    main()
