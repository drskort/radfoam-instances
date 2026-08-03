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
    DOWNSAMPLE,
    arm_dir,
    resolve_output_root,
    scene_image_dir,
)
from sam_masks.store import FrameMasks, save_frame, write_meta


def run(scene, model, output_root=None, config=None, limit=None, force=False):
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
    out = arm_dir(output_root, scene, model, "image")
    out.mkdir(parents=True, exist_ok=True)

    source = scene_image_dir(scene)
    sequence = build_sequence(source, out / "frames")
    names = sequence.names[:limit] if limit else sequence.names

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
            masks, scores = backend.propose_masks(image, image_path=source / name)
            save_frame(
                out,
                FrameMasks(
                    frame_idx=frame_idx,
                    masks=masks,
                    obj_ids=list(range(1, masks.shape[0] + 1)),
                    scores=scores,
                    shape=image.shape[:2],
                ),
            )
        except Exception as exc:  # keep a multi-hour job alive
            failures.append({"frame_idx": frame_idx, "name": name, "error": repr(exc)})

    write_meta(
        out,
        {
            "scene": scene,
            "model": model,
            "mode": "image",
            "downsample": DOWNSAMPLE[scene],
            "n_frames": len(names),
            "n_computed": len(todo),
            "n_skipped": skipped,
            "config": vars(config),
            "failures": failures,
            "elapsed_s": round(time.time() - started, 1),
        },
    )
    if failures:
        print(f"{len(failures)} frame(s) failed; see meta.json")
    return out


def main():
    parser = argparse.ArgumentParser(description="Run the per-image segmentation arm.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", required=True, choices=["sam21", "sam31"])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N frames (for smoke tests).")
    parser.add_argument("--force", action="store_true",
                        help="Recompute frames that already have output.")
    args = parser.parse_args()

    out = run(args.scene, args.model, args.output_root, limit=args.limit,
              force=args.force)
    print(json.dumps({"output": str(out)}))


if __name__ == "__main__":
    main()
