"""Video-propagated arm: seed objects, track them, re-seed periodically.

Re-seeding is load-bearing. mip-NeRF 360 captures orbit a scene, so frame 0 sees
only part of it; seeding once would measure "objects visible at the start"
instead of the scene. Every reseed_every frames the proposer re-runs and any
proposal not already matched to a tracked mask is promoted to a new object.

Every promotion is recorded, because a high promotion rate against an
already-populated tracker is itself evidence that tracking is not holding.

Unlike the per-image arm, this one cannot resume frame by frame: propagation is a
single stateful pass and the tracker's memory is not reconstructible from masks
already on disk. Resume is therefore whole-arm -- a completed arm is skipped, an
interrupted one restarts.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam_masks.automask import AutomaskConfig, match_to_existing
from sam_masks.backends import get_backend
from sam_masks.frames import build_sequence
from sam_masks.paths import (
    DOWNSAMPLE,
    arm_dir,
    resolve_output_root,
    scene_image_dir,
)
from sam_masks.store import FrameMasks, read_meta, save_frame, write_meta

DEFAULT_RESEED_EVERY = 10
DEFAULT_RESEED_IOU = 0.5
# 512 tracked objects peaks around 30 GiB; 576 measured 33.9 GiB and 1024 OOMs a
# 48 GB A40. Re-seeding accumulates, so on a long scene this cap will bind --
# that is expected, and it is logged rather than applied silently.
DEFAULT_MAX_OBJECTS = 512
# Objects unseen for this many frames are released before the next re-seed. The
# model's cap counts every object ever registered, so without pruning a long
# orbit fills its budget with dead tracks and stops seeding entirely -- measured
# on garden, SAM 3.1 would saturate around frame 20 of 185.
DEFAULT_STALE_AFTER = 10


def reseed_frames(n_frames, every):
    """Frame indices at which the proposer re-runs."""
    if every <= 0:
        raise ValueError(f"reseed interval must be positive, got {every}")
    return list(range(0, n_frames, every))


def promote_unmatched(add_masks, frame_idx, proposals, tracked, iou_thresh):
    """Register proposals that no tracked mask already covers.

    Returns the newly registered masks, so the caller can extend its tracked set
    without re-running the match. Doing the matching once is not just cheaper --
    two independent calls could disagree and leave `tracked` out of step with
    what the session actually holds.
    """
    empty = np.zeros((0, *proposals.shape[1:]), dtype=bool)
    if proposals.shape[0] == 0:
        return empty

    unmatched = match_to_existing(proposals, tracked, iou_thresh)
    if not unmatched:
        return empty

    new = proposals[unmatched]
    add_masks(frame_idx, new)
    return new


def _is_complete(out, n_frames):
    """True if a previous invocation finished this arm over the same frame count.

    The frame count has to match. A completed 24-frame smoke run also carries
    complete=True, and skipping a 185-frame request on the strength of it would
    silently leave the arm eight times shorter than asked for.
    """
    if not (Path(out) / "meta.json").exists():
        return False
    try:
        meta = read_meta(out)
    except (OSError, ValueError):
        return False
    return bool(meta.get("complete")) and meta.get("n_frames") == n_frames


def run(
    scene,
    model,
    output_root=None,
    config=None,
    reseed_every=DEFAULT_RESEED_EVERY,
    reseed_iou=DEFAULT_RESEED_IOU,
    max_objects=DEFAULT_MAX_OBJECTS,
    stale_after=DEFAULT_STALE_AFTER,
    limit=None,
    force=False,
):
    config = config or AutomaskConfig()
    output_root = Path(output_root) if output_root else resolve_output_root()
    out = arm_dir(output_root, scene, model, "video")
    out.mkdir(parents=True, exist_ok=True)

    source = scene_image_dir(scene)
    # limit is applied when building the farm, not after: the video predictors
    # propagate over every frame present in that directory.
    sequence = build_sequence(source, out / "frames", limit=limit)
    names = sequence.names

    if not force and _is_complete(out, len(names)):
        print(f"arm already complete over {len(names)} frames; pass --force to redo")
        return out

    height, width = np.array(Image.open(source / names[0]).convert("RGB")).shape[:2]
    backend = get_backend(model, config=config, max_objects=max_objects)
    seed_points = reseed_frames(len(names), reseed_every)
    promotions, cap_hits, failures = [], [], []
    started = time.time()

    with backend.start_session(out / "frames") as session:
        # Seeding and propagation are INTERLEAVED, one segment per re-seed.
        #
        # The point is what a new proposal gets compared against. Matching it
        # against proposals harvested at earlier frames does not work on an
        # orbiting camera: the same physical object sits at different pixels a
        # few frames later, image-space IoU collapses to ~0, and every proposal
        # is promoted as new -- duplicating the whole scene at every re-seed and
        # turning the video arm into per-image segmentation with extra
        # bookkeeping. Measured on garden: 0 of 37 matched for SAM 2.1, 4 of 235
        # for SAM 3.1.
        #
        # So each segment propagates up to and including the next seed frame,
        # and the tracker's LIVE masks at that frame -- same viewpoint, so IoU
        # is meaningful -- are what the next round of proposals is matched
        # against.
        live = {}
        written = {}
        last_seen = {}
        prunings = []
        for i, seed in enumerate(seed_points):
            # Release tracks that have been gone long enough to call dead. This
            # is what keeps the object budget spent on things that still exist.
            stale = [o for o, f in last_seen.items() if seed - f > stale_after]
            if stale:
                session.remove_objects(stale, frame_idx=seed)
                for o in stale:
                    last_seen.pop(o, None)
                    live.pop(o, None)
                prunings.append({"frame_idx": seed, "n_pruned": len(stale)})

            try:
                image = np.array(Image.open(source / names[seed]).convert("RGB"))
                proposals, _, _ = backend.propose_masks(
                    image, image_path=source / names[seed]
                )
            except Exception as exc:  # a bad frame must not kill a long run
                failures.append(
                    {"stage": "seed", "frame_idx": seed, "error": repr(exc)}
                )
                proposals = np.zeros((0, height, width), dtype=bool)

            live_masks = (
                np.stack([live[o] for o in sorted(live)])
                if live
                else np.zeros((0, height, width), dtype=bool)
            )
            n_live = len(live)

            budget = max_objects - n_live
            if proposals.shape[0] and budget <= 0:
                cap_hits.append({"frame_idx": seed, "dropped": int(proposals.shape[0])})
                proposals = np.zeros((0, height, width), dtype=bool)
            elif proposals.shape[0] > budget:
                # proposals arrive score-ordered from filter_and_dedupe, so this
                # keeps the highest-scoring ones.
                cap_hits.append(
                    {"frame_idx": seed, "dropped": int(proposals.shape[0] - budget)}
                )
                proposals = proposals[:budget]

            new = promote_unmatched(
                session.add_masks, seed, proposals, live_masks, reseed_iou
            )
            promotions.append(
                {
                    "frame_idx": seed,
                    "n_live_before": n_live,
                    "n_proposed": int(proposals.shape[0]),
                    "n_new": int(new.shape[0]),
                }
            )

            # Propagate to the next seed frame inclusive, so its live masks are
            # known when that seed runs. The overlap frame is rewritten by the
            # next segment, which is what we want -- that pass includes the
            # objects added there.
            last = seed_points[i + 1] if i + 1 < len(seed_points) else len(names) - 1
            for frame_idx, masks_by_id in session.propagate(
                start_frame=seed, max_frames=last - seed + 1
            ):
                if frame_idx >= len(names):
                    break
                obj_ids = sorted(masks_by_id)
                masks = (
                    np.stack([masks_by_id[o] for o in obj_ids])
                    if obj_ids
                    else np.zeros((0, height, width), dtype=bool)
                )
                save_frame(
                    out,
                    FrameMasks(
                        frame_idx=frame_idx,
                        masks=masks,
                        obj_ids=obj_ids,
                        scores=[1.0] * len(obj_ids),
                        shape=(height, width),
                    ),
                )
                written[frame_idx] = len(obj_ids)
                for o in obj_ids:
                    last_seen[o] = frame_idx
                if frame_idx == last:
                    live = masks_by_id

        counts = [written[i] for i in sorted(written)]
        total_objects = session._next_id - 1

    if cap_hits:
        dropped = sum(c["dropped"] for c in cap_hits)
        print(
            f"max_objects={max_objects} bound at {len(cap_hits)} seed frame(s); "
            f"{dropped} proposal(s) dropped. See meta.json."
        )
    if failures:
        print(f"{len(failures)} seed frame(s) failed; see meta.json")

    write_meta(
        out,
        {
            "scene": scene,
            "model": model,
            "mode": "video",
            "downsample": DOWNSAMPLE[scene],
            "n_frames": len(names),
            "reseed_every": reseed_every,
            "reseed_iou": reseed_iou,
            "max_objects": max_objects,
            "stale_after": stale_after,
            "prunings": prunings,
            "n_pruned_total": sum(p["n_pruned"] for p in prunings),
            "cap_hits": cap_hits,
            "promotions": promotions,
            "failures": failures,
            "total_objects": int(total_objects),
            "objects_per_frame": counts,
            "frames_written": len(counts),
            "config": vars(config),
            "elapsed_s": round(time.time() - started, 1),
            "complete": len(counts) == len(names),
        },
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Run the video-propagated arm.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", required=True, choices=["sam1", "sam21", "sam21_levels", "sam31"])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reseed-every", type=int, default=DEFAULT_RESEED_EVERY)
    parser.add_argument("--reseed-iou", type=float, default=DEFAULT_RESEED_IOU)
    parser.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS)
    parser.add_argument("--stale-after", type=int, default=DEFAULT_STALE_AFTER,
                        help="Release objects unseen for this many frames.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the arm is already complete.")
    args = parser.parse_args()

    out = run(
        args.scene,
        args.model,
        args.output_root,
        reseed_every=args.reseed_every,
        reseed_iou=args.reseed_iou,
        max_objects=args.max_objects,
        stale_after=args.stale_after,
        limit=args.limit,
        force=args.force,
    )
    print(json.dumps({"output": str(out)}))


if __name__ == "__main__":
    main()
