"""Evaluate stored arms and render the cross-arm comparison.

Read `balanced` -- the mean of same-pair and diff-pair co-segmentation agreement.
It needs no object ids, so the video and per-image arms are comparable on it, and
it is granularity-invariant: chance is 0.5 and perfect is 1.0 no matter how many
masks an arm emits.

Do NOT compare arms on `agreement`. At pure chance it reads 0.50 with two masks
and 0.98 with a hundred, so an arm that emits many small masks scores near
perfect for free. It is kept for diagnosis only.

`purity` is computed for the video arms alone. The per-image arms assign object
ids per frame arbitrarily, so cross-frame id agreement there would measure
nothing but coincidence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sam_masks.colmap_tracks import load_observations
from sam_masks.frames import load_frame_index
from sam_masks.metrics import (
    agreement_by_separation,
    cosegmentation_agreement,
    label_purity,
    sample_point_labels,
    stability_descriptors,
    track_descriptors,
)
from sam_masks.paths import DOWNSAMPLE, arm_dir, resolve_output_root, scene_sparse_dir
from sam_masks.store import read_meta

ARMS = [
    ("sam31", "video"),
    ("sam31", "image"),
    ("sam21", "video"),
    ("sam21", "image"),
]

# A point labelled in only one view is trivially pure and would drag mean_purity
# toward 1.0 while evidencing nothing about cross-view consistency.
MIN_PURITY_OBSERVATIONS = 2


def evaluate_arm(arm_path, observations, n_frames, mode="video", seed=0):
    """Compute all metric families for one stored arm.

    observations maps frame_idx -> [(point_id, row, col), ...]. mode selects
    whether id-based metrics are meaningful; see the module docstring.
    """
    arm_path = Path(arm_path)
    per_frame_labels = {}
    counts, areas, obj_ids_per_frame = [], [], []

    for frame_idx in range(n_frames):
        label_path = arm_path / "labels" / f"{frame_idx:06d}.png"
        if not label_path.exists():
            continue
        labels = np.array(Image.open(label_path))

        present = np.unique(labels)
        present = present[present != 0]
        counts.append(len(present))
        areas.append([int((labels == v).sum()) for v in present])
        obj_ids_per_frame.append([int(v) for v in present])

        points = observations.get(frame_idx, [])
        if points:
            per_frame_labels[frame_idx] = sample_point_labels(labels, points)

    rng = np.random.default_rng(seed)
    result = {
        "cosegmentation": cosegmentation_agreement(per_frame_labels, rng=rng),
        "by_separation": agreement_by_separation(
            per_frame_labels, rng=np.random.default_rng(seed)
        ),
        "stability": stability_descriptors(counts, areas),
        "frames_evaluated": len(counts),
    }
    if mode == "video":
        result["purity"] = label_purity(
            per_frame_labels, min_observations=MIN_PURITY_OBSERVATIONS
        )
        result["tracks"] = track_descriptors(obj_ids_per_frame)
    else:
        result["purity"] = None
        result["tracks"] = None
    return result


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "undef" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def render_summary(scene, reports, provenance=None):
    """Render a markdown comparison of every evaluated arm."""
    lines = [
        f"# Cross-view mask consistency — {scene}",
        "",
        "**Read `balanced`.** It is the mean of same-pair and diff-pair",
        "co-segmentation agreement over COLMAP 3D-point correspondences. It uses",
        "no object ids, so video and per-image arms are directly comparable, and",
        "it does not reward emitting more masks. **Chance is 0.500, perfect is",
        "1.000** — read every value against that scale. `undef` means the arm",
        "produced no differing pairs at all, which is undefined, not perfect.",
        "",
        "**Do not compare arms on `agreement`.** At pure chance it reads 0.50",
        "with two masks and 0.98 with a hundred, so it rewards granularity",
        "rather than consistency. It is shown for diagnosis only.",
        "",
        "`purity` (does one 3D point keep one object id across views) needs",
        "cross-frame identity, so it is n/a for the per-image arms by",
        "construction — those assign ids per frame arbitrarily.",
        "",
        "| arm | balanced | purity | agreement | same-pair | diff-pair | masks/frame | tracks | frames |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm, report in reports.items():
        coseg = report.get("cosegmentation", {})
        stability = report.get("stability", {})
        tracks = report.get("tracks") or {}
        purity = report.get("purity")
        lines.append(
            "| {} | **{}** | {} | {} | {} | {} | {} | {} | {} |".format(
                arm,
                _fmt(coseg.get("balanced")),
                _fmt(purity.get("mean_purity")) if purity else "n/a",
                _fmt(coseg.get("agreement")),
                _fmt(coseg.get("same_pair_agreement")),
                _fmt(coseg.get("diff_pair_agreement")),
                _fmt(stability.get("mean_masks_per_frame")),
                _fmt(tracks.get("n_tracks")) if tracks else "n/a",
                _fmt(report.get("frames_evaluated")),
            )
        )

    # The separation curve, which is the informative view: a method carrying real
    # cross-view identity holds up as views get further apart.
    curves = {a: r.get("by_separation") for a, r in reports.items() if r.get("by_separation")}
    if curves:
        buckets = list(next(iter(curves.values())).keys())
        lines += [
            "",
            "## Agreement by viewpoint separation",
            "",
            "`balanced` restricted to view pairs that many frames apart. Adjacent",
            "views are easy for any method; the question is how far identity",
            "survives. A flat row carries identity across the orbit, a row that",
            "decays towards 0.5 is mostly reporting that nearby frames look alike.",
            "",
            "| arm | " + " | ".join(f"{b} apart" for b in buckets) + " |",
            "| --- | " + " | ".join("---" for _ in buckets) + " |",
        ]
        for arm, curve in curves.items():
            cells = [_fmt(curve[b].get("balanced")) for b in buckets]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    if provenance:
        lines += ["", "## Run provenance", ""]
        lines.append("| arm | frames | objects | pruned | cap bound | failures | elapsed |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for arm, meta in provenance.items():
            lines.append(
                "| {} | {}/{} | {} | {} | {} | {} | {} s |".format(
                    arm,
                    meta.get("frames_written", meta.get("n_frames", "?")),
                    meta.get("n_frames", "?"),
                    meta.get("total_objects", "n/a"),
                    meta.get("n_pruned_total", "n/a"),
                    len(meta.get("cap_hits", [])) or "no",
                    len(meta.get("failures", [])),
                    meta.get("elapsed_s", "?"),
                )
            )
        lines += [
            "",
            "A cap that bound means proposals were dropped at that re-seed and the",
            "arm did not observe everything it could have. Failures are frames that",
            "errored and were skipped.",
            "",
            "**Confound to keep in mind:** SAM 3.1 enforces non-overlapping masks",
            "across objects; SAM 2.1's automatic mask generator does not. Mask",
            "counts and areas are therefore not measuring quite the same thing",
            "between the two models, though `balanced` is unaffected by it.",
        ]
    return "\n".join(lines) + "\n"


def run(scene, output_root=None, seed=0):
    output_root = Path(output_root) if output_root else resolve_output_root()
    summary_path = Path(output_root) / scene / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    sparse_dir = scene_sparse_dir(scene)
    if not sparse_dir.exists():
        # Masks stay valid; only the geometric metrics need COLMAP. Say so
        # rather than emitting a summary that looks complete.
        summary_path.write_text(
            f"# Cross-view mask consistency — {scene}\n\n"
            f"No COLMAP reconstruction at {sparse_dir}. Consistency metrics "
            "require sparse-point correspondences and were skipped. Masks under "
            "this scene are unaffected.\n"
        )
        print(f"WARNING: no COLMAP reconstruction for {scene}; metrics skipped")
        return summary_path

    reports, provenance = {}, {}
    for model, mode in ARMS:
        path = arm_dir(output_root, scene, model, mode)
        if not (path / "meta.json").exists():
            continue

        meta = read_meta(path)
        first_label = path / "labels" / "000000.png"
        if not first_label.exists():
            print(f"skipping {model}_{mode}: no labels written")
            continue

        shape = np.array(Image.open(first_label)).shape
        sequence = load_frame_index(path)
        name_to_frame = {name: i for i, name in enumerate(sequence.names)}

        obs = load_observations(sparse_dir, name_to_frame, DOWNSAMPLE[scene], shape)
        by_frame = {}
        for frame_idx, point_id, row, col in obs.xy:
            by_frame.setdefault(frame_idx, []).append((point_id, row, col))

        report = evaluate_arm(path, by_frame, meta["n_frames"], mode=mode, seed=seed)
        reports[f"{model}_{mode}"] = report
        provenance[f"{model}_{mode}"] = meta
        (path / "report.json").write_text(json.dumps(report, indent=2))
        print(f"{model}_{mode}: balanced={_fmt(report['cosegmentation']['balanced'])}")

    if not reports:
        print(f"no completed arms found for {scene}")

    summary_path.write_text(render_summary(scene, reports, provenance))
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate arms and write summary.md.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(run(args.scene, args.output_root, args.seed))


if __name__ == "__main__":
    main()
