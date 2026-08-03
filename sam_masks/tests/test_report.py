import numpy as np
import pytest

from sam_masks.report import evaluate_arm, render_summary
from sam_masks.store import FrameMasks, save_frame


def write_arm(path, n_frames, split_row=5, shift=0):
    """Two horizontal-band masks per frame; shift moves the boundary."""
    for frame_idx in range(n_frames):
        masks = np.zeros((2, 10, 10), dtype=bool)
        cut = split_row + (shift if frame_idx % 2 else 0)
        masks[0, :cut, :] = True
        masks[1, cut:, :] = True
        save_frame(
            path,
            FrameMasks(
                frame_idx=frame_idx,
                masks=masks,
                obj_ids=[1, 2],
                scores=[0.9, 0.9],
                shape=(10, 10),
            ),
        )


def test_evaluate_arm_computes_all_families_for_video(tmp_path):
    write_arm(tmp_path, 2)
    # Two points in each mask, so the pair set contains both same-mask and
    # different-mask pairs -- with only one point per mask there are no
    # same-mask pairs and `balanced` is legitimately undefined.
    pts = [(1, 2, 2), (2, 3, 3), (3, 7, 7), (4, 8, 8)]
    observations = {0: pts, 1: pts}

    result = evaluate_arm(tmp_path, observations, n_frames=2, mode="video")

    assert result["cosegmentation"]["balanced"] == 1.0
    assert result["purity"]["mean_purity"] == 1.0
    assert result["stability"]["mean_masks_per_frame"] == 2.0
    assert result["tracks"]["n_tracks"] == 2
    assert result["purity"]["n_points"] == 4
    assert result["frames_evaluated"] == 2


def test_image_arm_reports_no_purity_or_tracks(tmp_path):
    # Per-image ids are per-frame arbitrary, so id-based metrics would measure
    # coincidence rather than consistency.
    write_arm(tmp_path, 2)
    pts = [(1, 2, 2), (2, 3, 3), (3, 7, 7), (4, 8, 8)]
    observations = {0: pts, 1: pts}

    result = evaluate_arm(tmp_path, observations, n_frames=2, mode="image")

    assert result["purity"] is None
    assert result["tracks"] is None
    assert result["cosegmentation"]["balanced"] == 1.0


def test_evaluate_arm_skips_missing_frames(tmp_path):
    write_arm(tmp_path, 2)

    result = evaluate_arm(tmp_path, {}, n_frames=10, mode="video")

    assert result["frames_evaluated"] == 2


def test_render_summary_includes_every_arm():
    reports = {
        "sam31_video": {
            "cosegmentation": {"balanced": 0.91, "agreement": 0.95},
            "purity": {"mean_purity": 0.88},
            "stability": {"mean_masks_per_frame": 40.0},
            "tracks": {"n_tracks": 120},
            "frames_evaluated": 185,
        },
        "sam21_image": {
            "cosegmentation": {"balanced": 0.55, "agreement": 0.97},
            "purity": None,
            "stability": {"mean_masks_per_frame": 115.0},
            "tracks": None,
            "frames_evaluated": 185,
        },
    }

    text = render_summary("garden", reports)

    assert "sam31_video" in text and "sam21_image" in text
    assert "0.910" in text and "0.550" in text
    assert "garden" in text


def test_render_summary_marks_image_arm_purity_not_applicable():
    reports = {
        "sam21_image": {
            "cosegmentation": {"balanced": 0.5},
            "purity": None,
            "stability": {},
            "tracks": None,
            "frames_evaluated": 3,
        }
    }
    assert "n/a" in render_summary("garden", reports)


def test_render_summary_distinguishes_undefined_from_perfect():
    # A segmentation that collapsed to one mask everywhere yields no differing
    # pairs; that is undefined, and must not read as a perfect score.
    reports = {
        "sam31_video": {
            "cosegmentation": {
                "balanced": float("nan"),
                "agreement": 1.0,
                "diff_pair_agreement": float("nan"),
            },
            "purity": {"mean_purity": 1.0},
            "stability": {},
            "tracks": {},
            "frames_evaluated": 5,
        }
    }

    row = [
        line for line in render_summary("garden", reports).splitlines()
        if line.startswith("| sam31_video |")
    ][0]

    balanced_cell = row.split("|")[2].strip()
    assert balanced_cell == "**undef**", f"balanced read {balanced_cell!r}"


def test_render_summary_warns_against_comparing_on_agreement():
    text = render_summary("garden", {})
    assert "Do not compare arms on `agreement`" in text
    assert "Chance is 0.500" in text


def test_render_summary_reports_provenance_and_confound():
    provenance = {
        "sam31_video": {
            "n_frames": 185,
            "frames_written": 185,
            "total_objects": 669,
            "n_pruned_total": 170,
            "cap_hits": [{"frame_idx": 20, "dropped": 30}],
            "failures": [],
            "elapsed_s": 2751.0,
        }
    }

    text = render_summary("garden", {}, provenance)

    assert "669" in text and "170" in text
    assert "non-overlapping" in text, "the SAM3.1-vs-SAM2.1 confound must be stated"
