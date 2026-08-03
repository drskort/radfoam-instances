"""Cross-view consistency metrics computed against COLMAP correspondences.

Two families answer two different questions:

label_purity asks "does one 3D point keep one object id across views". It only
applies to the video arms, where ids mean something across frames.

cosegmentation_agreement asks "do two 3D points that share a mask in one view
also share a mask in another". It never looks at id values, only at whether two
points group together, so it applies equally to the video and per-image arms.
That is what makes the headline comparison fair.

The decomposition matters. A segmentation that puts everything in one mask
achieves perfect same-pair agreement, so the balanced mean of same-pair and
diff-pair agreement is the number to read, not the raw agreement rate.
"""

import itertools
from collections import Counter, defaultdict

import numpy as np

BACKGROUND = 0


def sample_point_labels(labels, points):
    """Look up the label at each (point_id, row, col); returns {point_id: label}."""
    return {int(pid): int(labels[row, col]) for pid, row, col in points}


def label_purity(per_frame_labels, min_observations=1):
    """Majority-vote purity and entropy of object ids per 3D point.

    per_frame_labels maps frame_idx -> {point_id: object_id}. Background
    observations (id 0) are dropped: falling on background is an absence of a
    label, not a label of its own, so it neither confirms nor breaks a point's
    identity.

    min_observations is the number of surviving labels a point needs to be
    counted. The default of 1 keeps every point that was labelled at all. Note
    that a point labelled in only one view is trivially pure, so it drags
    mean_purity towards 1.0 without evidencing any cross-view consistency; pass
    min_observations=2 to restrict the statistic to points that actually testify
    about consistency. Report which setting was used — the two differ.
    """
    by_point = defaultdict(list)
    for labels in per_frame_labels.values():
        for point_id, obj_id in labels.items():
            if obj_id != BACKGROUND:
                by_point[point_id].append(obj_id)

    purities, entropies = [], []
    for observed in by_point.values():
        if len(observed) < min_observations:
            continue
        counts = np.array(list(Counter(observed).values()), dtype=np.float64)
        total = counts.sum()
        purities.append(counts.max() / total)
        probabilities = counts / total
        entropies.append(float(-(probabilities * np.log2(probabilities)).sum()))

    if not purities:
        return {"mean_purity": float("nan"), "mean_entropy": float("nan"), "n_points": 0}

    return {
        "mean_purity": float(np.mean(purities)),
        "mean_entropy": float(np.mean(entropies)),
        "n_points": len(purities),
    }


def _sample_pairs(items, rng, max_pairs):
    """Return up to max_pairs unordered index pairs drawn from items."""
    n = len(items)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return list(itertools.combinations(range(n), 2))
    first = rng.integers(0, n, size=max_pairs * 2)
    second = rng.integers(0, n, size=max_pairs * 2)
    keep = first != second
    pairs = {(int(min(a, b)), int(max(a, b))) for a, b in zip(first[keep], second[keep])}
    return list(pairs)[:max_pairs]


def cosegmentation_agreement(
    per_frame_labels, rng, max_pairs=200_000, max_view_pairs=500
):
    """Rand-index-style agreement of point grouping between pairs of views.

    For each sampled pair of views, take the 3D points observed in both, sample
    pairs of those points, and check whether "same mask in view A" agrees with
    "same mask in view B". Background observations are dropped.
    """
    frames = sorted(per_frame_labels)
    view_pairs = list(itertools.combinations(frames, 2))
    if len(view_pairs) > max_view_pairs:
        chosen = rng.choice(len(view_pairs), size=max_view_pairs, replace=False)
        view_pairs = [view_pairs[i] for i in sorted(chosen)]

    same_hits = same_total = diff_hits = diff_total = 0
    used_view_pairs = 0
    budget_per_pair = max(1, max_pairs // max(1, len(view_pairs)))

    for frame_a, frame_b in view_pairs:
        labels_a = per_frame_labels[frame_a]
        labels_b = per_frame_labels[frame_b]
        common = [
            p
            for p in labels_a.keys() & labels_b.keys()
            if labels_a[p] != BACKGROUND and labels_b[p] != BACKGROUND
        ]
        if len(common) < 2:
            continue
        used_view_pairs += 1

        for i, j in _sample_pairs(common, rng, budget_per_pair):
            point_i, point_j = common[i], common[j]
            same_in_a = labels_a[point_i] == labels_a[point_j]
            same_in_b = labels_b[point_i] == labels_b[point_j]
            if same_in_a:
                same_total += 1
                same_hits += int(same_in_b)
            else:
                diff_total += 1
                diff_hits += int(not same_in_b)

    total = same_total + diff_total
    if total == 0:
        return {
            "agreement": float("nan"),
            "same_pair_agreement": float("nan"),
            "diff_pair_agreement": float("nan"),
            "balanced": float("nan"),
            "n_pairs": 0,
            "n_view_pairs": used_view_pairs,
        }

    same_rate = same_hits / same_total if same_total else float("nan")
    diff_rate = diff_hits / diff_total if diff_total else float("nan")
    rates = [r for r in (same_rate, diff_rate) if not np.isnan(r)]

    return {
        "agreement": (same_hits + diff_hits) / total,
        "same_pair_agreement": same_rate,
        "diff_pair_agreement": diff_rate,
        "balanced": float(np.mean(rates)),
        "n_pairs": total,
        "n_view_pairs": used_view_pairs,
    }


def stability_descriptors(masks_per_frame, areas_per_frame):
    """Descriptive statistics on how mask counts and sizes vary across frames."""
    if not masks_per_frame:
        return {
            "mean_masks_per_frame": float("nan"),
            "std_masks_per_frame": float("nan"),
            "min_masks_per_frame": 0,
            "max_masks_per_frame": 0,
            "median_mask_area": float("nan"),
            "n_frames": 0,
        }

    all_areas = [a for frame_areas in areas_per_frame for a in frame_areas]
    return {
        "mean_masks_per_frame": float(np.mean(masks_per_frame)),
        "std_masks_per_frame": float(np.std(masks_per_frame)),
        "min_masks_per_frame": int(np.min(masks_per_frame)),
        "max_masks_per_frame": int(np.max(masks_per_frame)),
        "median_mask_area": float(np.median(all_areas)) if all_areas else float("nan"),
        "n_frames": len(masks_per_frame),
    }


def track_descriptors(obj_ids_per_frame):
    """Track lifetime statistics from per-frame object id lists (video arms)."""
    lifetimes = Counter()
    for obj_ids in obj_ids_per_frame:
        for obj_id in obj_ids:
            lifetimes[obj_id] += 1

    if not lifetimes:
        return {"n_tracks": 0, "mean_lifetime": float("nan"), "max_lifetime": 0}

    values = np.array(list(lifetimes.values()), dtype=np.float64)
    return {
        "n_tracks": len(lifetimes),
        "mean_lifetime": float(values.mean()),
        "max_lifetime": int(values.max()),
    }
