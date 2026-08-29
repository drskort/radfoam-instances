"""Export one scene's predictions in ScanNet++'s official evaluation format.

Cross-check for `radfoam_model/scannetpp_eval.py`, which reimplements their
scorer. The README compares its output against baseline numbers produced by the
official evaluator, so a systematic offset between the two would invalidate that
comparison. This writes what the official `semantic/eval/eval_instance.py`
expects, so the same predictions can be scored both ways.

Class-agnostic is expressed by collapsing every ground-truth instance and every
prediction onto a single instance class -- the official scorer is per-class and
averages over classes, so one populated class reproduces the class-agnostic
number rather than diluting it across 84 empty ones.

    python scripts/export_scannetpp_official.py --checkpoint output/<run> \
        --model model_020000.pt --out /path/to/export
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json

import numpy as np
import torch

from configs import *  # noqa: F401,F403
from radfoam_model.checkpoint import load_model  # noqa: E402
from radfoam_model.instance_cluster import NOISE_ID  # noqa: E402
from radfoam_model.scannetpp_eval import (  # noqa: E402
    SCANNETPP_META,
    assign_by_containment,
    average_precision,
    fill_noise_labels,
    load_gt_instances,
    predictions_from_labels,
    split_connected,
)

AGNOSTIC_CLASS = "table"   # any class in top100_instance.txt


def rle_encode(mask):
    """ScanNet++'s RLE, from their common/utils/rle.py.

    Their eval_instance.py header documents plain 0/1-per-line text masks, but
    the code reads `rle_decode(load_json(...))` -- the docstring is stale and
    the format is RLE JSON. Reproduced here rather than imported so this script
    does not depend on a clone of their repo.
    """
    length = mask.shape[0]
    padded = np.concatenate([[0], mask.astype(np.uint8), [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"length": int(length), "counts": " ".join(str(int(x)) for x in runs)}


def write_mask(path, mask):
    path.write_text(json.dumps(rle_encode(mask)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="model_020000.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cluster-size", type=int, default=512)
    ap.add_argument("--min-samples", type=int, default=16)
    ap.add_argument("--min-vertices", type=int, default=100)
    args = ap.parse_args()

    # Their evaluator indexes metadata/semantic_classes.txt (2878 entries), NOT
    # the 100-class benchmark list -- a label id means a different class in each.
    # Class 0 there is "wall", which is not an instance class, so gt_id 0 lands
    # in exactly the void bucket their bool_void test expects.
    semantic = [l.strip() for l in
                (SCANNETPP_META / "semantic_classes.txt")
                .read_text().splitlines() if l.strip()]
    instance = {l.strip() for l in
                (SCANNETPP_META / "instance_classes.txt")
                .read_text().splitlines() if l.strip()}
    class_id = semantic.index(AGNOSTIC_CLASS)
    assert AGNOSTIC_CLASS in instance, f"{AGNOSTIC_CLASS} is not an instance class"
    assert semantic[0] not in instance, "class 0 must be an ignore class for void"
    print(f"class-agnostic label: {AGNOSTIC_CLASS!r} -> id {class_id}")

    device = torch.device("cuda")
    model, _, dataset_args = load_model(args.checkpoint, device, args.model)
    scene = dataset_args.scene

    from radfoam_model.instance_cluster import fit_clusters_full
    labels, clustering = fit_clusters_full(
        model.att_feat, min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples)
    labels = fill_noise_labels(labels, model.att_feat.detach(),
                               clustering.centroids).cpu().numpy()
    from radfoam_model.instance_graph import undirected_edges
    edges = undirected_edges(model.point_adjacency,
                             model.point_adjacency_offsets).cpu().numpy()
    labels = split_connected(labels, edges, min_size=args.min_vertices,
                             noise_id=NOISE_ID)

    vertices, gt_instances = load_gt_instances(scene)
    # void = every vertex outside the benchmark instances (wall, floor, and the
    # annotated objects whose class is not one of the 83). Built exactly as
    # eval_scannetpp.py does, so the comparison is against the same input.
    void = np.ones(len(vertices), dtype=bool)
    for _, idx in gt_instances:
        void[idx] = False
    vertex_labels = assign_by_containment(vertices, model, labels, device=device)
    predictions = predictions_from_labels(
        vertex_labels, noise_id=NOISE_ID, min_vertices=args.min_vertices,
        score="uniform")

    # our own scorer, on exactly these predictions
    ours = average_precision(predictions, gt_instances, len(vertices), void=void)
    print(f"ours:  AP {100*ours['AP']:.2f}  AP50 {100*ours['AP50']:.2f}  "
          f"AP25 {100*ours['AP25']:.2f}")

    out = Path(args.out)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "pred" / scene).mkdir(parents=True, exist_ok=True)

    # GT: class_id * 1000 + instance number; 0 (= "wall", an ignore class) is void
    gt_ids = np.zeros(len(vertices), dtype=np.int64)
    kept = [g for g in gt_instances if len(g[1]) >= args.min_vertices]
    for i, (_, idx) in enumerate(kept, start=1):
        gt_ids[idx] = class_id * 1000 + i
    np.savetxt(out / "gt" / f"{scene}.txt", gt_ids, fmt="%d")
    print(f"gt: {len(kept)} instances over {len(vertices):,} vertices")

    lines = []
    for i, (idx, score) in enumerate(predictions):
        mask = np.zeros(len(vertices), dtype=bool)
        mask[idx] = True
        rel = f"{scene}/{i:04d}.json"
        write_mask(out / "pred" / rel, mask)
        lines.append(f"{rel} {class_id} {score:.4f}")
    (out / "pred" / f"{scene}.txt").write_text("\n".join(lines) + "\n")
    print(f"pred: {len(predictions)} masks -> {out/'pred'}")
    print(f"\nrun the official scorer against {out}")


if __name__ == "__main__":
    main()
