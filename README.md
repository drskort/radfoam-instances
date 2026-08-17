# Instance segmentation and open-vocabulary retrieval in Radiant Foam

A fork of [Radiant Foam](https://github.com/theialab/radfoam) that learns a
per-cell **instance identity** alongside radiance, clusters it into 3D objects,
and answers text queries against them — the idea behind
[OpenSplat3D](https://arxiv.org/abs/2506.07697), moved from Gaussian splats to a
representation that *tessellates* space.

That difference is the point. Gaussians overlap and have no notion of
adjacency, so "which primitive owns this point" is answered by nearest-centre
plus a smoothing pass. A Voronoi diagram answers it by containment, and the
Delaunay graph says exactly which cells touch — so instance grouping can be
posed as a graph partition rather than as clustering in feature space.

<p align="center">
  <img src="docs/assets/teatime_instances/frame_0025.jpg" width="49%">
  <img src="docs/assets/snpp_instances/frame_0025.jpg" width="49%">
  <br><em>Learned instances, rendered by argmax over per-cell identity.
  Left: LERF teatime. Right: a ScanNet++ room.</em>
</p>

Videos: [teatime](docs/assets/teatime_instances/instances_model_020000.mp4) ·
[ScanNet++](docs/assets/snpp_instances/instances_model_016000.mp4)

---

## What is here

| | |
|---|---|
| `radfoam_model/instance_loss.py` | multi-level contrastive loss over SAM masks |
| `radfoam_model/variance_loss.py` | variance of composited features, with an analytic CUDA backward (`src/tracing/pipeline.cu`) |
| `radfoam_model/instance_cluster.py` | HDBSCAN over every cell (cuML), cached and fingerprinted |
| `radfoam_model/instance_graph.py` | multicut / GAEC, Felzenszwalb and threshold partitions on the Delaunay graph |
| `radfoam_model/instance_language.py` | OpenSplat3D's crop recipe; SigLIP and MasQCLIP encoders |
| `radfoam_model/occupancy_loss.py` | binarisation + total-variation prior on cell opacity |
| `radfoam_model/scannetpp_eval.py` | 3D instance AP against the scanned mesh |
| `sam_masks/` | SAM 2.1 / 3.1 mask precompute, resumable, one arm per (scene, model, mode) |
| `scripts/eval_*.py` | LERF-Mask, LERF-OVS and ScanNet++ harnesses |

The CUDA work is the variance loss: the renderer accumulates `V = Σ wₙ fₙ²`
alongside `F = Σ wₙ fₙ`, and the backward pass for `s² = V − F²` is derived by
hand in [`docs/variance_backward.md`](docs/variance_backward.md) and checked
with `torch.autograd.gradcheck` (`scripts/gradcheck_variance.py`). The subtlety
is deciding which gradient paths autograd already carries and which the kernel
must supply — getting that wrong produces a loss that trains and is silently
wrong.

## Results

**ScanNet++ 3D instance segmentation**, class-agnostic, scene `7b6477cb95`.
Scored on points of the aligned mesh, which is a real 3D measurement rather than
a rendered mask compared against a 2D polygon.

| method | AP | AP50 | AP25 |
|---|---|---|---|
| SAM3D | 3.9 | 9.3 | 22.1 |
| Segment3D | 13.0 | 23.8 | 38.3 |
| OpenSplat3D | 19.2 | 37.3 | 56.2 |
| OpenSplat3D + DBSCAN denoising | **24.5** | 41.7 | 57.1 |
| **this repo** (HDBSCAN, `min_cluster_size=512`) | 18.3 | **52.3** | **69.1** |

*One scene, at 16k iterations. Not a 50-scene benchmark number.*

**LERF-Mask**, grounded protocol, mIoU / mBIoU over figurines, ramen, teatime:

| | mIoU | mBIoU |
|---|---|---|
| Gaussian Grouping | 72.8 | 67.6 |
| ILGS (ICCV 2025) | 80.5 | 76.0 |
| **this repo** | **83.1** | **77.9** |
| OpenSplat3D | 84.0 | — |

## A benchmark caveat worth knowing

**LERF-OVS is not reproducible at 3–4 scenes.** Training the same recipe twice
and evaluating both:

| scene | run A | run B | Δ |
|---|---|---|---|
| ramen | 52.08 | 50.71 | −1.4 |
| teatime | 67.74 | 54.41 | **−13.3** |

The same pair differs by 0.1–0.5 on LERF-Mask, and the LERF-OVS *oracle* differs
by ~1 — so the 3D partition is stable and the **language retrieval** is not. The
two runs produced 418 vs 425 clusters, and one category (`plate`) flipped from
83.1 IoU to 0.0, moving a 14-category mean by six points on its own. The same
run at 16k and 20k iterations differs by 14.8.

A single LERF-OVS number over three or four scenes measures very little. This
may be part of why the same method appears as 9.66, 51.4, 52.8 and 57.6 across
four different papers' tables.

## Scene editing

Instances can be deleted from the field and the scene re-rendered.

<p align="center">
  <img src="docs/assets/teatime_removal/remove_400_frame0040.jpg" width="49%">
  <img src="docs/assets/teatime_removal/remove_416_frame0040.jpg" width="49%">
</p>

Honest limitation: the reconstruction represents objects as **opaque shells over
vacuum** — 15% of cells sit at density 0.000 at every percentile — so deleting an
object exposes empty space rather than what is behind it. Inpainting the hole is
not solved here.

## Running it

```bash
# 1. masks (once per scene)
python -m sam_masks.run_image --scene teatime --model sam21_levels --tag t70

# 2. train
python train.py -c configs/lerf_mask.yaml --scene teatime \
    --instance_guided_geometry --instance_weight 0.1 --variance_weight 0.5

# 3. cluster once, share the result with every consumer
python scripts/foamviz.py cluster --checkpoint output/<run> --method full

# 4. evaluate
python scripts/eval_lerf_ovs.py    --checkpoint output/<run> --encoder siglip
python scripts/eval_scannetpp.py   --checkpoint output/<run> \
    --clustering hdbscan --min-cluster-size 512 --fill-noise --split-connected
```

Slurm wrappers for each stage are in `scripts/*_slurm.sh`.

## What was measured, and what it cost

Ablations on LERF, single seed, so read them against the variance above:

| change | effect |
|---|---|
| instance gradients also shaping density | **+22.4** mIoU — ramen goes from 3 clusters to 131 |
| variance loss (0.5) | +1.5 LERF-Mask, −0.8 LERF-OVS |
| occupancy prior (binarisation + TV) | +0.2, and −2.1 mBIoU — see below |
| dropping SAM level 1 | −7.6 |

The occupancy prior is a negative result kept in the repo because the mechanism
is instructive: it commits cells to solid or empty (undecided cells fall 4–11×,
reliably), but 73% of the commitment is *deletion*, the total-variation term
contributes nothing measurable, boundaries get worse rather than better, and
with sites free to move it makes incremental Delaunay updates grow without
bound. `docs/specs/` records the design and the outcome.

## Attribution

Built on [Radiant Foam](https://github.com/theialab/radfoam) (Apache 2.0);
the renderer, tracer and Delaunay machinery are theirs. `third_party/masqclip.py`
is vendored from [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d),
which adapts [MasQCLIP](https://github.com/mlpc-ucsd/MasQCLIP); its weights are
not redistributed here. Method and protocol follow OpenSplat3D throughout —
where this repo differs from their published numbers it is noted above.
