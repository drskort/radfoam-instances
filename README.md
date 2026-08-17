# Instance segmentation for Radiant Foam

Learns a per-cell instance embedding alongside radiance in
[Radiant Foam](https://github.com/theialab/radfoam), clusters it into 3D objects,
and queries those objects with text. Follows the method of
[OpenSplat3D](https://arxiv.org/abs/2506.07697), applied to a Voronoi
tessellation instead of Gaussian splats.

<p align="center">
  <img src="docs/assets/teatime_instances/frame_0025.jpg" width="49%">
  <img src="docs/assets/snpp_instances/frame_0025.jpg" width="49%">
  <br><sub>Instances rendered by argmax over per-cell identity — LERF teatime, ScanNet++ room.
  Videos: <a href="docs/assets/teatime_instances/instances_model_020000.mp4">teatime</a>,
  <a href="docs/assets/snpp_instances/instances_model_016000.mp4">ScanNet++</a></sub>
</p>

## Results

**ScanNet++ 3D instance segmentation** (class-agnostic, scored on mesh points,
scene `7b6477cb95` at 16k iterations):

| method | AP | AP50 | AP25 |
|---|---|---|---|
| SAM3D | 3.9 | 9.3 | 22.1 |
| Segment3D | 13.0 | 23.8 | 38.3 |
| OpenSplat3D | 19.2 | 37.3 | 56.2 |
| OpenSplat3D + DBSCAN denoising | **24.5** | 41.7 | 57.1 |
| this repo, HDBSCAN `min_cluster_size=512` | 18.3 | **52.3** | **69.1** |

**LERF-Mask** (grounded protocol, mean over figurines / ramen / teatime):

| method | mIoU | mBIoU |
|---|---|---|
| Gaussian Grouping | 72.8 | 67.6 |
| ILGS (ICCV 2025) | 80.5 | 76.0 |
| this repo | 83.1 | 77.9 |
| OpenSplat3D | 84.0 | — |

**LERF-OVS** (4 scenes, flat mIoU): 66.1 with SigLIP-so400m, 63.3 with MasQCLIP,
against 59.7 for OpenSplat3D. Single runs — see notes below.

## Install

Follow the upstream [Radiant Foam](https://github.com/theialab/radfoam) build,
then:

```bash
pip install -e .
pip install cuml-cu12 transformers open-clip-torch plyfile
```

MasQCLIP weights (optional, for the `masqclip` encoder) go in
`ckpts/MasQCLIP/base_novel.pth`; they are not redistributed here.

## Usage

```bash
# 1. precompute SAM masks for a scene (resumable)
python -m sam_masks.run_image --scene teatime --model sam21_levels --tag t70

# 2. train with instance features
python train.py -c configs/lerf_mask.yaml --scene teatime \
    --instance_guided_geometry --instance_weight 0.1 --variance_weight 0.5

# 3. cluster once; every downstream consumer reads the cache
python scripts/foamviz.py cluster --checkpoint output/<run> --method full

# 4. evaluate
python scripts/eval_lerf_mask.py  --checkpoint output/<run>
python scripts/eval_lerf_ovs.py   --checkpoint output/<run> --encoder siglip
python scripts/eval_scannetpp.py  --checkpoint output/<run> \
    --clustering hdbscan --min-cluster-size 512 --fill-noise --split-connected
```

Slurm wrappers for each stage are in `scripts/*_slurm.sh`.

## Layout

| path | |
|---|---|
| `radfoam_model/instance_loss.py` | multi-level contrastive loss over SAM masks |
| `radfoam_model/variance_loss.py` | variance of composited features; CUDA backward in `src/tracing/pipeline.cu` |
| `radfoam_model/instance_cluster.py` | HDBSCAN over all cells (cuML), cached with a feature fingerprint |
| `radfoam_model/instance_graph.py` | multicut/GAEC, Felzenszwalb, threshold partitions on the Delaunay graph |
| `radfoam_model/instance_language.py` | crop pipeline and SigLIP / MasQCLIP encoders |
| `radfoam_model/occupancy_loss.py` | opacity binarisation + total-variation prior |
| `radfoam_model/scannetpp_eval.py` | 3D instance AP against the scanned mesh |
| `sam_masks/` | SAM 2.1 / 3.1 mask precompute |
| `scripts/eval_*.py` | LERF-Mask, LERF-OVS, ScanNet++ harnesses |

## Implementation notes

**Variance loss.** The renderer accumulates `V = Σ wₙ fₙ²` alongside
`F = Σ wₙ fₙ`, so `s² = V − F²` penalises rays whose cells disagree. The
backward pass is derived in [`docs/variance_backward.md`](docs/variance_backward.md)
and checked with `torch.autograd.gradcheck` (`scripts/gradcheck_variance.py`).

**Point assignment.** Mesh points are assigned to the nearest cell *that renders*
(density > 1e-3), not the nearest cell. About a third of cells never render, and
plain nearest-site lands in one of them 25% of the time.

**Connected-component split.** Instances are split into spatially connected
components using the Delaunay adjacency, which is the exact form of the DBSCAN
step OpenSplat3D applies to Gaussian positions. Worth +3.3 AP here.

## Ablations

LERF, single seed:

| change | effect |
|---|---|
| instance gradients also shaping density | +22.4 mIoU |
| variance loss (weight 0.5) | +1.5 LERF-Mask, −0.8 LERF-OVS |
| dropping SAM granularity level 1 | −7.6 |
| occupancy prior (binarisation + TV) | +0.2 mIoU, −2.1 mBIoU |

The occupancy prior is kept as a negative result. It commits cells to solid or
empty reliably (cells with α between 0.1 and 0.9 drop 4–11×), but most of the
commitment is deletion, the total-variation term does not measurably contribute,
and with sites still moving it makes incremental Delaunay updates progressively
more expensive.

## Scene editing

<p align="center">
  <img src="docs/assets/teatime_removal/remove_400_frame0040.jpg" width="49%">
  <img src="docs/assets/teatime_removal/remove_416_frame0040.jpg" width="49%">
</p>

Instances can be removed and the scene re-rendered. Objects are reconstructed as
opaque shells over empty space, so deletion exposes a hole rather than interior
geometry; inpainting is not addressed here.

## Notes

- ScanNet++ numbers are one scene. The clustering sweep that produced
  `min_cluster_size=512` is in `scripts/eval_scannetpp.py --clustering hdbscan`.
- Multicut on the Delaunay graph was tried across τ ∈ [0.1, 0.8] and lost to
  feature-space HDBSCAN on ScanNet++ (best 7.8 AP vs 18.3).
- LERF-OVS results come from single runs and moved by several mIoU between
  repeats in the cases checked, so treat small differences there with care.

## Attribution

Built on [Radiant Foam](https://github.com/theialab/radfoam) (Apache 2.0) — the
renderer, tracer and Delaunay machinery are theirs. `third_party/masqclip.py` is
vendored from [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d),
which adapts [MasQCLIP](https://github.com/mlpc-ucsd/MasQCLIP). Method and
evaluation protocols follow OpenSplat3D.
