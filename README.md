# Instance segmentation for Radiant Foam

Open-vocabulary 3D instance segmentation in a Voronoi radiance field. Per-cell
instance embeddings are learned alongside radiance in
[Radiant Foam](https://github.com/theialab/radfoam), clustered into objects, and
queried with text — the [OpenSplat3D](https://arxiv.org/abs/2506.07697) recipe
applied to a space-tiling representation instead of Gaussian splats.

<p align="center">
  <img src="assets/teatime_instances/frame_0025.jpg" width="96%">
  <br><sub>RGB, instance overlay, argmax over per-cell identity — LERF teatime.
  Videos: <a href="assets/teatime_instances/instances_model_020000.mp4">teatime</a>,
  <a href="assets/snpp_instances/instances_model_016000.mp4">ScanNet++</a></sub>
</p>

## What this does

SAM masks are precomputed for every training view at three granularity levels. A
16-dimensional embedding on each Voronoi cell is trained by a contrastive loss
over those masks, composited along rays by the same tracer that renders colour.
Clusters of cells become objects; each gets a language embedding from multi-scale
crops of the views that see it best.

Two additions to the OpenSplat3D recipe, both mine:

- **The instance gradient also shapes geometry** — it moves site positions and
  densities, not just features. Worth [+7.4 mIoU](#geometry-guided-gradients).
- **A variance loss** accumulates a second moment per ray, penalising rays whose
  cells disagree. Needs a custom CUDA backward, derived analytically and checked
  against `torch.autograd.gradcheck` (`src/tracing/pipeline.cu`,
  `scripts/gradcheck_variance.py`). The derivation lives in those files.

Also mine: the Delaunay-graph multicut, and the ScanNet++ evaluation.

## Results

Every number below is regenerated from `results/` by
`scripts/summarize_results.py`; the figures by `scripts/make_figures.py`.

### ScanNet++ 3D instance segmentation

Class-agnostic, scored on mesh points by ScanNet++'s
[official evaluator](https://github.com/scannetpp/scannetpp).

| method | scenes | AP | AP50 | AP25 |
|---|---|---|---|---|
| SAM3D | 50 | 3.9 | 9.3 | 22.1 |
| Segment3D | 50 | 13.0 | 23.8 | 38.3 |
| OpenSplat3D | 50 | 19.2 | 37.3 | 56.2 |
| OpenSplat3D + DBSCAN denoising | 50 | 24.5 | 41.7 | 57.1 |
| this repo, HDBSCAN `min_cluster_size=512` | 8 | 23.9 | 49.4 | 67.6 |

> **The baseline rows are reference values, not a comparison.** They are 50-scene
> means from the OpenSplat3D paper; this row is 8 of those scenes, where
> per-scene AP spans 13.5–30.7. No baseline publishes per-scene results, so the
> rows cannot be reconciled. Read them for order of magnitude only.

### LERF-Mask

Grounded protocol, mean over figurines / ramen / teatime — the same three scenes
the baselines use.

| method | mIoU | mBIoU |
|---|---|---|
| Gaussian Grouping | 72.8 | 67.6 |
| ILGS (ICCV 2025) | 80.5 | 76.0 |
| this repo | 82.7 | 77.7 |
| OpenSplat3D | 84.0 | — |

### Geometry-guided gradients

Both arms retrained from scratch, paired per scene.

| scene | with | without | Δ mIoU | Δ mBIoU |
|---|---|---|---|---|
| figurines | 91.15 | 89.99 | +1.16 | +1.59 |
| ramen | 75.74 | 65.54 | +10.20 | +20.06 |
| teatime | 81.16 | 70.33 | +10.83 | +12.94 |
| **mean** | **82.68** | **75.29** | **+7.39** | **+11.53** |

<p align="center"><img src="assets/figures/guided_geometry.png" width="62%"></p>

figurines starts at 89.99 without the term, so its +1.16 is as easily a ceiling
effect as a real gain.

### HDBSCAN vs multicut

The cells carry a feature *and* sit in a Delaunay graph, so the partition can be
found either way. Best configuration of each, paired on the same scenes and
checkpoints.

| clustering | AP | AP50 | AP25 | per scene |
|---|---|---|---|---|
| HDBSCAN, `min_cluster_size=512` | 23.9 | 49.4 | 67.6 | 368 s |
| multicut, τ=0.3, `min_size=512` | 22.4 | 44.8 | 62.6 | 23 s |

The graph cut does not win, though it is 16× cheaper. Both methods leave cells
unlabelled — HDBSCAN abstains on ~70% — and the numbers above fill those by
nearest centroid in feature space. That fill is worth +6.8 AP to HDBSCAN and
+4.0 to multicut; without it, multicut leads on 7 of 8 scenes. The graph encodes
real structure, and a feature-space fill encodes more of it for less.

### Scene editing

<p align="center">
  <img src="assets/teatime_removal/remove_400_frame0040.jpg" width="49%">
  <img src="assets/teatime_removal/remove_416_frame0040.jpg" width="49%">
</p>

Instances can be removed and the scene re-rendered. Objects are opaque shells
over empty space, so deletion exposes a hole rather than interior geometry.

## Install

Build the upstream [Radiant Foam](https://github.com/theialab/radfoam) CUDA
extension, then:

```bash
git clone --recursive <this repo>
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt && pip install -e .
```

CUDA 12.1, torch 2.3.0, Python 3.10, one 24 GB GPU. **The cuML 24.10 pin
matters** — later versions build the HDBSCAN kNN graph differently and can return
a different partition. `--encoder masqclip` additionally needs OpenAI CLIP
(`pip install git+https://github.com/openai/CLIP.git`) and weights at
`ckpts/MasQCLIP/base_novel.pth`.

## Data

No dataset is redistributed. Roots resolve through `radfoam_model/data_paths.py`:
`$RADFOAM_<NAME>` if set, else `data/<name>`. Symlink them:

```bash
ln -s /path/to/lerf_mask  data/lerf_mask   # Gaussian Grouping's annotated LERF scenes
ln -s /path/to/lerf_ovs   data/lerf_ovs    # LangSplat's LERF-OVS labels
ln -s /path/to/scannetpp  data/scannetpp   # the release's data/ directory
```

Training images come from `data_path` in the YAML config, so run from the repo
root. ScanNet++ evaluation also reads `metadata/` and `splits/` as siblings of
`data/`.

## Reproducing

```bash
# 1. SAM masks. Own environment: SAM needs python >=3.12 / torch >=2.7, this
#    repo is pinned to torch 2.3 by _GLIBCXX_USE_CXX11_ABI=0 in src/CMakeLists.
bash sam_masks/scripts/setup_env.sh
python -m sam_masks.run_image --scene teatime --model sam21_levels --tag t70

# 2. Train
python train.py -c configs/lerf_mask.yaml --scene teatime \
    --instance_guided_geometry --instance_weight 0.1 --variance_weight 0.5

# 3. Cluster once; every consumer reads this cache, so instance ids agree
python scripts/cluster_cells.py --checkpoint output/<run> --method full

# 4. Evaluate
python scripts/eval_scannetpp.py --checkpoint output/<run> --model model_020000.pt \
    --clustering hdbscan --min-cluster-size 512 --fill-noise --split-connected
python scripts/eval_lerf_grounded.py --checkpoint output/<run> \
    --model model_020000.pt --clustering hdbscan_full
```

`t70` names a threshold set, not just a directory (32×32 grid, `pred_iou_thresh`
0.70, `stability_thresh` 0.88); training aborts if the masks are missing.
`--clustering hdbscan_full` is required for the LERF-Mask numbers — the default
refits on a 60k subsample and scores ~4 mIoU lower. `eval_lerf_grounded.py`
pulls GroundingDINO and SAM-ViT-H from the Hub, so warm the cache before
submitting to offline nodes. Slurm wrappers in `scripts/*_slurm.sh` are a record
of how the reported runs were launched, not a portable recipe.

The 8 scenes are the first of `nvs_sem_val.txt` in file order, capped at 300
frames each; 311 of their 691 annotated instances survive the 83-class benchmark
restriction and the 100-vertex minimum.

## Code map

Most of this tree is upstream Radiant Foam. Mine:

| path | |
|---|---|
| `radfoam_model/instance_loss.py` | multi-level contrastive loss over SAM masks |
| `radfoam_model/instance_cluster.py` | HDBSCAN over all cells (cuML), cached |
| `radfoam_model/instance_graph.py` | multicut/GAEC on the Delaunay graph |
| `radfoam_model/instance_language.py` | crop pipeline, SigLIP / MasQCLIP encoders |
| `radfoam_model/scannetpp_eval.py` | 3D instance AP against the scanned mesh |
| `src/tracing/pipeline.cu` | feature + second-moment accumulation, analytic backward |
| `sam_masks/` | SAM 2.1 / 3.1 mask precompute |
| `scripts/` | training, clustering, evaluation, figures |
| `results/` | raw eval output behind every table above |

## Limitations

- **Tie order.** Uniform prediction confidences leave the precision-recall
  ranking to arbitrary cluster order. Permuting it 100× per scene moves AP by
  sd 1.73 and AP50/AP25 by ~3.2; the reported order flatters the mean by +0.84
  AP, +1.74 AP50, +3.50 AP25. Differences under ~2 AP anywhere above are noise,
  including the HDBSCAN–multicut gap.
- **Two scorers.** `scannetpp_eval.py` reimplements ScanNet++'s scorer and reads
  ~6 AP low, by a margin varying −0.5 to +10.7 across scenes. Reported numbers
  use the official evaluator; the reimplementation appears in no table.
- **Comparisons.** Those within this repo are paired on identical scenes and
  checkpoints and are sound. Those against published means are not.
- **LERF-OVS**: 66.1 mIoU with SigLIP, 63.3 with MasQCLIP, against 59.7 for
  OpenSplat3D. Single runs on a metric that moved several mIoU between repeats,
  checkpoint lost, and its annotated frames are training views. Nothing rests
  on it.
- **LERF-Mask** has no external evaluator to check against, unlike ScanNet++.
- **The occupancy prior did not work.** Binarising opacity with a
  total-variation term commits cells reliably, but most of the commitment is
  deletion. Kept in `occupancy_loss.py` as a negative result.

<p align="center">
  <img src="assets/figures/official_vs_ours.png" width="49%">
  <img src="assets/figures/tie_order.png" width="49%">
</p>

## Attribution

Fork of [Radiant Foam](https://github.com/theialab/radfoam) (Apache 2.0) — the
renderer, tracer, Delaunay machinery and build system are theirs, as are
`test.py`, `benchmark.py` and `viewer.py`. `third_party/masqclip.py` is vendored
from [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d),
which adapts [MasQCLIP](https://github.com/mlpc-ucsd/MasQCLIP). Method and
evaluation protocols follow OpenSplat3D.
