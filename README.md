# Instance segmentation for Radiant Foam

Learns a per-cell instance embedding alongside radiance in
[Radiant Foam](https://github.com/theialab/radfoam), clusters it into 3D objects,
and queries those objects with text. Follows the method of
[OpenSplat3D](https://arxiv.org/abs/2506.07697), applied to a Voronoi
tessellation instead of Gaussian splats.

<p align="center">
  <img src="assets/teatime_instances/frame_0025.jpg" width="49%">
  <img src="assets/snpp_instances/frame_0025.jpg" width="49%">
  <br><sub>Instances rendered by argmax over per-cell identity — LERF teatime, ScanNet++ room.
  Videos: <a href="assets/teatime_instances/instances_model_020000.mp4">teatime</a>,
  <a href="assets/snpp_instances/instances_model_016000.mp4">ScanNet++</a></sub>
</p>

## Results

Three findings, in the order I would defend them.

**1. Letting the instance gradient shape geometry is what makes this work.**
Cells are not fixed — the same gradient that trains the per-cell features can
move sites and densities too. Turning that off costs **7.4 mIoU** on all three
LERF scenes. This is the only measurement here with both arms retrained from
scratch, paired per scene, and committed as artifacts.

Both arms trained from scratch, scored under the grounded protocol, committed
under `results/ablation/guided_geometry/`. The with-arm files are the same runs
as the LERF-Mask table below, not separate evaluations:

| scene | with | without | Δ mIoU | Δ mBIoU |
|---|---|---|---|---|
| figurines | 91.15 | 89.99 | +1.16 | +1.59 |
| ramen | 75.74 | 65.54 | +10.20 | +20.06 |
| teatime | 81.16 | 70.33 | +10.83 | +12.94 |
| **mean** | **82.68** | **75.29** | **+7.39** | **+11.53** |

An earlier single-seed run on one scene put this at +22.4; that figure does not
survive a paired three-scene measurement and has been replaced rather than kept.
Note figurines barely moves (+1.16) while ramen and teatime gain ~10. Two
readings fit: figurines contains colour-labelled duplicates (green/red apple,
green/red toy chair) that the term should help disambiguate, but it also starts
at 89.99 without the term, so a ceiling effect explains the small delta at least
as well. The data here does not separate them.

**2. On ScanNet++ the foam is competitive with OpenSplat3D, and clearly ahead
at looser IoU.** Scored by ScanNet++'s own evaluator: **23.9 / 49.4 / 67.6**
against 19.2 / 37.3 / 56.2, and against 24.5 / 41.7 / 57.1 for their
DBSCAN-denoised variant — level on AP with the denoised baseline, ahead by 7.7
AP50 and 10.5 AP25. On 8 of their 50 scenes, so read it as competitive rather
than better.

An earlier version of this README argued from the AP25/AP *ratio* that objects
were found but localised loosely. That was an artifact of this repo's own
scorer understating AP; under the official evaluator the ratio is 2.8 against
OpenSplat3D's 2.9 — no difference. The claim is withdrawn.

**3. The graph cut — the part I most wanted to work — is a negative result.**
Multicut on the Delaunay adjacency loses 1.66 AP to plain feature clustering and
wins 2 of 8 scenes, despite being 16× cheaper and the only method that stands up
without noise-filling. See [Clustering study](#clustering-study).

### Benchmarks

**LERF-Mask** (grounded protocol, mean over figurines / ramen / teatime):

| method | mIoU | mBIoU |
|---|---|---|
| Gaussian Grouping | 72.8 | 67.6 |
| ILGS (ICCV 2025) | 80.5 | 76.0 |
| this repo | 82.7 | 77.7 |
| OpenSplat3D | 84.0 | — |

Per scene: figurines 91.15 / 88.91, ramen 75.74 / 68.68, teatime 81.16 / 75.62,
committed under `results/lerf_mask/`. An earlier run of the same configuration
scored 83.13 / 77.87; its checkpoint was lost, so the numbers above are from a
retrain and are the ones backed by artifacts. The two differ by 0.45 mIoU. No
seed-to-seed variance was measured on this benchmark, so that gap is reported
rather than explained away.

**ScanNet++ 3D instance segmentation** (class-agnostic, scored on mesh points,
mean over 8 scenes at 20k iterations):

| method | AP | AP50 | AP25 |
|---|---|---|---|
| SAM3D | 3.9 | 9.3 | 22.1 |
| Segment3D | 13.0 | 23.8 | 38.3 |
| OpenSplat3D | 19.2 | 37.3 | 56.2 |
| OpenSplat3D + DBSCAN denoising | 24.5 | 41.7 | 57.1 |
| this repo, HDBSCAN `min_cluster_size=512` | **23.9** | **49.4** | **67.6** |

**This row is scored by ScanNet++'s official evaluator**, the same one the
baselines use — predictions exported with `scripts/export_scannetpp_official.py`
and scored by `semantic/eval/eval_instance.py` from
[scannetpp/scannetpp](https://github.com/scannetpp/scannetpp), per scene, with
raw output in `results/scannetpp_official.json`.

That matters, because this repo also carries its own reimplementation of that
scorer in `radfoam_model/scannetpp_eval.py`, and **the two disagree**. On
identical predictions the reimplementation reports 17.65 / 42.29 / 65.56 against
the official 23.9 / 49.4 / 67.6 — it understates AP by 6.2 on average, and the
gap is not a constant offset but ranges from −0.5 to +10.7 across the eight
scenes. The reimplementation is kept because every clustering experiment below
was run through it and it is far cheaper to iterate on, but it is a development
tool, not a scorer: **only the official numbers are comparable to published
work.**

The remaining non-comparability cuts against this row: the baselines are on all
50 scenes of the validation split, this repo on the first 8, where per-scene AP
under our own scorer spans 6.9 to 29.8 (sem ±2.9). Eight scenes cannot separate
methods a few points apart.

The margin is at looser IoU: +7.7 AP50 and +10.5 AP25 over the denoised
baseline, against a dead heat on AP. Objects are found and separated at least as
well as by the Gaussian-splat pipeline; tight-boundary agreement is where the
two are level.

The headline row uses `--fill-noise --split-connected`. Both are part of the
method, not tuning: HDBSCAN abstains on ~70% of cells, which costs nothing on a
2D readout that skips unlabelled pixels and is a guaranteed miss in 3D. Without
the fill the same model scores 10.84 AP — see [Clustering study](#clustering-study).

**LERF-OVS** (4 scenes, flat mIoU): 66.1 with SigLIP-so400m, 63.3 with MasQCLIP,
against 59.7 for OpenSplat3D. This benchmark's annotated frames are ordinary
*training* views rather than a held-out split — that is the published protocol,
so the comparison is like-for-like, but the number is not a generalisation
measurement. Single runs — see [Notes](#notes).

> LERF-OVS is **not** backed by a committed artifact — the checkpoint behind it
> was lost and it has not been regenerated. It is a single run of a metric that
> moved by several mIoU between repeats, so it is reported for completeness and
> nothing here rests on it. ScanNet++ and LERF-Mask both have their raw eval
> output committed.

## Install

Follow the upstream [Radiant Foam](https://github.com/theialab/radfoam) build
for the CUDA extension, then:

```bash
git clone --recursive <this repo>   # --recursive: six submodules under external/
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
```

Developed against CUDA 12.1, torch 2.3.0, cuML 24.10 on a single 24 GB GPU
(RTX 3090 / A40). **The cuML pin matters**: 24.10 has no `build_algo` on
HDBSCAN, so a newer release builds the kNN graph differently and can return a
different partition than the numbers reported here.

Optional, only for the `masqclip` language encoder:

```bash
pip install git+https://github.com/openai/CLIP.git   # OpenAI CLIP, not on PyPI
```

and put the MasQCLIP weights at `ckpts/MasQCLIP/base_novel.pth`, resolved
relative to the working directory (the Slurm wrapper sets `WORKSPACE_PATH` for
this; running from the repo root does the same). They come from
the [MasQCLIP release](https://github.com/mlpc-ucsd/MasQCLIP) and are not
redistributed here. `eval_lerf_ovs.py` defaults to `--encoder masqclip`,
matching OpenSplat3D; pass `--encoder siglip` to avoid the checkpoint entirely,
which pulls `google/siglip-so400m-patch14-384` through `transformers` instead
and is the setting behind the higher of the two LERF-OVS numbers.

## Datasets

No dataset is redistributed here. Each root is resolved by
`radfoam_model/data_paths.py`: an environment variable if set, else
`data/<name>`, with `RADFOAM_DATA` moving that whole directory.

One caveat: this covers the **eval-side** roots — the SAM mask store, the
ScanNet++ mesh, metadata and splits, and the LERF-OVS labels. Training images
come from `data_path` in the YAML config (`configs/*.yaml`), which is baked into
each run's `config.yaml` and re-read by anything that reloads that run, so it is
a relative path and commands are run from the repo root. The intended setup is
symlinks into `data/` (gitignored), which satisfies both:

```bash
mkdir -p data
ln -s /path/to/lerf_mask   data/lerf_mask     # or export RADFOAM_LERF_MASK=...
ln -s /path/to/lerf_ovs    data/lerf_ovs      #        RADFOAM_LERF_OVS
ln -s /path/to/scannetpp   data/scannetpp     #        RADFOAM_SCANNETPP
ln -s /path/to/sam_masks   data/sam_masks     #        RADFOAM_SAM_MASKS
```

| root | what it is | where to get it |
|---|---|---|
| `data/lerf_mask` | the 3 LERF scenes with Gaussian Grouping's mask annotations, COLMAP layout with `images_train/` | [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping) |
| `data/lerf_ovs` | LERF-OVS text queries and label polygons, under `lerf_ovs/label/` | [LangSplat](https://github.com/minghanqin/LangSplat) |
| `data/scannetpp` | the official release's `data/` directory (DSLR captures, per-scene `dslr/colmap` and `dslr/resized_images`) | [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) — registration required |
| `data/sam_masks` | written by step 1 below, not downloaded | — |

ScanNet++ evaluation additionally reads `metadata/` and `splits/` as siblings of
`data/`; override with `RADFOAM_SCANNETPP_RELEASE` if your layout differs. For
your own captures, `prepare_colmap_data.py` runs COLMAP over a directory of
images into the layout the loaders expect.

## Reproducing the numbers

Every stage has a Slurm wrapper alongside it in `scripts/*_slurm.sh`. Those
hardcode one particular cluster's partitions and module system and are kept
only as a record of how the reported runs were launched — the plain commands
below are what they run, and are what you want anywhere else.

**1. Precompute SAM masks** (resumable; the mask store is keyed by scene and tag)

SAM runs in its own virtual environment. It needs Python ≥3.12 and torch ≥2.7,
while this repo is pinned to torch 2.3 / CUDA 12.1 by the
`_GLIBCXX_USE_CXX11_ABI=0` constraint in `src/CMakeLists.txt`, so the two cannot
share an interpreter. The setup script builds it and clones the SAM checkouts:

```bash
bash sam_masks/scripts/setup_env.sh      # needs `uv` on PATH
python -m sam_masks.run_image --scene teatime --model sam21_levels --tag t70
```

`t70` names a threshold set, not just a directory: 32×32 point grid,
`pred_iou_thresh=0.70`, `stability_thresh=0.88`, chosen over the defaults
(16×16 / 0.8 / 0.95) because it reaches 0.946 mask coverage against 0.978 at a
third of the over-segmentation. `TAG_PRESETS` in `sam_masks/automask.py` applies
it, so the tag and the settings cannot drift apart; explicit flags still
override. The tag is also not free-form: training looks for the arm
`sam21_levels_image_t70`
(`DEFAULT_ARM` in `radfoam_model/instance_masks.py`), so `--model sam21_levels
--tag t70` must match. Training aborts with an explicit error if the masks are
missing rather than quietly skipping the instance loss.

**2. Train with instance features**

```bash
# LERF
python train.py -c configs/lerf_mask.yaml --scene teatime \
    --instance_guided_geometry --instance_weight 0.1 --variance_weight 0.5

# ScanNet++ (20k iterations, 300 frames per scene)
python train.py -c configs/scannetpp.yaml --scene 7b6477cb95 \
    --instance_guided_geometry --instance_weight 0.1 --variance_weight 0.5
```

`scripts/train_resumable_slurm.sh` wraps this to restart from the newest
checkpoint after preemption.

**3. Evaluate**

```bash
# ScanNet++ 3D AP -- the headline table
python scripts/eval_scannetpp.py --checkpoint output/<run> --model model_020000.pt \
    --clustering hdbscan --min-cluster-size 512 --fill-noise --split-connected

# LERF-Mask, grounded protocol -- this is the 82.7 / 77.7 row.
# --clustering hdbscan_full is required: it reads the cached per-cell labels
# and uses the argmax readout. The default (hdbscan) refits on a 60k subsample
# and reads out by centroid, which loses small objects and scores ~4 mIoU lower.
python scripts/eval_lerf_grounded.py --checkpoint output/<run> \
    --model model_020000.pt --clustering hdbscan_full

# LERF-OVS
python scripts/eval_lerf_ovs.py --checkpoint output/<run> --encoder siglip
```

Three grounding paths exist and they are not interchangeable.
`eval_lerf_grounded.py` is OpenSplat3D's protocol — GroundingDINO + SAM ground
the prompt in one reference frame, and every 3D instance projecting mostly
inside that mask is selected; no language embedding is involved, and it is what
produces the reported LERF-Mask numbers. `eval_lerf_mask.py` instead queries the
per-instance SigLIP embeddings directly. `vlm_ground.py` is an exploratory
Florence-2 captioning variant, tuned against failures on these same scenes, and
is **not** used for any reported number.

The LERF harnesses read a cached clustering; produce it once with
`python scripts/cluster_cells.py --checkpoint output/<run> --method full`.
`scripts/eval_scannetpp.py` refits instead, so `--clustering` controls it directly.

`eval_lerf_grounded.py` pulls `IDEA-Research/grounding-dino-base` and
`facebook/sam-vit-huge` from the Hub on first use. Compute nodes are often
offline, so warm the cache from a machine with network access before submitting.

**4. Check the tables.** The eval scripts write their JSON next to the
checkpoint in `output/<run>/`; the copies under `results/` are those same files
renamed to `<benchmark>/<scene>.json`, byte-identical apart from the path. Every
ScanNet++, LERF-Mask and ablation number in this README is committed as the raw
output of the command that produced it, under `results/scannetpp/<scene>/`
(80 runs: 10 configurations × 8 scenes). Regenerate the tables from those files
rather than trusting the markdown:

```bash
python scripts/summarize_results.py
```

### ScanNet++ subset

The first 8 scenes of the official `nvs_sem_val.txt` split, in file order —
a prefix rather than a sample, so the choice involves no selection. Frames are
capped at 300 with a uniform stride, matching `num_frames: 300` in
OpenSplat3D's `configs/scannetpp.yaml`, implemented here as `MAX_FRAMES` in
`data_loader/scannetpp.py`. DSLR captures only. Per-scene AP is
for the reported configuration.

| scene | frames used | GT scored | AP | AP50 | AP25 |
|---|---|---|---|---|---|
| `7b6477cb95` | 300 | 48 | 15.0 | 42.2 | 67.3 |
| `c50d2d1d42` | 300 | 39 | 22.7 | 59.9 | 82.7 |
| `cc5237fd77` | 300 | 31 | 6.9 | 21.6 | 45.2 |
| `acd95847c5` | 300 | 55 | 21.5 | 48.0 | 70.7 |
| `fb5a96b1a2` | 300 | 56 | 9.4 | 38.5 | 73.8 |
| `a24f64f7fb` | 300 | 18 | 29.8 | 49.9 | 64.7 |
| `1ada7a0617` | 300 | 31 | 25.1 | 50.2 | 71.5 |
| `5eb31827b7` | 151 | 33 | 10.7 | 28.0 | 48.7 |

The `frames used` column is a property of the loader (`MAX_FRAMES` capped
against the frames on disk), not of the eval, so it is the one column here that
`summarize_results.py` does not regenerate; every other figure in this table
comes from the committed JSONs. The scenes carry 691 annotated instances
between them; **311** survive the
restriction to the 83 instance-benchmark classes and the 100-vertex minimum that
ScanNet++'s own scorer applies, and those are what the AP columns are computed
against. The counts above are the `n_gt` field of the committed JSONs.
Regenerate the scene list with
`python -c "from data_loader.scannetpp import val_scenes; print(val_scenes()[:8])"`.

## Layout

| path | |
|---|---|
| `radfoam_model/instance_loss.py` | multi-level contrastive loss over SAM masks |
| `radfoam_model/instance_cluster.py` | HDBSCAN over all cells (cuML), cached with a feature fingerprint |
| `radfoam_model/instance_graph.py` | multicut/GAEC, Felzenszwalb, threshold partitions on the Delaunay graph |
| `radfoam_model/instance_language.py` | crop pipeline and SigLIP / MasQCLIP encoders |
| `radfoam_model/occupancy_loss.py` | opacity binarisation + total-variation prior |
| `radfoam_model/scannetpp_eval.py` | 3D instance AP against the scanned mesh |
| `sam_masks/` | SAM 2.1 / 3.1 mask precompute |
| `radfoam_model/data_paths.py` | dataset root resolution (env var, then `data/`) |
| `scripts/eval_*.py` | LERF-Mask, LERF-OVS, ScanNet++ harnesses |
| `scripts/eval_lerf_grounded.py` | OpenSplat3D's LERF-Mask protocol (GroundingDINO + SAM) |
| `scripts/cluster_cells.py` | fits and caches the per-cell clustering every consumer reads |
| `scripts/summarize_results.py` | rebuilds the reported tables from `results/` |
| `results/scannetpp/`, `results/lerf_mask/` | raw eval output backing the reported numbers |

## Implementation notes

### Variance loss and its backward pass

The renderer already composites one quantity per ray. For a ray crossing cells
$n = 1 \dots N$ with transmittance $T_n$ and opacity $\alpha_n$, write
$w_n = T_n \alpha_n$. The instance features give

$$F = \sum_n w_n f_n, \qquad V = \sum_n w_n f_n^2, \qquad s^2 = V - F^2$$

so a second accumulator $V$ turns into a per-ray variance that penalises cells
disagreeing along the same ray. $V$ is the raw second moment; the subtraction
happens outside the kernel, which matters below.

The paper writes the loss as $\lVert \operatorname{Var}(F) \rVert_2^2$ per
ray. $\operatorname{Var}(F)$ is a $D$-vector whose $d$-th entry is the scalar
variance of channel $d$, so there is no cross term and
$\partial s^2_d / \partial f_{n,e} = 0$ for $e \neq d$. This implementation
averages over channels rather than summing,

$$\mathcal{L} = \frac{1}{RD} \sum_{r,d} \bigl(s^2_{r,d}\bigr)^2$$

which differs from the paper by a constant $D$ that `variance_weight` absorbs.
Channel-diagonality is what lets the *feature* gradient avoid any dot product
across the feature axis; the density gradient under `instance_guided_geometry`
necessarily sums over channels (`pipeline.cu:351`).

Backward, two upstream gradients arrive per ray: $g_F$ from the contrastive
term and $g_V = \partial \mathcal{L}/\partial V = 2 s^2 / (RD)$, the second
because $\partial s^2/\partial V = 1$. The existing backward is a linear
operator — given gradient $g$ on $\sum_n w_n x_n$ it returns $w_n g$ — so it is
applied twice, once with $x_n = f_n$ and once with $x_n = f_n^2$. With the
weights held fixed, $\partial F/\partial f_n = w_n$ and
$\partial V/\partial f_n = 2 w_n f_n$, giving

$$\frac{\partial \mathcal{L}}{\partial f_n} = w_n g_F + 2 w_n f_n g_V$$

**The subtlety is who applies the coupling.** $F$ reaches the loss twice —
directly, and through $s^2 = V - F^2$ with $\partial s^2/\partial F = -2F$.
Exactly one of autograd and the kernel may account for it. Here
`s² = V − F²` is a plain tensor op in the live graph, so the `grad_feature`
handed to the kernel *already* contains the $-2F g_V$ path and the kernel
applies no correction. OpenSplat3D wraps the same subtraction in
`torch.no_grad()`, which is why they subtract it by hand instead. The two
conventions are individually correct and cannot be mixed.

An earlier version of this kernel did mix them, using
$\hat g_F = g_F - 2F g_V$ under our convention. That double-counts the coupling
and flips the gradient sign on exactly the cells that dominate a ray. It passed
a finite-difference check at 0.1 feature scale — 1 failure in 32, dismissable as
round-off — and failed 22 of 32 at unit scale. `scripts/gradcheck_variance.py`
runs at unit scale for that reason.

Composing the two terms when $g_F$ carries only the $s^2$ path gives
$\partial s^2/\partial f_n = 2 w_n (f_n - F)$: each cell is pulled toward the
ray's own mean feature, scaled by how much it actually contributed. A cell the
ray barely touches barely moves; one that dominates the ray and disagrees gets
pushed hard.

The kernel is in `src/tracing/pipeline.cu`.

**Point assignment.** Mesh points are assigned to the nearest cell *that renders*
(density > 1e-3), not the nearest cell. About a third of cells never render, and
plain nearest-site lands in one of them 25% of the time.

**Connected-component split.** Instances are split into spatially connected
components using the Delaunay adjacency, which is the exact form of the DBSCAN
step OpenSplat3D applies to Gaussian positions.

## Ablations

LERF, single seed:

| change | effect |
|---|---|
| variance loss (weight 0.5) | +1.5 mIoU |
| dropping SAM granularity level 1 | −7.6 mIoU |
| occupancy prior (binarisation + TV) | +0.2 mIoU, −2.1 mBIoU |

The occupancy prior is kept as a negative result. It commits cells to solid or
empty reliably (cells with α between 0.1 and 0.9 drop 4–11× — measured once,
not committed), but most of the
commitment is deletion, the total-variation term does not measurably contribute,
and with sites still moving it makes incremental Delaunay updates progressively
more expensive.

> **Not backed by committed artifacts** — single seed, and the checkpoints
> behind them were lost. An earlier version of this table also quoted a −0.8
> LERF-OVS delta for the variance loss; that is an order of magnitude inside
> that benchmark's repeat-to-repeat spread and has been removed rather than
> defended.

## Clustering study

Cells carry a feature *and* sit in a Delaunay graph, so the partition can be
found by clustering the features or by cutting the graph. Both were swept over
the same 8 scenes and the same checkpoints, so every comparison below is paired
and per-scene difficulty cancels. Ten configurations were run; the eight with
`--fill-noise` are below, the two without are further down.

| clustering | AP | AP50 | AP25 |
|---|---|---|---|
| HDBSCAN `m=512` | **17.65** | 42.29 | 65.56 |
| HDBSCAN `m=256` | 17.45 | 42.40 | 64.81 |
| HDBSCAN `m=1024` | 17.42 | 42.44 | 63.33 |
| multicut τ=0.3 `m=512` | 15.99 | 37.87 | 59.48 |
| multicut + SAM votes `m=1024`, `w=1.0` | 15.98 | 36.71 | 59.68 |
| multicut τ=0.3 `m=1024` | 15.47 | 36.48 | 58.70 |
| multicut + SAM votes `m=1024`, `w=0.0` | 15.47 | 36.48 | 58.70 |
| multicut τ=0.3 `m=2048` | 13.75 | 32.52 | 56.81 |

The last two rows are identical on all 8 scenes and every field: `w=0.0` reduces
exactly to plain multicut at the same `min_size`, which is the control the
+0.51 AP figure below is measured against.

**Multicut loses, by 1.66 AP on 2 of 8 scenes won.** SAM co-occurrence votes on
the graph edges are worth +0.51 AP on 5 of 8 — inside the scene-to-scene spread,
so not a result. HDBSCAN is also insensitive to `min_cluster_size` (256/512/1024
span 0.23 AP), while multicut's grid spans 2.24, so the graph method is the more
tuning-sensitive of the two as well.

Two things cut the other way. Multicut is **16× cheaper** per scene as run
here: 23 s against 368 s end-to-end on a 3090. These timings were measured
once and are not committed as artifacts, unlike the AP figures above. Profiling puts essentially all of
HDBSCAN's cost in the cuML fit itself — data transfer and centroids are under
0.1 s — of which ~84 s is one-time initialisation, so a process fitting
repeatedly would see ~12× rather than 16×. And with `--fill-noise`
off, multicut **wins 8 of 8 scenes** by 2.20 AP (13.03 vs 10.84) and its clusters
need no connected-component split at all, being connected by construction.

The resolution is that filling abstaining cells by nearest centroid in feature
space supplies the same spatial coherence the graph does, and more of it: it is
worth +6.82 AP to HDBSCAN but only +2.96 to multicut. The graph encodes real
structure — it is simply the cheaper post-hoc step that encodes more.

```bash
python scripts/eval_scannetpp.py --checkpoint output/<run> --model model_020000.pt \
    --clustering multicut --tau 0.3 --min-cluster-size 512 \
    --fill-noise --split-connected      # drop --fill-noise for the no-fill arm
```

## Scene editing

<p align="center">
  <img src="assets/teatime_removal/remove_400_frame0040.jpg" width="49%">
  <img src="assets/teatime_removal/remove_416_frame0040.jpg" width="49%">
</p>

Instances can be removed and the scene re-rendered. Objects are reconstructed as
opaque shells over empty space, so deletion exposes a hole rather than interior
geometry; inpainting is not addressed here.

## Notes

- The connected-component and point-assignment gains are measured on one scene;
  the 8-scene tables apply both throughout.
- LERF-OVS results come from single runs and moved by several mIoU between
  repeats in the cases checked, so treat small differences there with care.
- `test.py` and `benchmark.py` at the root are upstream Radiant Foam's novel-view
  PSNR harnesses, kept as they are.

## Attribution

My contribution is `radfoam_model/instance_*`, `occupancy_loss.py`,
`scannetpp_eval.py`, `data_paths.py`, all of `sam_masks/`
and `scripts/`, the ScanNet++ loader in `data_loader/`, and the feature and
feature-squared accumulation with its analytic backward inside
`src/tracing/pipeline.cu`. Everything else — the renderer, tracer, Delaunay
machinery, build system and viewer — is upstream.

Built on [Radiant Foam](https://github.com/theialab/radfoam) (Apache 2.0). `third_party/masqclip.py` is
vendored from [OpenSplat3D](https://github.com/VisualComputingInstitute/opensplat3d),
which adapts [MasQCLIP](https://github.com/mlpc-ucsd/MasQCLIP). Method and
evaluation protocols follow OpenSplat3D.
