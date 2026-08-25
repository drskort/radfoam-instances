"""Semantics bridge between a trained Radiant Foam scene and the foamviz viewer.

    export   feat.bin -> pca.bin, cluster.bin, semantics.json
    serve    HTTP endpoints for the parts that need Python at query time:
             GroundingDINO + SAM grounding, and re-clustering.

Why the split. Per-point features are 16 floats x 4M points = 265 MB, and every
consumer of them needs either a PCA projection or a nearest-centroid search. Run
per cell during traversal on an integrated GPU that is hopeless, so it happens
once here and the browser receives a u16 instance id per point plus a small
per-cluster table. Everything the viewer does -- instance colouring, PCA view,
isolate, remove, search heatmap -- is then a lookup into that table.

Clustering deliberately calls radfoam_model.instance_cluster rather than
reimplementing HDBSCAN. That module's docstring is explicit about why: if the
viewer and the language table cluster independently, instance 7 in the viewer is
a different object from instance 7 in the language results.

Grounding reuses scripts/eval_lerf_grounded.py unmodified, so the viewer selects
instances by exactly the protocol the published numbers were computed with.

    # the subcommand used in this repo: cache a per-cell clustering that every
    # downstream consumer then reads, so ids agree between eval and viewer
    python scripts/foamviz.py cluster --checkpoint output/<run> --method full

    # export/serve target a separate WebGL viewer that is not part of this repo;
    # --scene points at that viewer's asset directory
    python scripts/foamviz.py export --scene <viewer>/public/<name> \
        --checkpoint output/<run>
"""

import argparse
import errno
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from radfoam_model.instance_cluster import (  # noqa: E402
    NOISE_ID,
    assign,
    fit_clusters,
    to_pca_rgb,
)

# u16 in cluster.bin, so noise needs a sentinel the browser can test cheaply.
NOISE_U16 = 0xFFFF

# cdist over 4M x K materialises a float per (point, cluster) pair. At 4M
# points and 57 clusters that is a 934 MB allocation, which will either OOM an
# 8 GB card or quietly push everything else out of it.
ASSIGN_CHUNK = 1 << 20


COLMAP_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}


def read_colmap_cameras(sparse_dir: Path):
    """Intrinsics and every registered pose, straight from the binary model.

    Deliberately not via pycolmap or DataHandler: this needs to run without a
    GPU, without the CUDA extension and without loading a 1.7 GB checkpoint,
    because its only job is to tell the viewer where the cameras were.

    radfoam's loader applies no recentering or rescaling, so these poses share
    the coordinate frame of scene.ply exactly.
    """
    import struct

    sparse_dir = Path(sparse_dir)
    with open(sparse_dir / "cameras.bin", "rb") as f:
        struct.unpack("<Q", f.read(8))
        _, model, width, height = struct.unpack("<iiQQ", f.read(24))
        name, count = COLMAP_MODELS[model]
        params = struct.unpack(f"<{count}d", f.read(8 * count))

    if name == "PINHOLE":
        fx, fy, cx, cy = params
    elif name == "SIMPLE_PINHOLE":
        fx, cx, cy = params
        fy = fx
    else:
        raise SystemExit(
            f"camera model {name} carries distortion, which the viewer's "
            f"pinhole unprojection does not model. Undistort first."
        )

    poses = []
    with open(sparse_dir / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            struct.unpack("<i", f.read(4))
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            struct.unpack("<i", f.read(4))
            chars = bytearray()
            while (c := f.read(1)) != b"\x00":
                chars += c
            n_points = struct.unpack("<Q", f.read(8))[0]
            f.read(n_points * 24)
            poses.append((chars.decode(), list(qvec), list(tvec)))

    poses.sort(key=lambda row: row[0])
    return {
        "intrinsics": {
            "model": name, "width": int(width), "height": int(height),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        },
        "poses": [
            {
                "name": nm,
                "qvec": [round(v, 9) for v in q],
                "tvec": [round(v, 9) for v in t],
                # radfoam holds out every 8th image, sorted by name.
                "isTest": i % 8 == 0,
            }
            for i, (nm, q, t) in enumerate(poses)
        ],
    }


def do_cameras(args):
    """Writes cameras.json into a foamviz scene folder."""
    checkpoint = Path(args.checkpoint)
    config = {}
    for line in (checkpoint / "config.yaml").read_text().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            config[key.strip()] = value.strip()

    sparse = Path(args.sparse) if args.sparse else (
        Path(config["data_path"]) / config["scene"] / "sparse" / "0"
    )
    if not sparse.exists():
        raise SystemExit(f"no COLMAP model at {sparse} (override with --sparse)")

    cameras = read_colmap_cameras(sparse)
    cameras["source"] = str(sparse)
    cameras["scene"] = config.get("scene")

    out = Path(args.scene).expanduser() / "cameras.json"
    out.write_text(json.dumps(cameras))
    intr = cameras["intrinsics"]
    print(f"wrote {out}: {len(cameras['poses'])} poses, "
          f"{intr['model']} {intr['width']}x{intr['height']} fx={intr['fx']:.1f}")


def clustering_cache_path(checkpoint: Path) -> Path:
    return Path(checkpoint) / "instances" / "clustering.pt"


def feature_fingerprint(features) -> str:
    """Cheap identity check for a feature block.

    Guards against a cache fitted on a different checkpoint or iteration. Hashes
    the shape plus a deterministic stride through the rows -- hashing 265 MB on
    every run would cost more than the fit it protects.
    """
    import hashlib

    h = hashlib.sha1()
    h.update(repr(tuple(features.shape)).encode())
    step = max(1, features.shape[0] // 4096)
    sample = features[::step][:4096]
    h.update(np.ascontiguousarray(
        sample.cpu().numpy() if hasattr(sample, "cpu") else sample
    ).tobytes())
    return h.hexdigest()


def checkpoint_features(checkpoint: Path, model_file="model.pt"):
    """att_feat straight out of the checkpoint, in its ORIGINAL order.

    Not from feat.bin: the converter may have reordered points along a Z-curve,
    and fit_clusters samples via randperm, so a permuted block would give a
    different clustering. Fitting here keeps instance ids identical to the ones
    render_instances and extract_instance_language produced.

    Reads the tensor directly rather than constructing a RadFoamScene, whose
    __init__ builds a CUDA Triangulation before loading anything -- which fails
    outright on a card the extension was not compiled for.
    """
    data = torch.load(Path(checkpoint) / model_file, map_location="cpu",
                      weights_only=False)
    features = data.get("att_feat")
    if features is None:
        raise SystemExit(
            f"{checkpoint}/{model_file} has no att_feat "
            "(run trained with feat_dim = 0)."
        )
    return features.float()


def fit_or_load_clustering(checkpoint, seed, device, refit=False,
                           model_file="model.pt", method="full",
                           min_cluster_size=None, min_samples=None):
    """The clustering for a run, from cache when it is valid.

    method="full" runs cuML HDBSCAN over EVERY primitive and caches an exact
    label per cell. method="sample" is instance_cluster.fit_clusters, which
    fits on a 60k subsample -- 1.5% of a 4M-point cloud -- and hands the rest to
    nearest-centroid. The full fit is the default because the viewer's whole
    point is selecting and isolating objects, and the sampled fit loses small
    ones outright: an object holding 0.1% of the cells contributes ~60 sampled
    points against min_cluster_size=32.

    Returns (clustering, labels_or_None, features). `labels` is per-cell and in
    the CHECKPOINT's point order.
    """
    from radfoam_model.instance_cluster import Clustering

    cache = clustering_cache_path(checkpoint)
    features = checkpoint_features(checkpoint, model_file)
    fingerprint = feature_fingerprint(features)

    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }

    if cache.exists() and not refit:
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        matches = (
            blob.get("fingerprint") == fingerprint
            and blob.get("seed") == seed
            and blob.get("method") == method
            and blob.get("params") == params
        )
        if matches:
            labels = blob.get("labels")
            print(f"clustering cached  {blob['n_clusters']} clusters, "
                  f"method={method}, "
                  f"{'exact labels' if labels is not None else 'centroids only'} "
                  f"({cache})")
            return Clustering(
                mean=blob["mean"], basis=blob["basis"], lo=blob["lo"],
                hi=blob["hi"], centroids=blob["centroids"],
                colours=blob["colours"], noise_fraction=blob["noise_fraction"],
            ), labels, features
        print(f"clustering cache stale ({cache}), refitting", flush=True)

    t0 = time.time()
    labels = None
    if method == "full":
        from radfoam_model.instance_cluster import fit_clusters_full

        kwargs = {k: v for k, v in params.items() if v is not None}
        labels, clustering = fit_clusters_full(
            features.to(device), seed=seed, **kwargs
        )
        labels = labels.cpu().to(torch.int32)
    elif method == "sample":
        clustering = fit_clusters(features.to(device), seed=seed)
    else:
        raise SystemExit(f"unknown clustering method {method!r}")

    print(f"clustering fitted  {clustering.n_clusters} clusters, "
          f"{100 * clustering.noise_fraction:.1f}% noise, method={method} "
          f"({time.time() - t0:.1f}s)", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": clustering.mean.cpu(),
        "basis": clustering.basis.cpu(),
        "lo": clustering.lo.cpu(),
        "hi": clustering.hi.cpu(),
        "centroids": clustering.centroids.cpu(),
        "colours": clustering.colours,
        "noise_fraction": clustering.noise_fraction,
        "n_clusters": clustering.n_clusters,
        # Exact per-cell labels, in the checkpoint's point order. This is the
        # expensive artefact -- cuML HDBSCAN over 4M points -- and the reason
        # the cache is worth having at all.
        "labels": labels,
        "method": method,
        "params": params,
        "seed": seed,
        "fingerprint": fingerprint,
        "model_file": model_file,
        "featDim": int(features.shape[1]),
        "pointCount": int(features.shape[0]),
    }, cache)
    print(f"cached -> {cache}  ({cache.stat().st_size / 1e6:.0f} MB)")
    return clustering, labels, features


def do_cluster(args):
    """Fit and cache a clustering, without touching any foamviz scene."""
    device = torch.device(args.device)
    clustering, labels, features = fit_or_load_clustering(
        Path(args.checkpoint), args.seed, device, args.refit, args.model,
        args.method, args.min_cluster_size, args.min_samples,
    )
    detail = ""
    if labels is not None:
        assigned = int((labels >= 0).sum())
        detail = (f", {assigned:,}/{labels.numel():,} cells assigned "
                  f"({100 * assigned / labels.numel():.1f}%)")
    print(f"{args.checkpoint}: {clustering.n_clusters} clusters over "
          f"{features.shape[0]:,} points{detail}")


def load_scene(scene_dir: Path):
    """Reads the foamviz scene manifest and its feature block."""
    manifest = json.loads((scene_dir / "manifest.json").read_text())

    if not manifest.get("hasFeat"):
        raise SystemExit(
            f"{scene_dir}/manifest.json has hasFeat=false.\n"
            "Re-run the converter with --feat, against a PLY exported by the\n"
            "patched save_ply (which emits feat_* when att_feat exists)."
        )

    n = manifest["pointCount"]
    d = manifest["featDim"]
    path = scene_dir / manifest["files"]["feat"]["name"]

    expected = n * d * 4
    actual = path.stat().st_size
    if actual != expected:
        raise SystemExit(
            f"{path} is {actual} bytes, expected {expected} "
            f"({n} points x {d} dims x 4). Stale or truncated export."
        )

    features = np.memmap(path, dtype=np.float32, mode="r").reshape(n, d)
    return manifest, features


def cluster_assignments(features_t, clustering, device):
    """Nearest-centroid id per point, in chunks so cdist cannot blow up."""
    n = features_t.shape[0]
    out = np.empty(n, dtype=np.uint16)

    for start in range(0, n, ASSIGN_CHUNK):
        block = features_t[start : start + ASSIGN_CHUNK].to(device)
        ids = assign(block, clustering).cpu().numpy()
        out[start : start + ASSIGN_CHUNK] = np.where(
            ids == NOISE_ID, NOISE_U16, ids
        ).astype(np.uint16)

    return out


def pca_colours(features_t, clustering, device):
    """PCA projection to RGBA8, chunked for the same reason."""
    n = features_t.shape[0]
    out = np.empty((n, 4), dtype=np.uint8)
    out[:, 3] = 255

    for start in range(0, n, ASSIGN_CHUNK):
        block = features_t[start : start + ASSIGN_CHUNK].to(device)
        out[start : start + ASSIGN_CHUNK, :3] = to_pca_rgb(block, clustering)

    return out


def do_export_ply(args):
    """Writes a PLY with feat_* columns straight from a checkpoint.

    Deliberately does NOT go through RadFoamScene. Its constructor calls
    random_initialize() -> radfoam.Triangulation before anything is loaded, and
    the CUDA extension here is compiled for the cluster's architecture, so it
    dies with "no kernel image is available" on a Pascal card. None of that is
    needed: model.pt already carries xyz, density, colour, adjacency and
    att_feat, which is exactly the set save_ply writes.

    The density activation is reproduced from RadFoamScene.get_primal_density
    (activation_scale * softplus(beta=10)); the checkpoint stores the raw
    parameter, not the activated value.
    """
    import torch.nn.functional as F
    from plyfile import PlyData, PlyElement

    checkpoint = Path(args.checkpoint)
    out = Path(args.out).expanduser()
    data = torch.load(checkpoint / args.model, map_location="cpu", weights_only=False)

    missing = {"xyz", "density", "color_dc", "color_sh", "adjacency",
               "adjacency_offsets"} - set(data)
    if missing:
        raise SystemExit(f"{args.model} is missing {sorted(missing)}")

    features = data.get("att_feat")
    if features is None:
        raise SystemExit(
            f"{checkpoint / args.model} has no att_feat, so the PLY would carry "
            "no features. Use a run trained with feat_dim > 0 "
            "(garden_inst_geo / garden_inst_nogeo, not garden_512538)."
        )

    # activation_scale lives in the run config, not the checkpoint.
    activation_scale = args.activation_scale
    if activation_scale is None:
        activation_scale = 1.0
        config = checkpoint / "config.yaml"
        if config.exists():
            for line in config.read_text().splitlines():
                if line.startswith("activation_scale:"):
                    activation_scale = float(line.split(":", 1)[1])
        print(f"activation_scale {activation_scale} (from {config.name})")

    points = data["xyz"].numpy()
    density = (activation_scale * F.softplus(data["density"], beta=10)).numpy()
    dc = data["color_dc"].float().numpy()
    sh = data["color_sh"].float().numpy()
    feat = features.float().numpy()
    adjacency = data["adjacency"].numpy()
    offsets = data["adjacency_offsets"].numpy()

    n = points.shape[0]
    n_sh, n_feat = sh.shape[1], feat.shape[1]
    assert offsets.shape[0] == n + 1, (
        f"expected {n + 1} adjacency offsets, got {offsets.shape[0]}"
    )

    print(f"points     {n:,}")
    print(f"sh / feat  {n_sh} / {n_feat}")
    print(f"adjacency  {adjacency.shape[0]:,}")

    dtype = [("x", np.float32), ("y", np.float32), ("z", np.float32),
             ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
             ("density", np.float32), ("adjacency_offset", np.uint32)]
    dtype += [(f"color_sh_{i}", np.float32) for i in range(n_sh)]
    dtype += [(f"feat_{i}", np.float32) for i in range(n_feat)]

    vertex = np.empty(n, dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]

    C0 = 0.28209479177387814
    for channel, name in enumerate(("red", "green", "blue")):
        vertex[name] = np.clip(255 * (0.5 + C0 * dc[:, channel]), 0, 255).astype(np.uint8)

    vertex["density"] = density[:, 0]
    vertex["adjacency_offset"] = offsets[1:].astype(np.uint32)
    for i in range(n_sh):
        vertex[f"color_sh_{i}"] = sh[:, i]
    for i in range(n_feat):
        vertex[f"feat_{i}"] = feat[:, i]

    t0 = time.time()
    PlyData([
        PlyElement.describe(vertex, "vertex"),
        PlyElement.describe(
            adjacency.astype(np.uint32).view([("adjacency", np.uint32)]), "adjacency"
        ),
    ]).write(str(out))
    print(f"wrote {out}  ({out.stat().st_size / 1e9:.2f} GB, {time.time() - t0:.0f}s)")


def do_export(args):
    device = torch.device(args.device)
    scene_dir = Path(args.scene).expanduser()
    manifest, features = load_scene(scene_dir)

    n, d = features.shape
    print(f"features   {n:,} x {d}  from {scene_dir}")

    if not args.checkpoint:
        raise SystemExit(
            "--checkpoint is required: the clustering is fitted on the "
            "checkpoint's att_feat in its original order, not on the "
            "(possibly Z-curve reordered) feat.bin."
        )

    clustering, labels, _ = fit_or_load_clustering(
        Path(args.checkpoint), args.seed, device, args.refit, args.model,
        args.method, args.min_cluster_size, args.min_samples,
    )

    features_t = torch.from_numpy(np.ascontiguousarray(features))

    if clustering.n_clusters >= NOISE_U16:
        raise SystemExit(
            f"{clustering.n_clusters} clusters does not fit in u16 alongside "
            f"the noise sentinel ({NOISE_U16})."
        )

    if labels is not None:
        # Exact labels exist, but they are in the checkpoint's point order and
        # the converter may have renumbered along a Z-curve. order.bin carries
        # that permutation: order[newIndex] = oldIndex.
        order_file = manifest.get("files", {}).get("order")
        if order_file:
            order = np.fromfile(scene_dir / order_file["name"], dtype=np.uint32)
            if order.shape[0] != n:
                raise SystemExit(
                    f"order.bin has {order.shape[0]} entries, expected {n}"
                )
            permuted = labels.numpy()[order]
            print(f"labels     exact, permuted through {order_file['name']}")
        else:
            permuted = labels.numpy()
            print("labels     exact, scene is in file order")

        if permuted.shape[0] != n:
            raise SystemExit(
                f"clustering has {permuted.shape[0]} labels but the scene has "
                f"{n} points -- cache and scene are from different runs."
            )
        ids = np.where(permuted < 0, NOISE_U16, permuted).astype(np.uint16)
    else:
        print("labels     nearest-centroid (sampled clustering)")
        ids = cluster_assignments(features_t, clustering, device)
    rgba = pca_colours(features_t, clustering, device)

    (scene_dir / "cluster.bin").write_bytes(ids.tobytes())
    (scene_dir / "pca.bin").write_bytes(rgba.tobytes())

    assigned = ids[ids != NOISE_U16]
    sizes = np.bincount(assigned, minlength=clustering.n_clusters).tolist()

    semantics = {
        "pointCount": int(n),
        "featDim": int(d),
        "nClusters": int(clustering.n_clusters),
        "noiseSentinel": NOISE_U16,
        # Fraction of the HDBSCAN *sample* left unassigned, not of all points.
        "clusteringMethod": args.method,
        "noiseFraction": float(clustering.noise_fraction),
        "exactLabels": labels is not None,
        "palette": clustering.colours.tolist(),
        "clusterSizes": sizes,
        # Recorded because fit_clusters is only reproducible for a fixed seed,
        # device and feature block -- a mismatch here is exactly how the viewer
        # and the language table drift apart.
        "seed": int(args.seed),
        "device": str(device),
        "files": {
            "cluster": {"name": "cluster.bin", "type": "Uint16Array", "components": 1},
            "pca": {"name": "pca.bin", "type": "Uint8Array", "components": 4},
        },
    }

    if args.language:
        table = torch.load(args.language, map_location="cpu", weights_only=False)
        embeddings = table["embeddings"].float().numpy()

        if int(table.get("n_clusters", -1)) != clustering.n_clusters:
            # Refuse rather than silently pair embeddings with the wrong ids.
            print(
                f"WARNING: {args.language} was built against "
                f"{table.get('n_clusters')} clusters but this export has "
                f"{clustering.n_clusters}. Not attaching embeddings -- re-run "
                f"extract_instance_language.py against the same features.",
                file=sys.stderr,
            )
        else:
            (scene_dir / "language.bin").write_bytes(
                embeddings.astype(np.float32).tobytes()
            )
            semantics["language"] = {
                "vlm": table.get("vlm"),
                "dim": int(embeddings.shape[1]),
                "instanceIds": [int(i) for i in table["instance_ids"]],
                "file": {
                    "name": "language.bin",
                    "type": "Float32Array",
                    "components": int(embeddings.shape[1]),
                },
            }
            print(f"language   {embeddings.shape} from {args.language}")

    (scene_dir / "semantics.json").write_text(json.dumps(semantics, indent=2))

    # A scene folder should carry its own cameras: they differ per
    # reconstruction, and the viewer cannot render garden through figurines'
    # intrinsics (5187x3361 at fx 3845 against 986x728 at fx 778).
    try:
        do_cameras(argparse.Namespace(
            scene=str(scene_dir), checkpoint=args.checkpoint, sparse=None
        ))
    except SystemExit as error:
        print(f"note: no cameras.json written ({error})", file=sys.stderr)

    print(f"\nwrote {scene_dir}")
    print(f"  cluster.bin   {ids.nbytes / 1e6:6.1f} MB")
    print(f"  pca.bin       {rgba.nbytes / 1e6:6.1f} MB")
    print(f"  semantics.json")
    print(f"  assigned      {len(assigned):,}/{n:,} points "
          f"({100 * len(assigned) / n:.1f}%)")


class Grounder:
    """Holds the heavy state a grounding query needs, loaded on first use.

    Nothing here is touched by `export`, and the detector weights are only
    fetched once someone actually searches -- the model alone is ~1.7 GB and
    SAM ViT-H another ~2.4 GB, which will not co-reside comfortably on an 8 GB
    card with the scene.
    """

    def __init__(self, checkpoint, scene_dir, device, split, model_file):
        self.checkpoint = checkpoint
        self.scene_dir = Path(scene_dir)
        self.device = torch.device(device)
        self.split = split
        self.model_file = model_file

        self.model = None
        self.data = None
        self.labels = None
        self.clustering = None
        self.view_names = None
        self.view_index = None
        self.view_centres = None
        self.dino = self.dino_proc = self.sam = self.sam_proc = None
        self.vlm = self.vlm_proc = None
        self._id_maps = {}
        # Set when the scene cannot be loaded on this machine at all, so
        # /search can explain itself instead of retrying and failing again.
        self.scene_error = None

    def vlm_name(self):
        """The VLM the language table was built with, so the query lands in
        the same embedding space. Guessing a different checkpoint here would
        produce plausible-looking scores that mean nothing."""
        path = self.scene_dir / "semantics.json"
        if path.exists():
            language = json.loads(path.read_text()).get("language")
            if language and language.get("vlm"):
                return language["vlm"]
        return "google/siglip2-base-patch16-384"

    def ensure_vlm(self):
        if self.vlm is not None:
            return
        from transformers import AutoModel, AutoProcessor

        name = self.vlm_name()
        print(f"[grounder] loading {name} ...", flush=True)
        self.vlm_proc = AutoProcessor.from_pretrained(name)
        self.vlm = AutoModel.from_pretrained(name).to(self.device).eval()

    def embed_text(self, text):
        """Normalised SigLIP2 text embedding.

        padding='max_length' is not incidental -- SigLIP was trained with fixed
        length padding, and the default dynamic padding shifts the embedding.
        Copied from extract_instance_language.run_query so viewer scores match
        the CLI exactly.
        """
        import torch.nn.functional as F

        self.ensure_vlm()
        with torch.no_grad():
            inputs = self.vlm_proc(
                text=[text], padding="max_length", return_tensors="pt"
            ).to(self.device)
            vector = F.normalize(self.vlm.get_text_features(**inputs), dim=-1)[0]
        return vector.float().cpu().tolist()

    def ensure_scene(self):
        if self.scene_error is not None:
            raise SystemError(self.scene_error)
        if self.model is not None:
            return
        from eval_lerf_grounded import load_model
        from data_loader import DataHandler

        print(f"[grounder] loading {self.checkpoint}/{self.model_file} ...", flush=True)
        self.model, dataset_args = load_model(
            self.checkpoint, self.device, self.model_file
        )
        self.clustering = fit_clusters(self.model.att_feat)
        self.labels = assign(
            self.model.att_feat.detach().float(), self.clustering
        )

        self.data = DataHandler(dataset_args, rays_per_batch=0, device=self.device)
        self.data.reload(split=self.split, downsample=min(dataset_args.downsample))

        # DataHandler exposes image_names and c2ws; `images`/`poses` live on
        # the split dataset it wraps, not on the handler. This is the same
        # indexing eval_lerf_grounded builds for its ground-truth lookup.
        self.view_names = list(self.data.image_names)
        self.view_index = {name: i for i, name in enumerate(self.view_names)}

        # Camera centres (the translation column of each c2w), so the viewer
        # can ask to ground in whichever view is nearest to where it is flying.
        self.view_centres = self.data.c2ws[:, :, 3].cpu().numpy()
        print(
            f"[grounder] {self.clustering.n_clusters} clusters, "
            f"{len(self.view_names)} {self.split} views",
            flush=True,
        )

    def ensure_detector(self):
        if self.dino is not None:
            return
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            SamModel,
        )
        from eval_lerf_grounded import DINO_MODEL, SAM_MODEL

        print(f"[grounder] loading {DINO_MODEL} + {SAM_MODEL} ...", flush=True)
        self.dino_proc = AutoProcessor.from_pretrained(DINO_MODEL)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            DINO_MODEL
        ).to(self.device).eval()
        self.sam_proc = AutoProcessor.from_pretrained(SAM_MODEL)
        self.sam = SamModel.from_pretrained(SAM_MODEL).to(self.device).eval()

    def nearest_view(self, position):
        """View whose camera centre is closest to `position`."""
        self.ensure_scene()
        if self.view_centres is None:
            return self.view_names[0]
        d = np.linalg.norm(self.view_centres - np.asarray(position), axis=1)
        return self.view_names[int(d.argmin())]

    def id_map(self, view_name):
        """Per-pixel instance id for a view. Cached: this is 4 full renders."""
        if view_name in self._id_maps:
            return self._id_maps[view_name]

        from eval_lerf_grounded import render_argmax_labels

        self.ensure_scene()
        t0 = time.time()
        maps = render_argmax_labels(
            self.model,
            self.data,
            self.labels,
            self.clustering.n_clusters,
            [view_name],
            self.view_index,
            self.device,
        )
        self._id_maps[view_name] = maps[0]
        print(f"[grounder] id map for {view_name} in {time.time() - t0:.1f}s", flush=True)
        return self._id_maps[view_name]

    def search(self, prompt, view_name):
        """IoA of every instance against the grounded mask, unthresholded.

        eval_lerf_grounded.select_by_ioa returns only the ids that clear 0.7 and
        discards the ratio. The viewer wants the continuous value for its
        confidence heatmap, so the ratio is recomputed here and thresholding is
        left to the client. That function is untouched, so the eval protocol
        stays exactly as published.
        """
        from eval_lerf_grounded import ground_prompt
        from PIL import Image

        self.ensure_scene()
        self.ensure_detector()

        ids = self.id_map(view_name)
        height, width = ids.shape

        rgb = (
            self.data.rgbs[self.view_index[view_name]]
            .reshape(height, width, -1)[..., :3]
            .cpu()
            .numpy()
        )
        image = Image.fromarray((rgb * 255).astype(np.uint8))

        t0 = time.time()
        grounded = ground_prompt(
            image, prompt, self.dino, self.dino_proc, self.sam, self.sam_proc,
            self.device,
        )

        scores = {}
        for k in np.unique(ids):
            if k == NOISE_ID:
                continue
            instance = ids == k
            area = int(instance.sum())
            if area:
                scores[int(k)] = float(
                    np.logical_and(instance, grounded).sum() / area
                )

        return {
            "view": str(view_name),
            "prompt": prompt,
            "scores": scores,
            "groundedPixels": int(grounded.sum()),
            "elapsedMs": int(1000 * (time.time() - t0)),
        }


class Handler(BaseHTTPRequestHandler):
    grounder: Grounder = None
    scene_dir: Path = None

    def log_message(self, fmt, *a):  # quieter than the default access log
        pass

    def _send(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The viewer is served by vite on another port, so every response needs
        # to be readable cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send({})

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send({
                "ok": True,
                "scene": str(self.scene_dir),
                "sceneLoaded": self.grounder.model is not None,
                "sceneError": self.grounder.scene_error,
                "vlmLoaded": self.grounder.vlm is not None,
                "detectorLoaded": self.grounder.dino is not None,
                "textSearch": self.grounder.vlm is not None or self.grounder.model is None,
                "grounding": self.grounder.scene_error is None,
                "device": str(self.grounder.device),
            })
        elif self.path.startswith("/semantics"):
            path = self.scene_dir / "semantics.json"
            if not path.exists():
                self._send({"error": f"no semantics.json in {self.scene_dir}; "
                                     f"run `foamviz.py export` first"}, 404)
            else:
                self._send(json.loads(path.read_text()))
        else:
            self._send({"error": f"no route {self.path}"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._send({"error": f"bad JSON: {e}"}, 400)
            return

        try:
            if self.path.startswith("/embed"):
                text = body.get("text", "").strip()
                if not text:
                    self._send({"error": "missing 'text'"}, 400)
                    return
                t0 = time.time()
                embedding = self.grounder.embed_text(text)
                self._send({
                    "text": text,
                    "vlm": self.grounder.vlm_name(),
                    "dim": len(embedding),
                    "embedding": embedding,
                    "elapsedMs": int(1000 * (time.time() - t0)),
                })

            elif self.path.startswith("/search"):
                prompt = body.get("prompt", "").strip()
                if not prompt:
                    self._send({"error": "missing 'prompt'"}, 400)
                    return

                view = body.get("view")
                if view is None and body.get("position") is not None:
                    view = self.grounder.nearest_view(body["position"])
                if view is None:
                    self.grounder.ensure_scene()
                    view = self.grounder.view_names[0]

                self._send(self.grounder.search(prompt, view))

            elif self.path.startswith("/views"):
                self.grounder.ensure_scene()
                self._send({
                    "views": [str(v) for v in self.grounder.view_names],
                    "split": self.grounder.split,
                })
            else:
                self._send({"error": f"no route {self.path}"}, 404)

        except Exception as e:  # a failed query must not take the server down
            import traceback
            traceback.print_exc()
            self._send({"error": f"{type(e).__name__}: {e}"}, 500)


def do_serve(args):
    Handler.grounder = Grounder(
        args.checkpoint, args.scene, args.device, args.split, args.model
    )
    Handler.scene_dir = Path(args.scene).expanduser()

    if args.preload:
        # The VLM first: it is what /embed needs, it always works, and paying
        # its ~20 s load here rather than inside the first search is the whole
        # point of preloading.
        try:
            Handler.grounder.ensure_vlm()
        except Exception as error:
            print(f"warning: could not preload the VLM: {error}", file=sys.stderr)

        # The scene is a different matter. Constructing a RadFoamScene builds a
        # CUDA Triangulation, which dies on a card the extension was not
        # compiled for -- so a failure here must not take down a service whose
        # text search needs no scene at all.
        try:
            Handler.grounder.ensure_scene()
        except Exception as error:
            Handler.grounder.scene_error = str(error)
            print(
                f"\nwarning: the scene could not be loaded, so grounding "
                f"(/search) is unavailable:\n  {error}\n"
                f"Text search (/embed) is unaffected and still works.\n"
                f"If this is 'no kernel image is available', the radfoam CUDA "
                f"extension was built for a different GPU architecture than "
                f"this host's.\n",
                file=sys.stderr,
            )

    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        raise SystemExit(
            f"port {args.port} is already in use -- most likely an earlier "
            f"foamviz service that is still running.\n"
            f"  find it:  ss -tlnp | grep {args.port}\n"
            f"  stop it:  pkill -f 'foamviz.py serve'\n"
            f"  or pick another port:  --port {args.port + 1} "
            f"(the viewer takes ?service=http://127.0.0.1:{args.port + 1})"
        ) from None
    print(f"foamviz service on http://{args.host}:{args.port}")
    print(f"  scene      {Handler.scene_dir}")
    print(f"  checkpoint {args.checkpoint}")
    print(f"  GET  /health  /semantics")
    print(f"  POST /embed  {{text}}                    (SigLIP2, no scene needed)")
    print(f"  POST /search {{prompt, view|position}}   (GroundingDINO + SAM)")
    print(f"  POST /views")
    print("Preloading on by default; --no-preload defers to the first query.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scene", required=True,
                        help="foamviz scene directory (holds manifest.json)")
    common.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    e = sub.add_parser("export", parents=[common],
                       help="feat.bin -> pca.bin, cluster.bin, semantics.json")
    e.add_argument("--checkpoint", required=True, help="radfoam output dir")
    e.add_argument("--model", default="model.pt")
    e.add_argument("--refit", action="store_true", help="ignore the cached clustering")
    e.add_argument("--method", default="full", choices=["full", "sample"])
    e.add_argument("--min-cluster-size", type=int, default=None)
    e.add_argument("--min-samples", type=int, default=None)
    e.add_argument("--language", help="instance_language_*.pt to attach")
    e.add_argument("--seed", type=int, default=0)
    e.set_defaults(func=do_export)

    m = sub.add_parser("cameras", help="COLMAP poses -> <scene>/cameras.json")
    m.add_argument("--scene", required=True, help="foamviz scene directory")
    m.add_argument("--checkpoint", required=True, help="radfoam output dir")
    m.add_argument("--sparse", help="override the COLMAP sparse/0 path")
    m.set_defaults(func=do_cameras)

    c = sub.add_parser("cluster", help="fit + cache HDBSCAN for a run")
    c.add_argument("--checkpoint", required=True, help="radfoam output dir")
    c.add_argument("--model", default="model.pt")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--refit", action="store_true")
    c.add_argument("--method", default="full", choices=["full", "sample"],
                   help="full = cuML HDBSCAN over every cell (exact labels)")
    c.add_argument("--min-cluster-size", type=int, default=None)
    c.add_argument("--min-samples", type=int, default=None)
    c.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    c.set_defaults(func=do_cluster)

    p = sub.add_parser("export-ply", help="checkpoint -> PLY with feat_* columns")
    p.add_argument("--checkpoint", required=True, help="radfoam output dir")
    p.add_argument("--model", default="model.pt")
    p.add_argument("--out", required=True, help="destination .ply")
    p.add_argument("--activation-scale", type=float, default=None,
                   help="override; read from config.yaml when omitted")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.set_defaults(func=do_export_ply)

    s = sub.add_parser("serve", parents=[common], help="grounding + query service")
    s.add_argument("--checkpoint", required=True, help="radfoam output dir")
    s.add_argument("--model", default="model.pt")
    s.add_argument("--split", default="test", choices=["train", "test"])
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8777)
    s.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True,
                   help="load the VLM and scene at startup rather than on first "
                        "query (default: on)")
    s.set_defaults(func=do_serve)

    args = parser.parse_args()
    if args.command == "export" and not args.language and args.checkpoint:
        default = sorted(Path(args.checkpoint).glob("instance_language_*.pt"))
        if default:
            args.language = str(default[-1])
            print(f"language   using {args.language}")

    args.func(args)


if __name__ == "__main__":
    main()
