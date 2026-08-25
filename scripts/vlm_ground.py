"""Ground a text query by captioning instances, not by localising the phrase.

GroundingDINO has to answer "where is 'wavy noodles in bowl'?" and answers with
the container -- measured 15.37 IoU against a 3.35%-of-frame ground truth, and
that single failure is the largest loss in the whole pipeline. The direction is
the problem: a phrase-grounding detector must localise a compositional
expression, and "bowl" is the salient noun.

This inverts it, following TrackRef3D. The instances are ALREADY localised by
the 3D field, so no localisation is needed. Each instance is captioned once by
a VLM, and a query is answered by matching text to text. A captioning model has
no container bias because it never has to decide where anything is.

    MODEL=model_020000.pt sbatch scripts/vlm_ground_slurm.sh output/ramen_var05_geo
"""


import sys
from pathlib import Path

# Run directly from a clone: this lives in scripts/ but imports configs/,
# radfoam_model/ and data_loader/ from the repo root, which pip does not
# install (setup.cfg packages only src/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from configs import *  # noqa: F401,F403
from data_loader import DataHandler
from radfoam_model.instance_cluster import NOISE_ID, load_cached_clustering

from eval_lerf_grounded import render_argmax_labels  # noqa: E402
from eval_lerf_mask import boundary_iou, iou, load_ground_truth  # noqa: E402
from radfoam_model.checkpoint import load_model  # noqa: E402

VLM = "microsoft/Florence-2-large"
TEXT = "sentence-transformers/all-MiniLM-L6-v2"
TOP_VIEWS = 3           # captions per instance, majority is not needed at 3
EXPANSION = 0.15        # context around the crop; a bare cutout captions badly
MIN_PIXELS = 200


# How much of the crop the instance itself must occupy. Without this a
# 200-pixel fragment lying on a large object inherits that object's caption.
MIN_FILL = 0.10
# Non-instance pixels are faded rather than blacked out: a hard cutout on a
# neutral field captions poorly, while a faint context keeps the VLM oriented
# without letting the surroundings dominate the description.
FADE = 0.85


def crop_of(rgb, mask, expansion=EXPANSION):
    """Crop around the instance, fading everything that is not the instance.

    Captioning the raw bounding box was the bug in the first prototype: for a
    small fragment the box is filled by whatever it sits on, so every speck on
    the bowl was captioned "a yellow bowl on a table" and won that query.
    """
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if rows.size == 0:
        return None, 0.0
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    pr, pc = int((r1 - r0) * expansion), int((c1 - c0) * expansion)
    r0, r1 = max(0, r0 - pr), min(mask.shape[0], r1 + pr)
    c0, c1 = max(0, c0 - pc), min(mask.shape[1], c1 + pc)

    patch = rgb[r0:r1, c0:c1].astype(np.float32)
    sub = mask[r0:r1, c0:c1]
    fill = float(sub.mean())
    white = np.full_like(patch, 255.0)
    faded = patch * (1 - FADE) + white * FADE
    out = np.where(sub[..., None], patch, faded).astype(np.uint8)
    return Image.fromarray(out), fill


def caption_instances(crops_by_instance, device):
    from transformers import AutoModelForCausalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(VLM, trust_remote_code=True)
    # Florence-2's remote modeling file predates transformers 4.57 and probes
    # _supports_sdpa, which no longer exists; pinning eager attention skips
    # that path.
    vlm = AutoModelForCausalLM.from_pretrained(
        VLM, trust_remote_code=True, torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device).eval()

    task = "<CAPTION>"
    captions = {}
    for n, (inst, crops) in enumerate(sorted(crops_by_instance.items())):
        texts = []
        for crop in crops:
            inputs = proc(text=task, images=crop, return_tensors="pt")
            inputs = {k: (v.to(device).half() if v.dtype == torch.float32
                          else v.to(device)) for k, v in inputs.items()}
            with torch.no_grad():
                # Florence-2's prepare_inputs_for_generation indexes
                # past_key_values as a tuple, which transformers 4.57 replaced
                # with Cache objects; disabling the cache avoids that path.
                ids = vlm.generate(**inputs, max_new_tokens=40, num_beams=1,
                                   do_sample=False, use_cache=False)
            raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = proc.post_process_generation(
                raw, task=task, image_size=crop.size)
            texts.append(parsed[task].strip())
        captions[inst] = texts
        if n % 25 == 0:
            print(f"\r  captioned {n + 1}/{len(crops_by_instance)}",
                  end="", flush=True)
    print()
    return captions


def match(captions, prompts, device):
    """Text-to-text similarity between each query and each instance caption."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEXT)
    enc = AutoModel.from_pretrained(TEXT).to(device).eval()

    def embed(strings):
        batch = tok(strings, padding=True, truncation=True, return_tensors="pt")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = enc(**batch).last_hidden_state
        m = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=-1)

    instances = sorted(captions)
    # One instance may have several captions; keep the best-matching one.
    flat, owner = [], []
    for i in instances:
        for c in captions[i]:
            flat.append(c); owner.append(i)
    cap_emb = embed(flat)
    q_emb = embed(list(prompts))
    sim = q_emb @ cap_emb.T
    owner = torch.tensor(owner, device=device)

    scores = {}
    for qi, prompt in enumerate(prompts):
        best = {}
        for j in range(sim.shape[1]):
            k = int(owner[j])
            best[k] = max(best.get(k, -1e9), float(sim[qi, j]))
        scores[prompt] = best
    return scores, dict(zip(range(len(flat)), flat)), owner.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Instances to union per query.")
    args = parser.parse_args()

    device = torch.device("cuda")
    model, _, dataset_args = load_model(args.checkpoint, device, args.model)
    scene_dir = Path(dataset_args.data_path) / dataset_args.scene
    truth = load_ground_truth(scene_dir)

    clustering, labels = load_cached_clustering(args.checkpoint, model.att_feat)
    if clustering is None or labels is None:
        raise SystemExit("no cached clustering with per-cell labels; run "
                         "`cluster_cells.py --method full` first")
    print(f"{clustering.n_clusters} instances", flush=True)

    # Captions come from TRAINING views; the graded views stay untouched.
    train = DataHandler(dataset_args, rays_per_batch=0, device=device)
    train.reload(split="train", downsample=min(dataset_args.downsample))
    th, tw = train.img_wh[1], train.img_wh[0]
    tviews = list(train.image_names)[::max(1, len(train.image_names) // 12)]
    tindex = {n: i for i, n in enumerate(train.image_names)}
    train_maps = render_argmax_labels(model, train, labels,
                                      clustering.n_clusters, tviews, tindex,
                                      device)

    crops = {}
    for vi, view in enumerate(tviews):
        rgb = (train.rgbs[tindex[view]].reshape(th, tw, -1)[..., :3]
               .clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        idm = train_maps[vi]
        for k in np.unique(idm):
            if k == NOISE_ID:
                continue
            m = idm == k
            if m.sum() < MIN_PIXELS or len(crops.get(int(k), [])) >= TOP_VIEWS:
                continue
            c, fill = crop_of(rgb, m)
            # A fragment that fills almost none of its own crop cannot be
            # described on its own terms; skip it rather than let it inherit a
            # neighbour's caption.
            if c is not None and fill >= MIN_FILL:
                crops.setdefault(int(k), []).append(c)
    print(f"cropped {len(crops)} instances from {len(tviews)} training views",
          flush=True)

    captions = caption_instances(crops, device)
    prompts = sorted({p for v in truth.values() for p in v})
    scores, flat, owner = match(captions, prompts, device)

    # Score on the graded views.
    graded = DataHandler(dataset_args, rays_per_batch=0, device=device)
    graded.reload(split="test", downsample=min(dataset_args.downsample))
    views = sorted(truth)
    gindex = {n: i for i, n in enumerate(graded.image_names)}
    gmaps = render_argmax_labels(model, graded, labels, clustering.n_clusters,
                                 views, gindex, device)
    id_maps = {v: gmaps[i] for i, v in enumerate(views)}

    per_iou, per_biou, picked = {}, {}, {}
    for prompt in prompts:
        ranked = sorted(scores[prompt], key=lambda k: -scores[prompt][k])
        chosen = ranked[: args.top_k]
        px = {k: int(sum((id_maps[v] == k).sum() for v in views))
              for k in chosen}
        picked[prompt] = [(k, round(scores[prompt][k], 3), px[k],
                           captions[k][0][:55]) for k in chosen]
        ious, bious = [], []
        for view in views:
            if prompt not in truth[view]:
                continue
            gt = truth[view][prompt]
            pred = np.isin(id_maps[view], chosen)
            ious.append(iou(gt, pred)); bious.append(boundary_iou(gt, pred))
        per_iou[prompt] = float(np.mean(ious))
        per_biou[prompt] = float(np.mean(bious))

    print(f"\n{'prompt':<24}{'IoU':>7}{'BIoU':>7}{'px':>9}  caption")
    for p in prompts:
        cap = picked[p][0][3] if picked[p] else ""
        npx = picked[p][0][2] if picked[p] else 0
        print(f"{p:<24}{100*per_iou[p]:>7.2f}{100*per_biou[p]:>7.2f}"
              f"{npx:>9}  {cap}")
    mi, mb = np.mean(list(per_iou.values())), np.mean(list(per_biou.values()))
    print(f"{'-'*28} {'-'*7} {'-'*7}")
    print(f"{dataset_args.scene + ' Mean':<28}{100*mi:>8.2f}{100*mb:>8.2f}")

    out = Path(args.checkpoint) / f"vlm_ground_{Path(args.model).stem}.json"
    out.write_text(json.dumps(
        {"scene": dataset_args.scene, "miou": float(mi), "mbiou": float(mb),
         "per_class_iou": per_iou, "picked": picked,
         "captions": {str(k): v for k, v in captions.items()}}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
