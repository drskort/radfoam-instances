#!/bin/bash
#SBATCH -J occ-probe -p a40-lo --gres=gpu:1 -c 8 --mem=64G -t 03:00:00 -o occ-probe-%j.log
#
# a40, not 3090. These scenes train at ~2M points and the render backward
# alone needs most of a 24 GB card; the occupancy term's extra allocations
# tipped five of six probe runs into OOM at ~640 steps on a 3090. The original
# training ran on a40 for the same reason.
#
# Fine-tune a trained scene under the Potts occupancy prior.
#
#   BIN=0.01 TV=0.001 sbatch scripts/occupancy_probe_slurm.sh teatime
#
# Densification and point motion are switched off, so only densities and
# attributes move. That is deliberate: it isolates the prior from the
# triangulation adapting around it, which makes this a cleaner test than a full
# retrain and also a weaker one -- the prior can re-weight a surface but cannot
# move it.
set -euo pipefail
SCENE="${1:?usage: occupancy_probe_slurm.sh <scene>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
export PYTHONPATH="$REPO"
# The probe allocates and frees per-step edge gathers next to a 2M-point
# backward, which fragments the caching allocator; expandable segments let it
# reshape those blocks instead of stranding them.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"

BIN="${BIN:-0.01}"
TV="${TV:-0.001}"
STEPS="${STEPS:-3000}"
# freeze_points doubles as the xyz scheduler's max_steps, so 0 divides by zero
# and any value <= STEPS triggers a triangulation rebuild mid-probe. Setting it
# past the run length disables both; points_lr 0 is what actually holds the
# geometry still.
SRC="${SRC:-output/${SCENE}_var05_geo}"
NAME="${NAME:-${SCENE}_occ_b${BIN}_t${TV}}"

python train.py -c "$SRC/config.yaml" \
    --experiment_name "$NAME" \
    --resume_from "$SRC/model_020000.pt" \
    --iterations "$STEPS" \
    --densify_until 0 --freeze_points $((STEPS + 1)) \
    --points_lr_init 0 --points_lr_final 0 \
    --checkpoint_every 1000 \
    --occupancy_bin_weight "$BIN" \
    --occupancy_tv_weight "$TV" \
    --occupancy_edge_sample "${EDGE_SAMPLE:-1000000}" \
    ${EXTRA_ARGS:-}

echo "=== evaluating PSNR (gate 1) ==="
python test.py -c "output/$NAME/config.yaml"
