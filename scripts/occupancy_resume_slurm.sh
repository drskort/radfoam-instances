#!/bin/bash
#SBATCH -J occ-resume -p a40-lo --gres=gpu:1 -c 8 --mem=64G -t 06:00:00 -o occ-resume-%j.log
#
# Continue a trained scene from 12k under the Potts occupancy prior, with the
# sites still free to move until freeze_points.
#
#   sbatch scripts/occupancy_resume_slurm.sh teatime
#
# Resuming from the CONTROL's checkpoint is deliberate. Both arms then share a
# bit-identical prefix, so the only difference is the prior -- which matters
# because two nominally identical ramen runs diverged by ~1 dB through
# densification chaos alone, more than any effect being measured.
#
# 12k rather than 10k because densification ends at 11k: past it the point
# count is fixed and only the mesh and the attributes move, so the resume needs
# no densification replay.
set -euo pipefail
SCENE="${1:?usage: occupancy_resume_slurm.sh <scene>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
export PYTHONPATH="$REPO"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"

BIN="${BIN:-0.01}"
TV="${TV:-0.001}"
FROM="${FROM:-12000}"
SRC="${SRC:-output/${SCENE}_occfull_control}"
NAME="${NAME:-${SCENE}_occresume}"

python train.py -c "$SRC/config.yaml" \
    --experiment_name "$NAME" \
    --resume_from "$SRC/model_$(printf %06d "$FROM").pt" \
    --start_iteration "$FROM" \
    --densify_until 0 \
    --occupancy_bin_weight "$BIN" \
    --occupancy_tv_weight "$TV" \
    --occupancy_from "$FROM" \
    ${EXTRA_ARGS:-}

echo "=== evaluating PSNR ==="
python test.py -c "output/$NAME/config.yaml"
