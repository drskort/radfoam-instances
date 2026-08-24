#!/bin/bash
#SBATCH --job-name=snpp-train
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=snpp-train-%j.log
#
# Training that survives preemption.
#
# The -lo partitions are PriorityTier=1 with PreemptMode=REQUEUE, so a higher
# priority job evicts these at any moment -- a batch of eight lost ~1h40 each
# that way. Slurm requeues the job, but a plain restart begins at iteration 0
# and throws the work away, so the wrapper looks for the newest checkpoint and
# continues from it. checkpoint_every therefore bounds what a preemption can
# cost, and must not be set so high that there is nothing to come back to.
#
#   EXPERIMENT_NAME=snpp8_<scene> sbatch scripts/train_resumable_slurm.sh <scene>
set -euo pipefail

SCENE="${1:?usage: train_resumable_slurm.sh <scene>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
export PYTHONPATH="$REPO"
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"

EXPERIMENT="${EXPERIMENT_NAME:-snpp8_${SCENE}}"
CONFIG="${CONFIG_OVERRIDE:-configs/scannetpp.yaml}"
DATA="${DATA_PATH:-data/scannetpp}"
OUT="output/$EXPERIMENT"

RESUME=()
LATEST=$(ls "$OUT"/model_0*.pt 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$LATEST" ]; then
    STEP=$(basename "$LATEST" | grep -oE '[0-9]+')
    STEP=$((10#$STEP))
    echo "=== resuming $EXPERIMENT from iteration $STEP ($LATEST) ==="
    RESUME=(--resume_from "$LATEST" --start_iteration "$STEP")
else
    echo "=== starting $EXPERIMENT from scratch ==="
fi

python train.py -c "$CONFIG" \
    --data_path "$DATA" --scene "$SCENE" --experiment_name "$EXPERIMENT" \
    --instance_guided_geometry --instance_weight 0.1 --variance_weight 0.5 \
    --instance_geometry_from 2000 --checkpoint_every "${CKPT_EVERY:-2000}" \
    "${RESUME[@]}" ${TRAIN_ARGS:-}

echo "=== done: $EXPERIMENT ==="
