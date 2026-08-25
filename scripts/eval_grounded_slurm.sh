#!/bin/bash
#SBATCH --job-name=lerf-grounded
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=lerf-grounded-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: eval_grounded_slurm.sh <output/experiment>}"
MODEL="${MODEL:-model.pt}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash; module load cuda/12.1
fi
source .venv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
# cuML links against the CUDA libs shipped inside the venv.
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
echo "=== $(hostname): grounded LERF-Mask eval of $CHECKPOINT ($MODEL) ==="
# hdbscan_full reads the per-cell clustering cached by foamviz.py and reads out
# by argmax. The script's own default (hdbscan) refits on a 60k subsample and
# reads out by centroid, which scores ~4 mIoU lower -- override deliberately.
python scripts/eval_lerf_grounded.py \
    --checkpoint "$CHECKPOINT" --model "$MODEL" \
    --clustering "${CLUSTERING:-hdbscan_full}" ${GROUNDED_ARGS:-}
echo "=== done ==="
