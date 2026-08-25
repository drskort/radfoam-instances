#!/bin/bash
#SBATCH -J cluster-cells -p 3090-lo --gres=gpu:1 -c 8 --mem=48G -t 02:00:00 -o cluster-cells-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: cluster_cells_slurm.sh <output/experiment>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
# cuML links against the CUDA libs shipped inside the venv; the full fit needs
# it, and without this it fails at import with a missing .so.
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
python scripts/cluster_cells.py --checkpoint "$CHECKPOINT" \
    --model "${MODEL:-model.pt}" --method "${METHOD:-full}" ${CLUSTER_ARGS:-}
