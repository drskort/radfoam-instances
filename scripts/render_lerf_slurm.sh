#!/bin/bash
#SBATCH --job-name=lerf-vis
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=lerf-vis-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: render_lerf_slurm.sh <output/experiment>}"
MODEL="${MODEL:-model.pt}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash; module load cuda/12.1
fi
source .venv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "=== $(hostname): visualising $CHECKPOINT ($MODEL) ==="
python scripts/render_lerf_results.py \
    --checkpoint "$CHECKPOINT" --model "$MODEL" ${VIS_ARGS:-}
echo "=== done ==="
