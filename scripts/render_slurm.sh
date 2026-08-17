#!/bin/bash
#SBATCH --job-name=radfoam-render
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=radfoam-render-%j.log
#
# Render RGB / feature-PCA / HDBSCAN panels for a finished run.
#   sbatch --dependency=afterok:<train_job> scripts/render_slurm.sh <checkpoint>
set -euo pipefail
CHECKPOINT="${1:?usage: render_slurm.sh <output/experiment>}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
    module load cuda/12.1
fi
source .venv/bin/activate
# python puts the script's dir on sys.path, not the cwd, so `configs`
# and `radfoam_model` are not importable without this.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
echo "=== $(hostname): rendering $CHECKPOINT ==="
python scripts/render_instances.py --checkpoint "$CHECKPOINT" ${RENDER_ARGS:-}
echo "=== done ==="
