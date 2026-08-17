#!/bin/bash
#SBATCH --job-name=radfoam-removal
#SBATCH --partition=3090-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=radfoam-removal-%j.log
#
# Remove instances from a finished run and render before/after panels.
#   sbatch scripts/render_removal_slurm.sh output/teatime_var05_geo
set -euo pipefail
CHECKPOINT="${1:?usage: render_removal_slurm.sh <output/experiment>}"
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
echo "=== $(hostname): instance removal on $CHECKPOINT ==="
python scripts/render_removal.py --checkpoint "$CHECKPOINT" ${REMOVAL_ARGS:-}
echo "=== done ==="
