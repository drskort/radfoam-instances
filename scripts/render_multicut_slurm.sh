#!/bin/bash
#SBATCH -J multicut-vis -p a40-lo --gres=gpu:1 -c 4 --mem=48G -t 01:30:00
#SBATCH -o multicut-vis-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: render_multicut_slurm.sh <output/experiment>}"
cd "$SLURM_SUBMIT_DIR"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
export PYTHONPATH="$SLURM_SUBMIT_DIR"
python scripts/render_multicut.py --checkpoint "$CHECKPOINT" \
    --model "${MODEL:-model.pt}" ${VIS_ARGS:-}
