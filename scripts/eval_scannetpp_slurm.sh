#!/bin/bash
#SBATCH -J snpp-inst -p 3090-lo --gres=gpu:1 -c 8 --mem=48G -t 02:00:00 -o snpp-inst-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: eval_scannetpp_slurm.sh <output/experiment>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
export PYTHONPATH="$REPO"; export HF_HUB_OFFLINE=1
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
python scripts/eval_scannetpp.py --checkpoint "$CHECKPOINT" --model "${MODEL:-model.pt}" ${SNPP_ARGS:-}
