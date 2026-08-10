#!/bin/bash
#SBATCH -J lerf-ovs -p a40-lo --gres=gpu:1 -c 8 --mem=64G -t 03:00:00 -o lerf-ovs-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: eval_lerf_ovs_slurm.sh <output/experiment>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
export PYTHONPATH="$REPO"; export HF_HUB_OFFLINE=1
# MasQCLIP resolves its checkpoint relative to the repo root.
export WORKSPACE_PATH="$REPO"
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
python scripts/eval_lerf_ovs.py --checkpoint "$CHECKPOINT" --model "${MODEL:-model.pt}" ${OVS_ARGS:-}
