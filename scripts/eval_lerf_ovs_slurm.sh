#!/bin/bash
#SBATCH -J lerf-ovs -p 3090-lo --gres=gpu:1 -c 8 --mem=48G -t 03:00:00 -o lerf-ovs-%j.log
#
# 3090, not A40. The LERF scenes cap at 2.1M points and MasQCLIP is a ViT-L,
# so the whole eval sits far inside 24 GB, and the venv is built for
# CUDA_ARCHS=86 which covers both cards. a40-lo is where the training runs
# queue up; sending evals there just makes them wait behind each other.
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
