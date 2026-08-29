#!/bin/bash
#SBATCH -J snpp-export -p 3090-lo --gres=gpu:1 -c 8 --mem=48G -t 02:00:00 -o snpp-export-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: export_official_slurm.sh <output/experiment>}"
OUT="${2:?usage: export_official_slurm.sh <checkpoint> <outdir>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"; cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
python scripts/export_scannetpp_official.py --checkpoint "$CHECKPOINT" \
    --model "${MODEL:-model_020000.pt}" --out "$OUT" ${EXPORT_ARGS:-}
