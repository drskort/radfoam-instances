#!/bin/bash
#SBATCH --job-name=lerf-eval
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=lerf-eval-%j.log
#
# Score a trained scene on LERF-Mask.
#   MODEL=model_020000.pt sbatch scripts/eval_lerf_slurm.sh output/figurines_inst_nogeo
set -euo pipefail
CHECKPOINT="${1:?usage: eval_lerf_slurm.sh <output/experiment>}"
MODEL="${MODEL:-model.pt}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
    module load cuda/12.1
fi
source .venv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(hostname): LERF-Mask eval of $CHECKPOINT ($MODEL) ==="
# top_k=1 is the strict reading: one instance answers one prompt. top_k=2
# separates "the features are wrong" from "the clustering split the object".
for K in 1 2; do
    echo "--- top_k=$K ---"
    python scripts/eval_lerf_mask.py \
        --checkpoint "$CHECKPOINT" --model "$MODEL" --top-k "$K" \
        ${K:+$([ "$K" = 1 ] && echo --dump)}
done
echo "=== done ==="
