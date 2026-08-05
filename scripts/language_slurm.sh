#!/bin/bash
#SBATCH --job-name=radfoam-lang
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=radfoam-lang-%j.log
#
# Extract one language embedding per 3D instance, then run demo queries.
#   MODEL=model_006000.pt sbatch scripts/language_slurm.sh output/garden_inst_nogeo
set -euo pipefail
CHECKPOINT="${1:?usage: language_slurm.sh <output/experiment>}"
MODEL="${MODEL:-model.pt}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
    module load cuda/12.1
fi
source .venv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "=== $(hostname): embedding instances of $CHECKPOINT ($MODEL) ==="
python scripts/extract_instance_language.py \
    --checkpoint "$CHECKPOINT" --model "$MODEL" ${LANG_ARGS:-}

echo "=== open-vocabulary queries ==="
# Garden: the objects actually in the scene, plus two that are not, so a
# uniformly-high score is visible as the failure it would be.
python scripts/extract_instance_language.py \
    --checkpoint "$CHECKPOINT" --model "$MODEL" --visualise --query \
    "a wooden table" \
    "a potted plant" \
    "a football" \
    "grass" \
    "a brick wall" \
    "a red sports car" \
    "an elephant"
echo "=== done ==="
