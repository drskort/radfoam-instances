#!/bin/bash
#SBATCH -J vlm-ground -p a40-lo --gres=gpu:1 -c 8 --mem=64G -t 03:00:00
#SBATCH -o vlm-ground-%j.log
set -euo pipefail
CHECKPOINT="${1:?usage: vlm_ground_slurm.sh <output/experiment>}"
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
source /usr/share/modules/init/bash; module load cuda/12.1
source .venv/bin/activate
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"
# The compute node can serve a stale site-packages over NFS; fail in seconds
# rather than after the clustering and crops are already done.
python -c "import timm, transformers; print(f'timm {timm.__version__}, transformers {transformers.__version__} visible on', __import__('socket').gethostname())"

python scripts/vlm_ground.py --checkpoint "$CHECKPOINT" --model "${MODEL:-model.pt}" ${VLM_ARGS:-}
