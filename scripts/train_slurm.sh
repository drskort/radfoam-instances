#!/bin/bash
#SBATCH --job-name=radfoam-train
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=radfoam-train-%j.log
#
# Train a RadFoam scene on an A40 using the venv built by build_slurm.sh.
#
#   sbatch scripts/train_slurm.sh garden          # defaults below
#   sbatch scripts/train_slurm.sh bonsai
#   sbatch -p a40-hi scripts/train_slurm.sh stump # any #SBATCH line can be overridden
#   FINAL_POINTS=2097152 sbatch -p 3090-lo scripts/train_slurm.sh bicycle
#
# Submit from the repository root. Runs standalone too (e.g. under srun).
#
# The venv from build_slurm.sh is compiled for CUDA_ARCHS=86, which covers both
# the A40 and the 3090 -- no rebuild needed to switch between those partitions.
# The 1080ti nodes are sm_61 and would fail at load with "no kernel image is
# available"; rebuild with CUDA_ARCHS="61;86" if you need them.

set -euo pipefail

SCENE="${1:-garden}"
DATA_PATH="${DATA_PATH:-/shared/user/datasets}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
VENV_DIR="${VENV_DIR:-.venv}"

# Under sbatch the script runs from a spool copy, so BASH_SOURCE does not point
# into the repo; Slurm gives us the submission directory instead.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

if [ ! -f train.py ]; then
    echo "$REPO is not the radfoam repo root; submit from there:" >&2
    echo "    sbatch scripts/train_slurm.sh <scene>" >&2
    exit 1
fi

# The COLMAP loader resolves the scene as data_path/scene.
if [ ! -d "$DATA_PATH/$SCENE" ]; then
    echo "no scene directory at $DATA_PATH/$SCENE" >&2
    echo "available: $(ls "$DATA_PATH" 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi

case "$SCENE" in
    bicycle|garden|stump|flowers|treehill) CONFIG=configs/mipnerf360_outdoor.yaml ;;
    figurines|ramen|teatime)               CONFIG=configs/lerf_mask.yaml ;;
    *)                                     CONFIG=configs/mipnerf360_indoor.yaml ;;
esac
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

echo "=== host $(hostname), scene $SCENE, config $CONFIG ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
fi
module load "$CUDA_MODULE"

if [ ! -d "$VENV_DIR" ]; then
    echo "$REPO/$VENV_DIR missing; build it first with: sbatch scripts/build_slurm.sh" >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

# The outdoor config asks for 4.2M final points, which needs the A40's 48GB.
# On a 24GB 3090 pass FINAL_POINTS=2097152 (or lower) for the outdoor scenes.
EXTRA=()
if [ -n "${FINAL_POINTS:-}" ]; then
    EXTRA+=(--final_points "$FINAL_POINTS")
fi
# Anything else to forward to train.py, e.g.
#   TRAIN_ARGS="--instance_guided_geometry" sbatch scripts/train_slurm.sh garden
if [ -n "${TRAIN_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA+=(${TRAIN_ARGS})
fi

EXPERIMENT="${EXPERIMENT_NAME:-${SCENE}_${SLURM_JOB_ID:-local}}"

echo "=== training -> output/$EXPERIMENT ==="
python train.py -c "$CONFIG" \
    --data_path "$DATA_PATH" \
    --scene "$SCENE" \
    --experiment_name "$EXPERIMENT" \
    "${EXTRA[@]}"

echo "=== evaluating ==="
python test.py -c "output/$EXPERIMENT/config.yaml"

echo "=== done; view with: python viewer.py -c output/$EXPERIMENT/config.yaml ==="
