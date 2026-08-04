#!/bin/bash
#SBATCH --job-name=sam-masks
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=sam-masks-%j.log
#
# One job per (scene, model, mode).
#
#   sbatch sam_masks/scripts/submit.sh garden sam31 video
#   sbatch sam_masks/scripts/submit.sh room   sam21 image
#
# All four arms for the two pilot scenes:
#   for s in garden room; do for m in sam31 sam21; do for k in video image; do
#     sbatch sam_masks/scripts/submit.sh $s $m $k; done; done; done
#
# Output goes to the shared view of host's disk, resolved by sam_masks.paths.
# Do NOT hardcode /work here: that path exists separately on every compute node
# as node-local scratch, so writing to it scatters results across machines.
#
# Both runners resume. run_image skips frames that already have output;
# run_video skips an arm that already completed (propagation is one stateful
# pass and cannot resume mid-stream). Pass --force via EXTRA_ARGS to redo work.
#
# Rough per-arm cost at the 16x16 grid, measured on garden:
#   sam21 image  ~2 s/frame     sam21 video  ~1.4 s/frame
#   sam31 image  ~35 s/frame    sam31 video  ~15 s/frame
# So room/sam31/image (311 frames) is the long pole at roughly 3 hours.

set -euo pipefail

usage() {
    echo "usage: submit.sh <scene> <sam31|sam21|sam21_levels> <video|image>" >&2
    exit 2
}

SCENE="${1:-}"; MODEL="${2:-}"; MODE="${3:-}"
[ -n "$SCENE" ] && [ -n "$MODEL" ] && [ -n "$MODE" ] || usage
case "$MODEL" in sam31|sam21|sam21_levels) ;; *) echo "bad model: $MODEL" >&2; usage ;; esac
case "$MODE"  in video|image) ;; *) echo "bad mode: $MODE"  >&2; usage ;; esac

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"

if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
    module load cuda/12.8
fi

# shellcheck disable=SC1091
source "$REPO/.venv-sam/bin/activate"

# The SAM 3.1 video arm allocates and frees masks for hundreds of tracked
# objects across a long orbit, which fragments the caching allocator badly: the
# first pilot run died with 30.9 GiB genuinely allocated but 14.1 GiB reserved
# and unusable on a 47.4 GiB card. Expandable segments let the allocator return
# and reshape those blocks instead of stranding them.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "=== $(hostname): $SCENE / $MODEL / $MODE ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git -C "$REPO" rev-parse --short HEAD

python -m "sam_masks.run_${MODE}" --scene "$SCENE" --model "$MODEL" ${EXTRA_ARGS:-}

echo "=== done: $SCENE / $MODEL / $MODE ==="
