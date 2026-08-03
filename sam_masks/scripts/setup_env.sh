#!/bin/bash
# Create the SAM venv. Kept entirely separate from radfoam's .venv, which is
# pinned to torch 2.3 / CUDA 12.1 by the _GLIBCXX_USE_CXX11_ABI=0 constraint in
# src/CMakeLists.txt and cannot host SAM 3 (needs Python >=3.12, torch >=2.7).
#
#   bash sam_masks/scripts/setup_env.sh
#
# Run on a GPU node, e.g.:
#   srun -p a40-lo --gres=gpu:1 --time=01:00:00 bash sam_masks/scripts/setup_env.sh
#
# The SAM backends are cloned into external/sam_backends/ rather than external/
# directly, so they stay clear of radfoam's own git submodules in
# external/submodules/ and of external/CMakeLists.txt.
#
# Environment overrides:
#   VENV_DIR    venv path relative to the repo root (default .venv-sam)
#   EXTERNAL    checkout dir for the backends (default external/sam_backends)
#   BACKENDS    which backends to install: "sam3 sam2" (default), "sam3", "sam2"

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

VENV_DIR="${VENV_DIR:-.venv-sam}"
EXTERNAL="${EXTERNAL:-$REPO/external/sam_backends}"
BACKENDS="${BACKENDS:-sam3 sam2}"

if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
    module load cuda/12.8
fi

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv not on PATH" >&2; exit 1; }

[ -d "$VENV_DIR" ] || uv venv --python 3.12 "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
# numpy is held below 2 by sam3's own pin (sam3/pyproject.toml: "numpy>=1.26,<2").
uv pip install 'numpy<2' pillow pycocotools pycolmap==3.11.1 tqdm pytest huggingface_hub

# sam3 imports einops (sam3/sam/rope.py) and psutil (sam3/model/sam3_video_predictor.py)
# at runtime but lists neither in its core [project.dependencies] -- einops only
# appears under the "notebooks" extra and psutil nowhere.
# cv2 is needed by sam2's mask post-processing (sam2/utils/amg.py). opencv 5.x
# requires numpy>=2, which conflicts with sam3's pin, so stay on 4.x.
uv pip install einops psutil 'opencv-python-headless<5'

mkdir -p "$EXTERNAL"

for backend in $BACKENDS; do
    case "$backend" in
        sam3) url=https://github.com/facebookresearch/sam3.git ;;
        sam2) url=https://github.com/facebookresearch/sam2.git ;;
        *) echo "unknown backend: $backend" >&2; exit 1 ;;
    esac
    [ -d "$EXTERNAL/$backend" ] || git clone "$url" "$EXTERNAL/$backend"
    # --no-build-isolation keeps the editable install from pulling a second
    # torch from PyPI (the default CPU index) while building.
    uv pip install --no-build-isolation -e "$EXTERNAL/$backend"
done

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
echo "=== done; activate with: source $REPO/$VENV_DIR/bin/activate ==="
