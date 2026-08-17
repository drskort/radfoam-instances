#!/bin/bash
#SBATCH --job-name=radfoam-build
#SBATCH --partition=3090-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=radfoam-build-%j.log
#
# Build RadFoam into a uv-managed venv on a GPU node.
#
#   sbatch scripts/build_slurm.sh                 # defaults below
#   sbatch -p a40-lo scripts/build_slurm.sh       # any #SBATCH line can be overridden
#   CUDA_ARCHS="61;86" sbatch scripts/build_slurm.sh
#
# Submit from the repository root. Runs standalone too (e.g. under srun, or on a
# node that already has a GPU assigned).
#
# Version pinning is not arbitrary. torch_bindings/CMakeLists.txt hard-fails unless
# the CUDA toolkit version is exactly torch.version.cuda, and src/CMakeLists.txt
# compiles the radfoam static lib with _GLIBCXX_USE_CXX11_ABI=0 -- which torch
# wheels only match up to 2.6, since 2.7+ switched to ABI=1. Keep CUDA_MODULE,
# TORCH_INDEX and TORCH_VERSION consistent if you change any of them.

set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.1}"
TORCH_VERSION="${TORCH_VERSION:-2.3.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.0}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu121}"
# Ampere (3090, A40). Add 61 for the 1080ti nodes; without a matching entry the
# resulting binary fails at load with "no kernel image is available".
CUDA_ARCHS="${CUDA_ARCHS:-86}"
VENV_DIR="${VENV_DIR:-.venv}"

# Under sbatch the script runs from a spool copy, so BASH_SOURCE does not point
# into the repo; Slurm gives us the submission directory instead.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    REPO="$SLURM_SUBMIT_DIR"
else
    REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO"

if [ ! -f setup.py ] || [ ! -f CMakeLists.txt ]; then
    echo "$REPO is not the radfoam repo root; submit from there:" >&2
    echo "    sbatch scripts/build_slurm.sh" >&2
    exit 1
fi

echo "=== host $(hostname), $(nproc) cores available ==="
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true

if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
fi
module load "$CUDA_MODULE"
# setup.py forwards CUDA_HOME to CMake as -DCUDA_TOOLKIT_ROOT_DIR.
echo "CUDA_HOME=$CUDA_HOME"
nvcc --version | tail -2

# Nothing in the CMake files sets CMAKE_CUDA_ARCHITECTURES, so without CUDAARCHS
# the build silently falls back to nvcc's sm_52 default.
export CUDAARCHS="$CUDA_ARCHS"
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export MAKEFLAGS="-j${CMAKE_BUILD_PARALLEL_LEVEL}"

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    echo "uv not found on PATH; install it from https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [ -n "${RECREATE_VENV:-}" ]; then
    rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "=== creating $VENV_DIR ==="
    uv venv --python 3.10 "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== torch $TORCH_VERSION from $TORCH_INDEX ==="
uv pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX"

echo "=== requirements ==="
uv pip install -r requirements.txt
# uv venvs ship no setuptools, and the build needs a backend without isolation.
uv pip install setuptools wheel ninja

echo "=== toolchain check ==="
python -c "import torch; print('torch', torch.__version__, '/ cuda', torch.version.cuda)"
python -c "import torch; print('_GLIBCXX_USE_CXX11_ABI =', torch._C._GLIBCXX_USE_CXX11_ABI)"
# requirements.txt pins cmake 3.29.2; system cmake is often below the 3.27 floor
# that CMakeLists.txt requires, so this must resolve inside the venv.
echo "cmake: $(command -v cmake) ($(cmake --version | head -1))"
# torch_bindings/CMakeLists.txt invokes a bare `python`, which must be this one.
echo "python: $(command -v python)"

# setup.py asserts torch is importable from the *build* interpreter, so the
# default isolated build env cannot work here.
echo "=== building ==="
uv pip install --no-build-isolation .

echo "=== smoke test ==="
# Run outside the repo so we import the installed package and not a local
# ./radfoam left behind by a direct CMake build.
cd /tmp
python -c "
import torch, radfoam
print('radfoam:', radfoam.__file__)
print('device :', torch.cuda.get_device_name(0))

pts = torch.rand(50_000, 3, device='cuda')
tri = radfoam.Triangulation(pts)
perm = tri.permutation().to(torch.long)
adj = tri.point_adjacency().to(torch.int64)
off = tri.point_adjacency_offsets().to(torch.int64)

assert torch.equal(perm.sort().values, torch.arange(pts.shape[0], device='cuda'))
assert 0 <= int(adj.min()) and int(adj.max()) < pts.shape[0]
assert int(off[-1]) == adj.shape[0] and bool(torch.all(off[1:] >= off[:-1]))
radfoam.build_aabb_tree(pts[perm])
torch.cuda.synchronize()
print('kernels OK: tets', tuple(tri.tets().shape), 'adjacency', tuple(adj.shape))
"

echo "=== done; activate with: source $REPO/$VENV_DIR/bin/activate ==="
