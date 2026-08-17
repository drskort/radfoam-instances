#!/bin/bash
#SBATCH --job-name=foamviz-cluster
#SBATCH --partition=a40-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=foamviz-cluster-%j.log
#
# Fit HDBSCAN once per run and cache it, so every downstream consumer shares
# one clustering instead of re-fitting its own.
#
# Defaults to method=full: cuML HDBSCAN over EVERY primitive, which yields an
# exact label per cell. The alternative (METHOD=sample) fits on a 60k subsample
# -- 1.5% of a 4M-point cloud -- and assigns the rest by nearest centroid,
# which loses small objects before anything downstream can see them.
#
# Until now only the summary was kept -- instances/clusters.json holds
# {n_clusters, noise_fraction} and nothing else -- so render_instances,
# extract_instance_language and eval_lerf_grounded each re-ran the fit. That is
# both wasted work and the exact failure instance_cluster.py warns about: if any
# of them disagrees, instance 7 in the viewer is a different object from
# instance 7 in the language table.
#
# The cache lands at output/<run>/instances/clustering.pt and carries a
# fingerprint of the features it was fitted on, so a stale one is detected
# rather than silently reused.
#
#   sbatch scripts/foamviz_cluster_slurm.sh                    # every run with features
#   sbatch scripts/foamviz_cluster_slurm.sh output/garden_inst_geo output/ramen_inst_geo
#   REFIT=1 sbatch scripts/foamviz_cluster_slurm.sh            # ignore existing caches
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

if [ -f /usr/share/modules/init/bash ]; then
    source /usr/share/modules/init/bash
    module load cuda/12.1
fi
source .venv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# cuML links against the CUDA libs shipped inside the venv; fit_clusters_full
# needs it, and without this it fails at import with a missing .so.
NVLIBS=$(find "$REPO/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:-}"

MODEL="${MODEL:-model.pt}"
METHOD="${METHOD:-full}"
# An `if`, not `[ ... ] && ...`: under `set -e` a false test makes the whole
# compound return 1 and takes the script down with it.
REFIT_FLAG=""
if [ -n "${REFIT:-}" ]; then
    REFIT_FLAG="--refit"
fi

# Default to every run that actually has instance features. feat_dim is read
# from the run config rather than the checkpoint, so selecting the scenes costs
# nothing -- opening each 1.7 GB model.pt just to filter would not.
if [ "$#" -gt 0 ]; then
    SCENES=("$@")
else
    SCENES=()
    for config in output/*/config.yaml; do
        run="$(dirname "$config")"
        [ -f "$run/$MODEL" ] || continue
        # `|| true`: runs without instance features have no feat_dim line, and
        # under `set -o pipefail` a failing grep would abort the whole script.
        dim="$(grep -E '^feat_dim:' "$config" | awk '{print $2}' || true)"
        if [ -n "$dim" ] && [ "$dim" -gt 0 ] 2>/dev/null; then
            SCENES+=("$run")
        fi
    done
fi

if [ "${#SCENES[@]}" -eq 0 ]; then
    echo "no runs with feat_dim > 0 and a $MODEL -- nothing to do" >&2
    exit 1
fi

echo "=== $(hostname): clustering ${#SCENES[@]} run(s), model=$MODEL, method=$METHOD ==="
printf '  %s\n' "${SCENES[@]}"

failed=()
for scene in "${SCENES[@]}"; do
    echo
    echo "--- $scene ---"
    # One bad run must not abandon the rest of the batch.
    if ! python scripts/foamviz.py cluster \
            --checkpoint "$scene" --model "$MODEL" \
            --method "$METHOD" $REFIT_FLAG; then
        echo "FAILED: $scene" >&2
        failed+=("$scene")
    fi
done

echo
if [ "${#failed[@]}" -gt 0 ]; then
    echo "=== done, ${#failed[@]} failed ==="
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
echo "=== done, all ${#SCENES[@]} cached ==="
