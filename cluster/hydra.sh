#!/bin/bash
# One entry point for everything on the Hydra cluster (shared). Run from the repo on hydra:
#
#   bash cluster/hydra.sh setup          # once: build the container (CPU job, ~15 min)
#   bash cluster/hydra.sh test           # run pytest on a CPU node (max 15 min)
#   bash cluster/hydra.sh data           # case studies -> /home/space/datasets/dopfn_case_studies
#   bash cluster/hydra.sh smoke          # 10-minute training test on a GPU
#   bash cluster/hydra.sh train [--long] [config]  # training (gpu-2h, --long: gpu-5h); again = continue
#   bash cluster/hydra.sh status         # your jobs and the end of the newest log
#
# Every step submits a SLURM job and returns immediately; `status` shows how it is going.
# See cluster/README.md for the step-by-step guide.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
SIF=${DOPFN_SIF:-$HOME/containers/dopfn.sif}
export DOPFN_CASE_STUDY_ROOT=${DOPFN_CASE_STUDY_ROOT:-/home/space/datasets/dopfn_case_studies}
export PYTHONPATH="$PWD/src:$PWD/Do-PFN-prior"
DO_PFN_URL=https://github.com/jr2021/Do-PFN
mkdir -p logs

usage() {
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

need_container() {
    if [[ ! -f "$SIF" ]]; then
        echo "No container at $SIF yet. Run 'bash cluster/hydra.sh setup' first and wait for the job." >&2
        exit 1
    fi
}

# submit NAME SBATCH_OPTIONS... : submits the job script read from stdin.
submit() {
    local name=$1
    shift
    local job
    job=$(sbatch --parsable --job-name="dopfn-$name" --output="logs/$name-%j.out" "$@")
    echo "Submitted job $job ($name). Log: logs/$name-$job.out"
    echo "Check it with: bash cluster/hydra.sh status"
}

header='#!/bin/bash
set -euo pipefail
echo "job ${SLURM_JOB_ID} on $(hostname) at $(date), commit $(git rev-parse --short HEAD)"'

cmd=${1:-}
[[ -n "$cmd" ]] || usage
shift

case "$cmd" in
setup)
    if [[ -f "$SIF" ]]; then
        echo "Container already exists: $SIF (delete it to rebuild after dependency changes)."
        exit 0
    fi
    mkdir -p "$(dirname "$SIF")"
    submit setup --partition=cpu-2h --ntasks-per-node=4 <<EOF
$header
export APPTAINER_CACHEDIR=/tmp/\$USER-apptainer-cache APPTAINER_TMPDIR=/tmp
apptainer build "$SIF" cluster/dopfn.def || apptainer build --fakeroot "$SIF" cluster/dopfn.def
echo "Container ready: $SIF"
EOF
    ;;

test)
    need_container
    submit test --partition=cpu-test --ntasks-per-node=4 <<EOF
$header
apptainer exec "$SIF" python -m pytest -q -rs
EOF
    ;;

data)
    need_container
    submit data --partition=cpu-2h --ntasks-per-node=2 <<EOF
$header
src=\$(mktemp -d /tmp/dopfn-data-XXXX)
git clone --quiet --depth 1 --filter=blob:none --sparse $DO_PFN_URL "\$src/Do-PFN"
git -C "\$src/Do-PFN" sparse-checkout set data/prior_sampling
apptainer exec "$SIF" python scripts/generate_data.py case-studies \\
    --source "\$src/Do-PFN/data/prior_sampling" --out "$DOPFN_CASE_STUDY_ROOT"
chmod -R a+rX "$DOPFN_CASE_STUDY_ROOT"
echo "Case studies ready in $DOPFN_CASE_STUDY_ROOT"
EOF
    ;;

smoke)
    need_container
    submit smoke --partition=gpu-test --gpus-per-node=1 --ntasks-per-node=4 <<EOF
$header
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv "$SIF" python scripts/train.py --config configs/train/debug.yaml \\
    --device cuda --num-workers 2 --max-minutes 10 --out-dir outputs/train/smoke
echo "Smoke test passed: outputs/train/smoke"
EOF
    ;;

train)
    need_container
    # Default: gpu-2h (starts sooner); --long: gpu-5h. The job stops cleanly 15 min before the
    # limit, and the next 'train' continues from the checkpoint.
    partition=gpu-2h
    minutes=105
    config=configs/train/default.yaml
    for arg in "$@"; do
        case "$arg" in
        --long) partition=gpu-5h minutes=285 ;;
        -*) echo "Unknown option: $arg" >&2; exit 1 ;;
        *) config=$arg ;;
        esac
    done
    [[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 1; }
    submit train --partition=$partition --gpus-per-node=1 --constraint="80gb|40gb" --ntasks-per-node=8 <<EOF
$header
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv "$SIF" python scripts/train.py --config "$config" \\
    --device cuda --resume --max-minutes $minutes
EOF
    ;;

status)
    squeue -u "$USER" -o "%.10i %.20j %.10T %.10M %.12P %R"
    newest=$(ls -t logs/*.out 2>/dev/null | head -n 1 || true)
    if [[ -n "$newest" ]]; then
        echo
        echo "== $newest (last 15 lines)"
        tail -n 15 "$newest"
    fi
    ;;

*)
    usage
    ;;
esac
