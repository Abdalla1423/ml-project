#!/bin/bash
# One entry point for everything on the Hydra cluster (shared). Run from the repo on hydra:
#
#   bash cluster/hydra.sh setup          # once: build the container (CPU job, ~15 min)
#   bash cluster/hydra.sh test           # run pytest on a CPU node (max 15 min)
#   bash cluster/hydra.sh data           # case studies -> /home/space/datasets/dopfn_case_studies
#   bash cluster/hydra.sh smoke          # 10-minute training test on a GPU
#   bash cluster/hydra.sh train [--long] [--name NAME] [--cpus N] [config]  # training (gpu-2h,
#                                        # --long: gpu-5h); again = continue; --name = separate run
#                                        # in outputs/train/NAME; --cpus = CPU cores (default 4)
#   bash cluster/hydra.sh status         # your jobs and the end of the newest log
#
# Every step submits a SLURM job and returns immediately; `status` shows how it is going.
# Run it in a normal login session (ssh <username>@hydra.ml.tu-berlin.de).
# Optional: export DOPFN_MAIL=<email> to get an email when a job ends or fails.
# See cluster/README.md for the step-by-step guide.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
SIF=${DOPFN_SIF:-$HOME/containers/dopfn.sif}
export DOPFN_CASE_STUDY_ROOT=${DOPFN_CASE_STUDY_ROOT:-/home/space/datasets/dopfn_case_studies}
export PYTHONPATH="$PWD/src:$PWD/Do-PFN-prior"
DO_PFN_URL=https://github.com/jr2021/Do-PFN
mkdir -p logs

for tool in sbatch squeue; do
    command -v "$tool" >/dev/null || {
        echo "SLURM command '$tool' not found. Run this on hydra in a login shell (bash -l)." >&2
        exit 1
    }
done

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
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
    local job mail=()
    if [[ -n "${DOPFN_MAIL:-}" ]]; then
        mail=(--mail-type=END,FAIL --mail-user="$DOPFN_MAIL")
    fi
    # --nodes=1: without it, SLURM on hydra spreads "N CPUs" over N nodes (one GPU on each).
    job=$(sbatch --parsable --nodes=1 --job-name="dopfn-$name" --output="logs/$name-%j.out" ${mail[@]+"${mail[@]}"} "$@")
    echo "Submitted job $job ($name). Log: logs/$name-$job.out"
    if [[ -n "${DOPFN_MAIL:-}" ]]; then
        echo "An email goes to $DOPFN_MAIL when the job ends or fails."
    fi
    echo "Check it with: bash cluster/hydra.sh status"
}

header='#!/bin/bash
set -euo pipefail
echo "job ${SLURM_JOB_ID} on $(hostname) at $(date), commit $(git rev-parse --short HEAD)"
# Apptainer only mounts $HOME, /tmp and the current folder; make the shared datasets visible too.
if [[ -d /home/space ]]; then export APPTAINER_BINDPATH=/home/space; fi'

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
mkdir -p "$DOPFN_CASE_STUDY_ROOT" || { echo "Cannot create $DOPFN_CASE_STUDY_ROOT (no write access?). Ask the cluster admins, or set DOPFN_CASE_STUDY_ROOT to a folder you own." >&2; exit 1; }
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
    # --name NAME: a separate run with its own output folder outputs/train/NAME (default: default),
    # so it can run next to another one.
    # --cpus N: CPU cores to request. Hydra allows 4 cores per GPU (CpusPerTres=gpu:4); a job that
    # asks for more stays pending forever with reason PartitionConfig.
    partition=gpu-2h
    minutes=105
    config=configs/train/default.yaml
    name=default
    cpus=4
    while [[ $# -gt 0 ]]; do
        case "$1" in
        --long) partition=gpu-5h minutes=285 ;;
        --name)
            name=${2:-}
            [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "--name needs a name of letters, digits, - or _" >&2; exit 1; }
            shift
            ;;
        --cpus)
            cpus=${2:-}
            [[ "$cpus" =~ ^[1-9][0-9]*$ ]] || { echo "--cpus needs a number, e.g. --cpus 4" >&2; exit 1; }
            shift
            ;;
        -*) echo "Unknown option: $1" >&2; exit 1 ;;
        *) config=$1 ;;
        esac
        shift
    done
    [[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 1; }
    job_name=train
    [[ "$name" == default ]] || job_name=train-$name
    out_dir=outputs/train/$name
    workers=$((cpus > 1 ? cpus - 1 : 1))  # data-generation processes; one core for the main process
    # Only one job per run name: two jobs on the same folder would overwrite each other's checkpoints.
    existing=$(squeue -h -u "$USER" -n "dopfn-$job_name" -o "%i (%T)" | paste -sd ' ' -)
    if [[ -n "$existing" ]]; then
        echo "A training job for '$name' is already queued or running: $existing." >&2
        echo "Wait for it, cancel it (scancel <JOBID>), or start a separate run: train --name <other-name>" >&2
        exit 1
    fi
    submit "$job_name" --partition=$partition --gpus-per-node=1 --ntasks-per-node="$cpus" <<EOF
$header
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv "$SIF" python scripts/train.py --config "$config" \\
    --device cuda --resume --max-minutes $minutes --out-dir "$out_dir" --num-workers $workers
echo "Results in $out_dir"
EOF
    ;;

status)
    squeue -u "$USER" -o "%.10i %.20j %.10T %.10M %.12P %R"
    # Jobs that have not started yet have no log file.
    while read -r id name state reason; do
        [[ "$name" == dopfn-* && "$state" == PENDING ]] || continue
        if [[ "$reason" == PartitionConfig ]]; then
            echo "-> Job $id ($name) cannot start: its request does not fit the partition (PartitionConfig)."
            echo "   Waiting does not help. See 'A job stays PENDING' in cluster/README.md."
        else
            echo "-> Job $id ($name) is waiting in the queue; its log appears when it starts."
        fi
    done < <(squeue -h -u "$USER" -o "%i %j %T %r")
    newest=$(ls -t logs/*.out 2>/dev/null | head -n 1 || true)
    if [[ -n "$newest" ]]; then
        echo
        echo "== newest log: $newest (last 15 lines)"
        tail -n 15 "$newest"
    fi
    ;;

*)
    usage
    ;;
esac
