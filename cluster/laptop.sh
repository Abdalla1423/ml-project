#!/bin/bash
# Run every cluster step from your own computer (macOS or Linux; on Windows use WSL).
# Run it inside your local copy of the repo:
#
#   bash cluster/laptop.sh connect <cluster-username>  # once: save username + connection test
#   bash cluster/laptop.sh setup                       # build the container on hydra (once)
#   bash cluster/laptop.sh test                        # run the tests on hydra
#   bash cluster/laptop.sh data                        # case studies -> /home/space/datasets
#   bash cluster/laptop.sh smoke                       # 10-minute GPU training test
#   bash cluster/laptop.sh train [--long] [config]     # training, 2 h (--long: 5 h); again = continue
#   bash cluster/laptop.sh status                      # jobs on hydra + end of the newest log
#   bash cluster/laptop.sh results                     # copy hydra's outputs/ to ./outputs/
#
# setup/test/data/smoke/train first copy your local code to ~/ml-project on hydra (no GitHub
# access needed there), then run `bash cluster/hydra.sh <step>` on hydra.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# The username is stored here by `connect` (gitignored, one line).
USER_FILE=.cluster-user
# Entry node: hydra, or karkinos when hydra is down (DOPFN_ENTRY=karkinos).
ENTRY=${DOPFN_ENTRY:-hydra}
REMOTE_DIR=${DOPFN_REMOTE_DIR:-ml-project}
# ssh options: fail after 15 s instead of hanging; optional explicit key (DOPFN_SSH_KEY).
SSH_OPTS=(-o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
if [[ -n "${DOPFN_SSH_KEY:-}" ]]; then
    [[ -f "$DOPFN_SSH_KEY" ]] || { echo "Error: DOPFN_SSH_KEY is set, but there is no file $DOPFN_SSH_KEY." >&2; exit 1; }
    SSH_OPTS+=(-i "$DOPFN_SSH_KEY")
fi

die() {
    echo "Error: $*" >&2
    exit 1
}

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

# user@hydra.ml.tu-berlin.de: exactly what a direct `ssh` uses, so the same ssh settings apply.
target() {
    [[ -s "$USER_FILE" ]] || die "Run 'bash cluster/laptop.sh connect <cluster-username>' first."
    echo "$(head -n 1 "$USER_FILE")@$ENTRY.ml.tu-berlin.de"
}

# Earlier versions added a 'Host hydra' block to ~/.ssh/config; remove it so it can't interfere.
remove_old_ssh_config() {
    local config="$HOME/.ssh/config" tmp
    [[ -f "$config" ]] && grep -q "^# Added by ml-project/cluster/laptop.sh" "$config" || return 0
    tmp=$(mktemp)
    awk '
        /^# Added by ml-project\/cluster\/laptop.sh/ { skip = 1; next }
        skip && (/^Host (hydra|karkinos)[[:space:]]*$/ || /^[[:space:]]/ || /^$/) { next }
        { skip = 0; print }
    ' "$config" >"$tmp" && cat "$tmp" >"$config" && rm -f "$tmp"
    echo "Removed the 'hydra'/'karkinos' entries an earlier version added to ~/.ssh/config."
}

connect() {
    local user=${1:-}
    [[ -n "$user" ]] || die "Give your cluster username: bash cluster/laptop.sh connect <username>"
    remove_old_ssh_config
    echo "$user" >"$USER_FILE"
    local host
    host=$(target)
    echo "Connecting to $host ..."
    ssh "${SSH_OPTS[@]}" -o StrictHostKeyChecking=accept-new "$host" 'echo "Connected to $(hostname) as $USER."' ||
        die "Could not connect as $host.
       Does 'ssh $host' work in this terminal? If you normally connect with 'ssh -i <keyfile> ...',
       run: DOPFN_SSH_KEY=<keyfile> bash cluster/laptop.sh connect $user
       (and set DOPFN_SSH_KEY for the other commands too, e.g. export it in your shell).
       If hydra is down (it reboots Mondays 01:00), try: DOPFN_ENTRY=karkinos bash cluster/laptop.sh connect $user"
    echo "Done. Next: bash cluster/laptop.sh setup"
}

sync_code() {
    local host=$1
    echo "Copying code to $host:$REMOTE_DIR ..."
    rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
        --exclude .venv/ --exclude outputs/ --exclude logs/ --exclude __pycache__/ \
        --exclude .pytest_cache/ --exclude CLAUDE.local.md --exclude .claude/ --exclude .DS_Store \
        --exclude "$USER_FILE" \
        ./ "$host:$REMOTE_DIR/"
}

# Runs hydra.sh in a login shell: only login shells have the SLURM commands (sbatch, squeue) on
# the PATH on hydra; plain `ssh host cmd` does not.
on_hydra() {
    local host=$1
    shift
    ssh "${SSH_OPTS[@]}" "$host" "bash -lc 'cd $REMOTE_DIR && bash cluster/hydra.sh $*'" ||
        die "The command on $host failed or the cluster did not answer (see the messages above).
       If hydra does not answer, try the second entry node: DOPFN_ENTRY=karkinos bash cluster/laptop.sh $1"
}

cmd=${1:-}
[[ -n "$cmd" ]] || usage
shift

case "$cmd" in
connect)
    connect "$@"
    ;;
setup | test | data | smoke | train)
    command -v rsync >/dev/null || die "rsync is not installed (on Windows, run this inside WSL)."
    host=$(target)
    sync_code "$host"
    on_hydra "$host" "$cmd" "$@"
    ;;
status)
    host=$(target)
    on_hydra "$host" status
    ;;
results)
    host=$(target)
    mkdir -p outputs
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$host:$REMOTE_DIR/outputs/" outputs/
    echo "Copied to ./outputs/ (e.g. outputs/train/default/loss_curve.png)."
    ;;
*)
    usage
    ;;
esac
