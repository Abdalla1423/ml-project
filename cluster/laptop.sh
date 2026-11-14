#!/bin/bash
# Run every cluster step from your own computer (macOS or Linux; on Windows use WSL).
# Run it inside your local copy of the repo:
#
#   bash cluster/laptop.sh connect <cluster-username>  # once: ssh shortcut + connection test
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
HOST=${DOPFN_HOST:-hydra}
REMOTE_DIR=${DOPFN_REMOTE_DIR:-ml-project}
KEY=${DOPFN_SSH_KEY:-$HOME/.ssh/id_rsa}

die() {
    echo "Error: $*" >&2
    exit 1
}

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

connect() {
    local user=${1:-}
    [[ -n "$user" ]] || die "Give your cluster username: bash cluster/laptop.sh connect <username>"
    [[ -f "$KEY" ]] || die "No ssh key at $KEY. Use the key whose .pub file you emailed for the account
       (or set DOPFN_SSH_KEY=/path/to/private/key)."
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
    touch "$HOME/.ssh/config"
    chmod 600 "$HOME/.ssh/config"
    if grep -qE "^Host[[:space:]]+$HOST([[:space:]]|$)" "$HOME/.ssh/config"; then
        echo "~/.ssh/config already has 'Host $HOST', leaving it unchanged."
    else
        cat >>"$HOME/.ssh/config" <<EOF

# Added by ml-project/cluster/laptop.sh
Host hydra
   HostName hydra.ml.tu-berlin.de
   User $user
   IdentityFile $KEY
Host karkinos
   HostName karkinos.ml.tu-berlin.de
   User $user
   IdentityFile $KEY
EOF
        echo "Added 'hydra' and 'karkinos' to ~/.ssh/config."
    fi
    echo "Connecting to $HOST ..."
    ssh -o StrictHostKeyChecking=accept-new "$HOST" 'echo "Connected to $(hostname) as $USER."' ||
        die "Could not connect. Check the username from the activation email and that $KEY is
       the key you emailed. If hydra is rebooting (Mondays 01:00), try: DOPFN_HOST=karkinos bash cluster/laptop.sh connect $user"
    echo "Done. Next: bash cluster/laptop.sh setup"
}

sync_code() {
    echo "Copying code to $HOST:$REMOTE_DIR ..."
    rsync -az --delete \
        --exclude .venv/ --exclude outputs/ --exclude logs/ --exclude __pycache__/ \
        --exclude .pytest_cache/ --exclude CLAUDE.local.md --exclude .claude/ --exclude .DS_Store \
        ./ "$HOST:$REMOTE_DIR/"
}

on_hydra() {
    ssh "$HOST" "cd $REMOTE_DIR && bash cluster/hydra.sh $*"
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
    sync_code
    on_hydra "$cmd" "$@"
    ;;
status)
    on_hydra status
    ;;
results)
    mkdir -p outputs
    rsync -az "$HOST:$REMOTE_DIR/outputs/" outputs/
    echo "Copied to ./outputs/ (e.g. outputs/train/default/loss_curve.png)."
    ;;
*)
    usage
    ;;
esac
