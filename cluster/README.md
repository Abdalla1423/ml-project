# Running the project on the Hydra cluster

The official cluster documentation (partitions, GPUs, file systems, rules) is copied in
[docs/hydra-cluster.md](../docs/hydra-cluster.md).

You need the cluster username from the activation email. Below it is written as `<username>`;
replace it everywhere (e.g. `ssh <username>@hydra.ml.tu-berlin.de`).

## 1. Copy the code to the cluster (on your computer)

In a terminal on your own computer, inside your local copy of this repo:

```bash
rsync -az --delete --exclude .venv/ --exclude outputs/ --exclude logs/ --exclude __pycache__/ \
    --exclude CLAUDE.local.md --exclude .claude/ \
    ./ <username>@hydra.ml.tu-berlin.de:ml-project/
```

This copies the repo to `~/ml-project` on the cluster. Run it again whenever the code changed
(e.g. after `git pull`). Results on the cluster (`outputs/`, `logs/`) are never deleted by it.

## 2. Log in and run the steps (on the cluster)

```bash
ssh <username>@hydra.ml.tu-berlin.de
cd ml-project
```

Then run the steps in this order. Each command submits a cluster job and returns right away. Check
it with `bash cluster/hydra.sh status` (shows your jobs and the end of the newest log) and wait
until the job is finished before starting the next step.

| Step | Command | Done when `status` shows | Takes |
|---|---|---|---|
| 1. Build the container (once) | `bash cluster/hydra.sh setup` | `Container ready` | 15–60 min |
| 2. Run the tests | `bash cluster/hydra.sh test` | `passed` | ~2 min |
| 3. Case studies to `/home/space/datasets` | `bash cluster/hydra.sh data` | `Case studies ready` | ~5 min |
| 4. Short GPU training test | `bash cluster/hydra.sh smoke` | `Smoke test passed` | ~10 min |
| 5. Full training | `bash cluster/hydra.sh train` | `step 30000` or `Stopping after ...` | up to 2 h |

- Training runs as a 2 h job (short jobs start sooner) and stops on its own shortly before the
  limit. If `status` shows `Stopping after ...`, run `bash cluster/hydra.sh train` again: it
  continues from the last checkpoint. `train --long` requests a 5 h job instead.
- **A second run next to the first:** `bash cluster/hydra.sh train --name run2` starts a separate
  run with its own folder `outputs/train/run2`, so it never touches the first run's checkpoints. To
  continue it later, use the same `--name`. Without `--name` the run is `outputs/train/default`.
- You can log out (`exit`) while a job runs; it keeps running. Log in again to check `status`.
- **Email when a job is done:** run this once on the cluster, with your email address:

  ```bash
  echo 'export DOPFN_MAIL=<your-email>' >> ~/.bashrc && source ~/.bashrc
  ```

  From then on, every step sends an email when its job ends or fails (jobs submitted before do
  not).

## 3. Get the results (on your computer)

```bash
rsync -az <username>@hydra.ml.tu-berlin.de:ml-project/outputs/ outputs/
```

The training results are then in `outputs/train/default/`: `loss_curve.png` (report figure),
`metrics.csv` (all numbers), `checkpoint_best.pt` (the trained model).

## If something goes wrong

- **A job failed:** `bash cluster/hydra.sh status` shows the end of its log; send that to the team.
  All logs are in `~/ml-project/logs/`.
- **hydra does not answer:** it reboots Mondays at 01:00. Use the second entry node
  `karkinos.ml.tu-berlin.de` instead of `hydra.ml.tu-berlin.de` (same files, same jobs).
- **`setup` fails:** the script already retries with `--fakeroot`; otherwise ask the cluster admins.
- **`data` says permission denied:** you need write access to `/home/space/datasets`; ask the
  cluster admins.
- **A job stays `PENDING`:** look at the reason in the last column of `status`.
  - `Priority` or `Resources`: the cluster is busy, just wait.
  - `PartitionConfig`: the job asks for something this partition does not offer, so it never
    starts. Cancel it (`scancel <job-id>`) and submit again with fewer CPU cores, e.g.
    `bash cluster/hydra.sh train --cpus 4`. To see whether a request can run at all, without
    submitting anything:
    `sbatch --test-only -p gpu-2h --gpus-per-node=1 --ntasks-per-node=4 --wrap=true`
- **Cancel a job:** `scancel <job-id>` (the id is shown by `status`).

## Getting an account (new team members)

Run `ssh-keygen -t rsa -b 4096` (Enter at every question) and email the public file
`~/.ssh/id_rsa.pub` as an attachment to Dominik Kühne (dominik.kuehne@tu-berlin.de) with the subject
**PML**. You get an activation email with your username.
