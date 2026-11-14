# Running the project on the Hydra cluster

You run everything from **your own computer**, inside your local copy of this repo, with
`cluster/laptop.sh`. It copies your code to hydra and starts the step there as a cluster job.
Works on macOS and Linux; on Windows, run it inside WSL.

**Before you start:** you have the cluster activation email (it contains your username), and the
key you emailed for the account is `~/.ssh/id_rsa` on this computer.

## 1. Connect (once)

```bash
bash cluster/laptop.sh connect <your-cluster-username>
```

It adds a `hydra` shortcut to `~/.ssh/config` and tests the connection. The first time, answer
`yes` if ssh asks about the fingerprint. Ends with `Done`.

## 2. Run the steps (in this order)

Each command returns right away; the job runs on the cluster. Check it with
`bash cluster/laptop.sh status` (shows your jobs and the end of the newest log). Wait until a job is
finished before starting the next step.

| Step | Command | Takes | Done when `status` shows |
|---|---|---|---|
| Build the container (once) | `bash cluster/laptop.sh setup` | ~15 min | `Container ready` |
| Run the tests | `bash cluster/laptop.sh test` | ~2 min | `passed` |
| Case studies to `/home/space/datasets` | `bash cluster/laptop.sh data` | ~5 min | `Case studies ready` |
| Short GPU training test | `bash cluster/laptop.sh smoke` | ~10 min | `Smoke test passed` |
| Full training | `bash cluster/laptop.sh train` | up to 2 h | `Stopping after ...` or `step 30000` |

- Training runs as a 2 h job (short jobs start sooner) and stops on its own shortly before the
  limit. If `status` shows `Stopping after ...`, run `bash cluster/laptop.sh train` again: it
  continues from the last checkpoint. `train --long` requests a 5 h job instead, which is only
  worth it for much bigger models.
- Every step uses the code currently on your computer, so `git pull` first to get the team's
  latest version.

## 3. Get the results

```bash
bash cluster/laptop.sh results
```

Copies everything to `./outputs/`. The training results are in `outputs/train/default/`:
`loss_curve.png` (report figure), `metrics.csv` (all numbers), `checkpoint_best.pt` (the trained
model).

## If something goes wrong

- **`connect` fails:** check the username in the activation email. If hydra is rebooting (Mondays
  01:00), use the second entry node: `DOPFN_HOST=karkinos bash cluster/laptop.sh status`.
- **A job failed:** `status` shows the end of its log; send that to the team.
- **`setup` fails:** the script already retries with `--fakeroot`; otherwise ask the cluster admins.
- **`data` says permission denied:** you need write access to `/home/space/datasets`; ask the
  cluster admins.
- **A job stays `PENDING`:** the cluster is busy, just wait.

## Working directly on hydra (optional)

After `ssh hydra`, the same steps are available as `cd ml-project && bash cluster/hydra.sh <step>`
(`setup`, `test`, `data`, `smoke`, `train`, `status`). `laptop.sh` only calls this script remotely.

## Getting an account (new team members)

Run `ssh-keygen -t rsa -b 4096` (Enter at every question) and email the public file
`~/.ssh/id_rsa.pub` as an attachment to Dominik Kühne (dominik.kuehne@tu-berlin.de) with the subject
**PML**. You get an activation email with your username; then continue with step 1.
