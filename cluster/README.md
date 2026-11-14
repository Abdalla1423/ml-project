# Running the project on the Hydra cluster

Everything goes through one script, `cluster/hydra.sh`. Each step submits a SLURM job and returns
right away. Check how it's going with `bash cluster/hydra.sh status`.

## Once: get access from your computer

Run these commands in a terminal on your own computer (macOS/Linux: Terminal, Windows 10/11:
PowerShell).

1. **Create a key pair.** Press Enter at every question to accept the defaults:

   ```bash
   ssh-keygen -t rsa -b 4096
   ```

   This creates `~/.ssh/id_rsa` (private, never share it) and `~/.ssh/id_rsa.pub` (public).
   If it asks to overwrite an existing key, answer `n` and use the existing one.

2. **Request the account.** Email the **public** file `~/.ssh/id_rsa.pub` as an attachment to
   Dominik Kühne (dominik.kuehne@tu-berlin.de) with the subject **PML**. On macOS, `open ~/.ssh`
   shows the folder (it is hidden in Finder otherwise); on Windows it is
   `C:\Users\<you>\.ssh`.

3. **Wait for the activation email.** It tells you your cluster username.

4. **Save a shortcut** so `ssh hydra` is enough. Add this to `~/.ssh/config` on your computer
   (create the file if it doesn't exist), with your cluster username:

   ```
   Host hydra
      HostName hydra.ml.tu-berlin.de
      User <username>
      IdentityFile ~/.ssh/id_rsa
   ```

5. **Connect:** `ssh hydra`. The first time, answer `yes` to the fingerprint question.
   If hydra is down (it reboots Mondays at 01:00), use the second entry node
   `karkinos.ml.tu-berlin.de` instead.

## Once: get the code onto the cluster

The repo is private, so the cluster needs its own access to GitHub. On hydra (after `ssh hydra`):

```bash
ssh-keygen -t ed25519            # Enter at every question
cat ~/.ssh/id_ed25519.pub        # copy this line
```

Add the copied line on GitHub under **Settings → SSH and GPG keys → New SSH key**. Then, still on
hydra:

```bash
git clone git@github.com:Abdalla1423/ml-project.git
cd ml-project
```

To update the code later: `cd ml-project && git pull`.

## Steps (in this order)

Wait for each job to finish before starting the next. `status` shows your jobs and the end of the
newest log.

| Step | Command | Takes | Done when the log ends with |
|---|---|---|---|
| 1. Build the container | `bash cluster/hydra.sh setup` | ~15 min | `Container ready` |
| 2. Run the tests | `bash cluster/hydra.sh test` | ~2 min | `passed` |
| 3. Put the case studies on the cluster | `bash cluster/hydra.sh data` | ~5 min | `Case studies ready` |
| 4. Short training test on a GPU | `bash cluster/hydra.sh smoke` | ~10 min | `Smoke test passed` |
| 5. Full training | `bash cluster/hydra.sh train` | up to 5 h | `Stopping after ...` or `step 30000` |

- Step 1 is only needed once (again only if `cluster/dopfn.def` changes).
- After step 3, running step 2 again also tests the real case studies (one skipped test fewer).
- Step 5 stops on its own shortly before the 5 h limit. If the log ends with `Stopping after ...`,
  run `bash cluster/hydra.sh train` again: it continues from the last checkpoint.

## Results

Training writes to `outputs/train/default/`:

- `loss_curve.png`: training and validation loss (report figure)
- `metrics.csv`: all logged numbers
- `checkpoint_best.pt`: the model to evaluate

Copy them to your computer (run this on your computer, not on hydra):

```bash
scp -r hydra:ml-project/outputs/train/default ./
```

## If something goes wrong

- **A job fails:** read its log in `logs/` (the name is printed when you submit) and send it to the
  team.
- **`setup` fails:** the cluster's Apptainer tutorial explains building; the script already retries
  with `--fakeroot`. Otherwise ask the cluster admins.
- **`data` fails with permission denied:** `/home/space/datasets` must be writable for you; ask the
  cluster admins, or set `DOPFN_CASE_STUDY_ROOT` to a folder you own before every command.
- **A job waits for a long time:** normal when the cluster is busy (`status` shows `PENDING`).

Settings you can change with environment variables: `DOPFN_SIF` (container path, default
`~/containers/dopfn.sif`) and `DOPFN_CASE_STUDY_ROOT` (default
`/home/space/datasets/dopfn_case_studies`).
