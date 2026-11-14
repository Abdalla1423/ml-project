# Third-party code

Copies of external repositories, one top-level folder each, as recommended by our supervisor
(decision D7 in `docs/decisions.md`). They may be changed and combined as needed; record each new
copy in the table and each change below. Paths are relative to the repo root. Neither repository
has a license file; all credit goes to the authors.

| Directory | Source | Commit | Used for |
|---|---|---|---|
| `Do-PFN-prior/` | https://github.com/oossen/Do-PFN-prior (Ole Ossen) | `325559029f355e33abc6edbfc60c2eb9bae3124a` (2026-03-29) | Our prior extends it (`dopfn.data.prior`). Its package `dopfnprior` is installed by our `pyproject.toml`. Its training loop (`dopfnprior/utils/train.py`, `example.py`) is a reference for `dopfn.training`. |
| nanoTabPFN (adapted, not copied) | https://github.com/automl/nanoTabPFN (Pfefferle et al.), Apache License 2.0 | `6071eec58a1a3225854480125001d5a62e279f6d` (2026-09-18) | `src/dopfn/model/transformer.py` adapts its `model.py`; the changes are listed in that file's docstring. |
| `Do-PFN/` | https://github.com/jr2021/Do-PFN (Robertson et al.) | `90d67433b43c4d52d752dc336070f525ff856e0b` (2025-07-16) | Reading the original case-study pickles (`scripts/generate_data.py case-studies`). Reference implementation of the model, the bar distribution and `DoPFNRegressor`. Not installed as a package (its top-level module names `datasets`, `model`, `utils`, `scripts` are too generic). |

## Not copied

- `Do-PFN/reproduce.ipynb`: the sheet says "Do not submit Jupyter notebooks", and this repo is
  what we submit.
- `Do-PFN-prior/workdir/`: two 60 MB training checkpoints and TensorBoard logs. Our choice, for
  size (our `.gitignore` also ignores `*.pth`); we train our own models.
- `Do-PFN/artifacts/`: the pretrained Do-PFN model (108 MB). Our choice, for size (GitHub warns
  above 50 MB per file). Needed later for comparisons: keep it on the cluster.
- `Do-PFN/data/`: the case studies (67 MB) and semi-real datasets. The sheet asks for data in
  `/home/space/datasets/<dataset>` on the cluster; converted case studies go there (main `README.md`).
- Git history and `__pycache__/` folders.

The copies were first uploaded with only part of the files (2026-11-14); the remaining code files
were added afterwards so that e.g. `Do-PFN/priors/doscm.py` can import `Do-PFN/utils.py`.

Clone the upstream repositories at the commits above to get these files.

## Changes to the upstream code

Every change is marked with `MODIFIED (dopfn replication)` in the code, or listed here.

- `Do-PFN-prior/requirements.txt`: TFM-Playground installed via https instead of ssh.
- `Do-PFN/requirements.txt`: trailing newline added.
- `Do-PFN-prior/dopfnprior/scm/scm.py`: `get_default_device` is defined locally (returns the CPU)
  instead of being imported from TFM-Playground, so the prior does not depend on TFM-Playground.
- `Do-PFN-prior/dopfnprior/configs/tabicl_activations.py`: one docstring made a raw string, to remove
  a `SyntaxWarning: invalid escape sequence`.

Our own extensions (treatment node, do(t), ground-truth CATE) are in `src/dopfn/data/`, not here.
