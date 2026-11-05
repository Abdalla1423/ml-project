# Milestone 1 requirements: Data and Prototype

Source: `sheet.pdf` in this folder (checked against the sheet on 2026-11-05). This checklist is used
to verify the submission. Tick an item only when its acceptance criterion is met, and note the proof
(test, file, figure) after it.

**Grading:** Implementation 1 point, Report 9 points (Dataset overview 4, Baseline and evaluation 3,
Discussion 2). If unsure about next steps, discuss them with the project supervisor.

**Goal.** Understand the paper and the data. Build a data loader, a visualization tool and a working
prototype, starting from Do-PFN-prior. Results need not match the paper, but:

- [ ] **G1. Pre-training converges.** Training loss decreases and plateaus on prior data.
  Accept: loss curve (train + validation) saved by `scripts/train.py` and shown in the report.
- [ ] **G2. Trained model beats simple baselines on prior data.** Accept: `scripts/evaluate.py`
  reports lower normalized MSE (interventional outcome and/or CATE) for the prototype than for every
  simple baseline (B1, B2) on held-out prior datasets.
- [ ] **G3. Built starting from Do-PFN-prior.** Do-PFN-prior only generates observational data and
  has no designated treatment; adding a binary treatment and sampling interventional data is our
  work. Accept: our prior extends Do-PFN-prior (dependency or adapted code with attribution, not a
  blind copy), noted in the report.

## Code deliverables

### Data loading (owner: A)

Signature: `def load_data(dataset='prior', transformation=None, n_train=None, n_test=None, ...)`
in `dopfn.data`.

- [ ] **D1. Signature.** `load_data` exists with exactly the parameters `dataset='prior'`,
  `transformation=None`, `n_train=None`, `n_test=None` (plus extras). Accept: `tests/test_data.py`
  checks the signature.
- [ ] **D2. One sample = one whole dataset.** Accept: each yielded item holds the full observational
  and interventional tables of one dataset. Verified by `tests/test_data.py`.
- [ ] **D3. Lazy iterable.** Accept: return value is an iterable/generator; datasets are produced on
  demand (no full materialization). Verified by `tests/test_data.py`.
- [ ] **D4. Yield format.** Each item contains `(X_ob, y_ob, X_in, y_in)`, the observational outcomes
  of the query points (for Dont-PFN in ms2), and the ground-truth CATE of the query points.
  Accept: `tests/test_data.py` checks presence and consistent shapes.
- [ ] **D5. Treatment in column 0 of X.** Accept: column 0 of `X_ob` and `X_in` is binary {0,1};
  verified by `tests/test_data.py`.
- [ ] **D6. `dataset='prior'`.** Samples an SCM, an observational dataset and interventional query
  points *from the same SCM*. Accept: `tests/test_data.py`; for a fixed seed the CATE and `y_in`
  are consistent with the SCM.
- [ ] **D7. Interventional query data correct.** `y_in` computed under do(t) (incoming edges of t cut,
  descendants recomputed, same noise as `x_in`, see `docs/decisions.md` D5). Ground-truth CATE =
  E[y|do(1),x] − E[y|do(0),x]. Accept: a test on a hand-built SCM with known effect.
- [ ] **D8. Case studies from disk.** Each case study of the Do-PFN repo selectable with one value of
  `dataset` (sheet example: `'observed_confounder'`), read from disk. Accept: `tests/test_data.py` loads every
  case study name (skipped if data is absent locally).
- [ ] **D9. `transformation` applied to every dataset.** Accept: a test with a dummy transformation
  confirms it is applied to every yielded item.
- [ ] **D10. `n_train` / `n_test` respected.** Accept: test that the number of rows / datasets matches.
- [ ] **D11. Data on the cluster.** Generated/case-study data stored in
  `/home/space/datasets/<dataset>` and readable by everyone (`chmod -R a+r`). Accept: manual check on
  the cluster, noted in `status.md`.

### Visualization (owner: C)

Signature: `def show(x, outfile=None, ...)` in `dopfn.viz`.

- [ ] **V1. Signature.** `show(x, outfile=None, ...)` exists. Accept: `tests/test_viz.py`.
- [ ] **V2. Input is a list of data points** (items as yielded by `load_data`). Accept: test calls
  `show` on a list from `load_data`.
- [ ] **V3. Visually pleasing image**, e.g. causal graph next to observational and interventional
  data. Accept: figures reviewed by the team and used in the report.
- [ ] **V4. Optional save.** With `outfile` set, the image is written to that path. Accept:
  `tests/test_viz.py` checks the file exists.
- [ ] **V5. Extensible for overlays** (e.g. model predictions on top of data). Accept: an overlay
  argument/hook exists and is used for at least one report figure.

### Model prototype (owner: B)

Signature: `class Model(torch.nn.Module)` with `__init__(self, ...)` and `forward(self, X, y_ob)`
in `dopfn.model`.

- [ ] **M1. Signature.** `Model` subclasses `torch.nn.Module`, `forward(self, X, y_ob)`.
  Accept: `tests/test_model.py`.
- [ ] **M2. Input.** `X` is a batch of tables, observational rows followed by query rows; `y_ob` is
  the observational outcomes. Accept: `tests/test_model.py` runs a forward pass on such a batch.
- [ ] **M3. Output is a predictive distribution** over the interventional outcomes of the query rows
  (e.g. bar distribution logits). Accept: output has one distribution per query row; test checks
  shape and that it normalizes.
- [ ] **M4. Differentiable, PyTorch ops only**, no Python loops over rows, careful with conditionals.
  Accept: `tests/test_model.py` backpropagates and checks all parameters receive gradients; code review.
- [ ] **M5. Treatment indicator.** The model knows column 0 is the treatment. Accept: present in the
  input representation, test checks swapping column 0 changes the output.
- [ ] **M6. Loss function.** NLL of the interventional outcomes under the predicted distribution.
  Accept: `tests/test_model.py` checks a finite scalar, lower for a correct than a wrong target.
- [ ] **M7. Pre-training loop on prior data** (e.g. based on the Do-PFN-prior training loop, which
  is adapted from TFM-Playground).
  Accept: `python scripts/train.py --config configs/train/<cfg>.yaml` runs on the cluster, saves
  checkpoints and loss logs.
- [ ] **M8. Trained until the loss converges.** Accept: same as G1, final checkpoint stored.

### Scikit-learn-like wrapper (owner: B)

Like `DoPFNRegressor` in the Do-PFN repo. `class Regressor` in `dopfn.model`.

- [ ] **R1. `__init__(self, model, ...)`.** Accept: `tests/test_model.py`.
- [ ] **R2. `fit(self, X_ob, y_ob)` only stores the observational data** (no training).
  Accept: test checks no parameter changes.
- [ ] **R3. `predict(self, X)`** returns the predicted outcome under the treatment given in column 0
  of X. Accept: test checks shape `(n,)`.
- [ ] **R4. `predict_cate(self, X)`** = prediction with t=1 minus prediction with t=0 (column 0
  overwritten). Accept: test checks it equals `predict(X with t=1) − predict(X with t=0)`.
- [ ] **R5. Works with the evaluation harness** (any object with `fit`/`predict`/`predict_cate`).
  Accept: `scripts/evaluate.py` evaluates `Regressor` and the baselines with the same code.

### Baselines and evaluation (owner: C)

Needed for G2 and the report.

- [ ] **B1. Regression baseline:** regression fitted on observational data, queried at t=0 and t=1.
- [ ] **B2. Difference of group means** (CATE = mean(y | t=1) − mean(y | t=0)).
- [ ] **E1. Metrics:** normalized MSE (by range of y) for outcomes and CATE.
- [ ] **E2. Evaluation harness** for any object with `fit`/`predict`/`predict_cate`, per case study
  and on held-out prior data. Accept: `tests/test_evaluation.py` runs it on a dummy model.
- [ ] **E3. Overfitting check:** validation loss on held-out prior datasets; train/test separation
  (see `docs/decisions.md` D6).

### General code requirements

- [ ] **C1. No Jupyter notebooks** in the repo. Accept: `git ls-files '*.ipynb'` is empty.
- [ ] **C2. Runs on the TU cluster.** Accept: training and evaluation run there, noted in `status.md`.

## Report (report-only, 9 pts, max 10 pages, LaTeX template from ISIS, no code in the report)

The sheet says these points are "guidelines, not a checklist to tick off". They are listed here so
nothing is forgotten, not to be answered one by one.

### Dataset overview (4 pts)
- [ ] **(report) Exploration.** What kinds of SCMs and datasets the prior generates; how different
  observational and interventional data are; what the case studies look like. Examples shown with `show`.
- [ ] **(report) Data quality and splits.** Problems that could distort the evaluation: degenerate
  SCMs, treatment imbalance, lack of overlap between treated and untreated, overlap between training
  and test data.
- [ ] **(report) Preprocessing and representation.** Preprocessing and encoding, *why* we chose them,
  which `transformation` we pass to `load_data`. Can the model handle datasets of different size or shape?
- [ ] **(report) Questions.** In which case studies is the causal effect identifiable? What kind of
  answers could an ML model provide about the data, and what would a human typically ask?

### Baseline and evaluation (3 pts)
- [ ] **(report) Baseline.** Simple baselines (e.g. regression on observational data queried with
  t=0 and t=1, difference of group means) and the first approach from our prototype. *Why* we chose them.
- [ ] **(report) Metrics.** Which metrics are meaningful and why. Why causal effects cannot be
  evaluated on purely observational real-world data.
- [ ] **(report) Results.** Training and validation loss curves of the prototype; results of baselines
  and trained prototype, if possible broken down by case study. Where do they fail?
- [ ] **(report) Overfitting.** Is it an issue? How do we check for it, how do we avoid it?
- [ ] **(report) Non-technical view.** What makes the task easy or hard.

### Discussion (2 pts)
- [ ] **(report) Method.** What the model approximates, and how the architecture and the prior achieve it.
- [ ] **(report) Challenges.** Why the task is not trivial; why a regression model fitted on
  observational data does not answer an interventional question.
- [ ] **(report) Summary.** What does (not) work, and why (not).
- [ ] **(report) Outlook.** What is possible on this data set; what we would like to try next.

### Formal
- [ ] **(report)** ≤ 10 pages (longer reports are not accepted), course LaTeX template from ISIS,
  no code in the report.

## Submission
- [ ] **S1.** All items above ticked or explicitly marked as not achieved (with reason) in `status.md`.
- [ ] **S2.** `pytest` passes on the submitted commit.
- [ ] **S3.** Repo shared with `marvinsxtr`.
- [ ] **S4.** Submitted commit tagged `ms1` and the tag pushed (`git push origin ms1`).
- [ ] **S5.** Report PDF submitted via ISIS.
