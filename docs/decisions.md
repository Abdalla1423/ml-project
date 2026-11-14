# Decision log

Interface and design decisions that affect more than one person. Code that depends on an open
decision is marked `TODO(team)` with the decision ID, e.g. `TODO(team): D2`.

Status values: `OPEN` (not agreed), `AGREED`, `SUPERSEDED` (link the replacement).

## Template

```markdown
### DN: <short title>
- **Status:** AGREED
- **Date:** YYYY-MM-DD
- **Decision:** <what we do>
- **Reason:** <why, alternatives considered>
- **Affected modules:** <e.g. data, model, evaluation>
- **Raised in:** msN
```

## Open decisions

### D1: Tuple format and tensor shapes from `load_data`
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** Exact contents and order of each yielded sample, dtype (numpy vs torch), and shapes.
  Sheet minimum: `(X_ob, y_ob, X_in, y_in)` with treatment in column 0 of X, plus observational
  outcomes of the query points (for Dont-PFN) and ground-truth CATE. Batch shape for the model,
  e.g. `(batch, rows, features)`.
- **Proposal (implemented on `b/vendor-upstream-and-fix-data`, 2026-11-14):** each item is a
  `dopfn.data.CausalDataset` NamedTuple `(X_ob, y_ob, X_in, y_in, y_in_ob, cate)` of CPU `float32`
  torch tensors, `X` of shape `(n, F)`, all `y`/`cate` of shape `(n,)`, treatment in column 0.
  No batch dimension: batching and padding belong to D2. Covariates are the visible ancestors of `y`
  (Do-PFN-prior also uses descendants of `y`).
- **Affected modules:** data, model, training, evaluation

### D2: Padding of datasets with different numbers of rows and features in a batch
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** Pad features to a fixed max with zeros (TabPFN style)? Batch only datasets of equal
  row counts, or pad rows and mask them? Who pads (loader collate fn vs model adapter)?
- **Affected modules:** data, model, training

### D3: How the observational/query split is passed to the model
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** The sheet's `forward(X, y_ob)` implies the split index is `y_ob.shape[1]`
  (rows of `X` after that are queries). Is that enough, or do we pass an explicit index / mask
  (needed if rows are padded, see D2)?
- **Affected modules:** model, training, evaluation

### D4: Normalization of x and y, and who does it
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** Standardize per dataset using observational statistics only? Done by
  `transformation` in `load_data` or inside the model/Regressor? The bar distribution bin borders
  depend on the y scale, and predictions must be mapped back to the original scale for metrics.
- **Affected modules:** data, model, evaluation

### D5: Query covariates x^in: pre-treatment values with shared noise
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** Follow the paper (x^in from the unintervened SCM, y^in from the intervened SCM with
  the same noise; mediators in x^in keep their factual values)? See `docs/paper-notes.md`.
- **Proposal (implemented):** yes, follow the paper. Do-PFN's own evaluation does the same
  (`TabularDataset.generate_valid_split` uses `x_obs` with only column 0 replaced). Tested in
  `tests/test_data.py::test_mediator_keeps_factual_value_in_queries`.
- **Affected modules:** data (and interpretation of results in evaluation)

### D6: Train, validation and test split without overlap
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** Split by seed ranges, by SCM, or both? How to guarantee no SCM/dataset is shared
  between training, validation and test? Case studies are test-only?
- **Proposal (implemented):** prior dataset `i` of a stream with base seed `s` uses
  `derive_seed(s, i)` (NumPy `SeedSequence`), so streams with different base seeds never share a
  dataset. Use e.g. seed 0 for training, 1 for validation, 1000 for the test set
  (`configs/eval/default.yaml`). Case studies are test-only.
- **Affected modules:** data, training, evaluation

### D8: Ground-truth CATE
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** The sheet defines CATE = E[y | do(1), x] − E[y | do(0), x]. On synthetic data we
  can compute it exactly only per unit.
- **Proposal (implemented):** `cate` of a query row is `y(do(1), ε) − y(do(0), ε)` with the row's
  own noise ε. This is exactly how the Do-PFN case studies define it (checked on all 8 case
  studies: `y_int − y_obs = (t_int − t_obs) · cate`), so prior and case studies are comparable.
  Its expectation given x is the CATE. The earlier Monte Carlo estimate (fix the visible pre-treatment
  covariates, resample all other noise) computed E[y | do(t), do(x)], which is wrong when a hidden
  variable affects both x and y.
- **Affected modules:** data, evaluation

### D9: Dataset locations on the cluster
- **Status:** OPEN
- **Raised in:** ms1
- **Proposal:** converted case studies in `/home/space/datasets/dopfn_case_studies/<Case_Study>/`
  (override with `$DOPFN_CASE_STUDY_ROOT`), fixed prior test sets in
  `/home/space/datasets/dopfn_prior_<split>/`. Both written by `scripts/generate_data.py`,
  then `chmod -R a+r`.
- **Affected modules:** data, evaluation

## Agreed decisions

### D7: External code is copied into the repo
- **Status:** AGREED
- **Date:** 2026-11-14
- **Decision:** External code may be copied into the repo (one top-level folder per repo) and changed or combined as needed.
  Do-PFN and Do-PFN-prior are copied (code only) at fixed commits; `dopfnprior` is installed by our
  `pyproject.toml`. For each copy, `docs/third-party.md` records source and commit, and changes
  are marked `MODIFIED (dopfn replication)`. Large binaries (checkpoints, pretrained model) and data
  stay out of git (size; data goes to the cluster as the sheet asks); notebooks stay out because the
  sheet forbids submitting them.
- **Reason:** Recommended by the supervisor; the sheet asks us to start from Do-PFN-prior and to base
  the training loop on it. The earlier rule "do not copy external repos" was our own and is not in
  the sheet; it is withdrawn. A pip/git dependency on Do-PFN-prior also pulled in TFM-Playground just
  for `get_default_device`, and the case-study pickles need the Do-PFN code to be importable.
- **Affected modules:** all
- **Raised in:** ms1
