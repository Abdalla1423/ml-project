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
- **Affected modules:** data (and interpretation of results in evaluation)

### D6: Train, validation and test split without overlap
- **Status:** OPEN
- **Raised in:** ms1
- **Question:** Split by seed ranges, by SCM, or both? How to guarantee no SCM/dataset is shared
  between training, validation and test? Case studies are test-only?
- **Affected modules:** data, training, evaluation

## Agreed decisions

_None yet._
