# Milestone 1 plan

## Team split

- **A (data).** Add binary treatment and $do(t)$ to Do-PFN-prior, generate observational and
  interventional data, ground-truth CATE, `load_data` (prior and case studies), data quality checks.
  Code: `src/dopfn/data/`, `scripts/generate_data.py`, `configs/data/`.
- **B (model and training).** Simplified Do-PFN based on nanoTabPFN, `forward`, bar
  distribution, NLL loss, training loop, convergence, loss curves, `Regressor`.
  Code: `src/dopfn/model/`, `src/dopfn/training/`, `scripts/train.py`, `configs/train/`.
- **C (baselines, evaluation, viz).** `show()`, baselines, metrics, evaluation harness for any object
  with `fit`/`predict`/`predict_cate`, comparison across case studies, overfitting analysis.
  Code: `src/dopfn/baselines/`, `src/dopfn/evaluation/`, `src/dopfn/viz/`, `scripts/evaluate.py`.

Personal working plans live in each person's `CLAUDE.local.md` (not in the repo).

## Dependencies between people

- B starts on a temporary data source until A's `load_data` exists. All data reaches the model through
  one adapter, `dopfn.model.adapter.to_model_batch`, so a change in A's yield format only touches
  that file.
- C's harness needs B's `Regressor` interface (fixed by the sheet) and A's yield format (D1).
- Open decisions blocking integration: D1–D6 in `docs/decisions.md`.
