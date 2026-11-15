# Milestone 1 status

Newest entries on top. Format: `YYYY-MM-DD (person): what was done / what is next`.
Reference requirement IDs from `requirements.md` (e.g. M1, R3).

## Log

- 2026-11-15 (A and B, on the cluster): D11 checked. The eight converted case studies are in
  `/home/space/datasets/dopfn_case_studies` (100 files each in `Common_Effect`, folders
  `drwxrwxr-x`, written by `cluster/hydra.sh data`, which runs `chmod -R a+rX`).
- 2026-11-15 (B, branch `b/model`, taking over C's part): C's branch `part-c/evaluation` (baselines,
  metrics, harness, `show`, `scripts/evaluate.py`) merged. `evaluate.py` fixed to use the real
  `load_data` interface (it passed `with_graph`, which does not exist), with `--out-dir`, `--device`
  and a summary table; the harness also reports medians over datasets. Cluster: course accounts must
  use the `gpu-teaching-*` partitions; `cluster/hydra.sh evaluate` added. Checked locally with a small
  CPU-trained model on 256 held-out prior datasets and all six case studies. Next: full training
  and evaluation on the cluster (G1, G2, M7, M8, R5, C2), then tick the boxes in `requirements.md`.
- 2026-11-14 (B, branch `b/model`): Model, bar distribution, NLL loss, adapter, training loop,
  `scripts/train.py`, `Regressor` and Hydra job files (`cluster/`) implemented; M1–M6 and R1–R4 done
  (tests in `tests/test_model.py`, `tests/test_training.py`). Proposals for D2, D3, D4 in
  `docs/decisions.md`. Next: smoke test on `gpu-test`, full run on `gpu-teaching-2h` (M7, M8, G1); R5 once
  C's harness exists.
- 2026-11-14 (B, on branch `b/vendor-upstream-and-fix-data`, needs review by A): Do-PFN and
  Do-PFN-prior copies at the repo root completed (decision D7, supervisor's recommendation). Data module
  fixed and tested, D1–D10 covered by `tests/test_data.py` (D8 on real data when
  `$DOPFN_CASE_STUDY_ROOT` is set). Fixes: every prior dataset came from the same SCM; CATE column
  offsets wrong and crash on hidden features; CATE replaced by the shared-noise effect (D8, same as
  Do-PFN); query covariates factual (D5); case studies split into context/query as in Do-PFN and
  read from converted tensor files; all randomness seeded and independent of `PYTHONHASHSEED`;
  `set_seed` and `load_yaml` implemented. Next: A reviews; agree on D1, D5, D6, D8, D9; convert the
  case studies on the cluster (D11).
- 2026-11-05 (B): Paper and sheet PDFs added; `requirements.md` checked against `sheet.pdf`.
- 2026-11-05 (B): Repo skeleton and docs set up. All deliverables are stubs (`NotImplementedError`).
  Next: team agrees on D1–D6, B starts plan step 1.

## Open blockers

- Interface decisions D1–D6 (`docs/decisions.md`) not yet agreed.
