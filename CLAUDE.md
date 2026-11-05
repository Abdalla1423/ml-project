# CLAUDE.md

## Current milestone: `ms1` (Data and Prototype)

Before starting any work, read:
1. `docs/milestones/ms1/requirements.md` (what must be delivered, as a checklist)
2. `docs/milestones/ms1/plan.md` (team split, dependencies)
3. `docs/milestones/ms1/status.md` (what is done, what is next)
4. `docs/decisions.md` (open and agreed interface decisions)

Read `docs/paper-notes.md` whenever paper details matter (prior, Algorithm 1, model, evaluation).
External links are in `docs/resources.md`. The paper PDF is in `docs/papers/`.

## Project

TU Berlin "Project ML" (WiSe 2026/27), group of three. We are understanding, replicating, presenting
and later extending **Do-PFN: In-Context Learning for Causal Effect Estimation** (Robertson et al.,
NeurIPS 2025, arXiv:2506.06039). Do-PFN is a TabPFN-style transformer pre-trained on synthetic SCMs:
the context is observational data and the targets are interventional outcomes, so it learns to predict
p(y | do(t), x) and the CATE in context.

The project is split into milestones, each with a sheet, code deliverables and a report.
- Code goes to the GitHub repo shared with the supervisor (GitHub handle `marvinsxtr`).
- Each submitted commit is tagged `ms1`, `ms2`, ...
- Code must run on the TU cluster. Datasets live in `/home/space/datasets/<dataset>` (`chmod -R a+r`).
- **No Jupyter notebooks.**

## Roles and module ownership

| Person | Area | Owns |
|---|---|---|
| A | Data | `src/dopfn/data/`, `scripts/generate_data.py`, `configs/data/` |
| B | Model and training | `src/dopfn/model/`, `src/dopfn/training/`, `scripts/train.py`, `configs/train/` |
| C | Baselines, evaluation, visualization | `src/dopfn/baselines/`, `src/dopfn/evaluation/`, `src/dopfn/viz/`, `scripts/evaluate.py` |

Shared: `src/dopfn/common/`, `pyproject.toml`, `README.md`, `docs/`. Tests live in `tests/test_<module>.py`
and belong to the module's owner.

**Respect ownership.** If a task needs a change in a module owned by someone else, propose the change
and tell the person you are working with. Do not edit it silently. Roles may change per milestone,
`docs/milestones/<current>/plan.md` says what applies.

**Personal context stays local.** Each team member keeps their own role, personal plan and
preferences in `CLAUDE.local.md` at the repo root (gitignored, loaded automatically by Claude Code).
Shared files in the repo describe the team, not one person.

## Coding conventions

- Scripts runnable from the command line (`python scripts/<name>.py --config ...`), on the TU cluster.
- Type hints and short docstrings everywhere.
- Configs via YAML (`configs/`) loaded into dataclasses. No magic numbers spread across files.
- Seeds controllable everywhere (`dopfn.common.seed.set_seed`, explicit `seed` / `torch.Generator` args).
- Model code must be differentiable and vectorized. No Python loops over rows.
- Interface choices not agreed with the team: mark `TODO(team)` in code and log them in `docs/decisions.md`.
- Do not copy external repos into the codebase. Propose a git submodule or pip dependency instead.
- Do not assume how external repos are structured. Inspect them before relying on any API.

## Repo layout

```
CLAUDE.md, README.md, pyproject.toml
docs/
  paper-notes.md, resources.md, decisions.md
  papers/Do-PFN.pdf
  milestones/README.md, ms1/{sheet.pdf, requirements.md, plan.md, status.md}
src/dopfn/
  common/      shared helpers (seeding, config loading)
  data/        A: prior with treatment + do(t), load_data, case studies
  model/       B: transformer, bar distribution, loss, adapter, Regressor
  training/    B: pre-training loop, checkpoints
  baselines/   C: simple and CATE baselines
  evaluation/  C: metrics, evaluation harness
  viz/         C: show()
scripts/       entry points: train.py, evaluate.py, generate_data.py
configs/       data/, train/, eval/ YAML configs
tests/         smoke tests per module
```

## Working rules

- Update `docs/milestones/<current>/status.md` when a piece of work is finished.
- Record agreed decisions in `docs/decisions.md` with date, reason and affected modules.
- Run `pytest` before pushing. `main` must stay green.
- Work on feature branches and merge into `main` via pull request; do not push personal work to `main`.
- Commit only when the person you work with asks.
