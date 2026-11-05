# Do-PFN replication (TU Berlin Project ML, WiSe 2026/27)

We understand, replicate, present and later extend
[Do-PFN: In-Context Learning for Causal Effect Estimation](https://arxiv.org/abs/2506.06039)
(Robertson et al., NeurIPS 2025). Do-PFN is a TabPFN-style transformer pre-trained on synthetic
structural causal models. It predicts outcomes under an intervention do(t) and the CATE from
observational data, in context.

## Structure

```
src/dopfn/
  common/      shared helpers (seeding, config loading)
  data/        prior with treatment + do(t), load_data, case studies      (Person A)
  model/       transformer, bar distribution, loss, adapter, Regressor    (Person B)
  training/    pre-training loop, checkpoints                             (Person B)
  baselines/   simple and CATE baselines                                  (Person C)
  evaluation/  metrics, evaluation harness                                (Person C)
  viz/         show()                                                     (Person C)
scripts/       command-line entry points
configs/       YAML configs (data/, train/, eval/)
tests/         smoke tests per module
docs/          paper notes, decisions, milestone requirements and status
```

## Roles

- **A, data:** prior with treatment and interventions, data loading, case studies, data quality.
- **B, model and training:** transformer, output distribution, loss, training loop, `Regressor`.
- **C, baselines, evaluation, visualization:** baselines, metrics, evaluation harness, plots.

Each person owns their modules. Changes to someone else's module go through them.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Running

```bash
python scripts/generate_data.py --config configs/data/prior.yaml --out /home/space/datasets/<name>
python scripts/train.py --config configs/train/default.yaml
python scripts/evaluate.py --config configs/eval/default.yaml --checkpoint outputs/train/<ckpt>
```

On the TU cluster, datasets live in `/home/space/datasets/<dataset>` (make them readable with
`chmod -R a+r`). No Jupyter notebooks: everything runs as scripts.

## Docs

- [docs/paper-notes.md](docs/paper-notes.md): the paper, summarized for implementation
- [docs/decisions.md](docs/decisions.md): open and agreed team decisions
- [docs/resources.md](docs/resources.md): external repos and literature
- [docs/milestones/](docs/milestones/): one folder per milestone (requirements, plan, status)

Milestone submissions are tagged `ms1`, `ms2`, ...
