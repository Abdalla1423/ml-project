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
  data/        prior with treatment + do(t), load_data, case studies
  model/       transformer, bar distribution, loss, adapter, Regressor
  training/    pre-training loop, checkpoints
  baselines/   simple and CATE baselines
  evaluation/  metrics, evaluation harness
  viz/         show()
scripts/       command-line entry points
configs/       YAML configs (data/, train/, eval/)
cluster/       Hydra cluster: container recipe and hydra.sh (all cluster steps)
tests/         smoke tests per module
docs/          paper notes, decisions, milestone requirements and status
Do-PFN/, Do-PFN-prior/   copies of the upstream repos (see docs/third-party.md)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # installs dopfn and the copied Do-PFN-prior package dopfnprior
pytest
```

## Data

The prior is sampled on the fly: `load_data("prior", seed=0)` (see `dopfn.data`).

The Do-PFN case studies are stored as `dill` pickles in the Do-PFN repo. We convert them once into
plain tensor files, which `load_data("<case_study>")` reads without running any pickled code:

```bash
pip install -e ".[dev,case-studies]"
git clone https://github.com/jr2021/Do-PFN /tmp/Do-PFN    # only its data/ folder is needed
python scripts/generate_data.py case-studies --source /tmp/Do-PFN/data/prior_sampling \
    --out /home/space/datasets/dopfn_case_studies
chmod -R a+r /home/space/datasets/dopfn_case_studies
```

Locally, convert into any folder and set `DOPFN_CASE_STUDY_ROOT=<folder>`. With that variable set,
`pytest` also checks the real case studies.

## Running

```bash
python scripts/generate_data.py --config configs/data/prior.yaml --out /home/space/datasets/<name>
python scripts/train.py --config configs/train/default.yaml
python scripts/evaluate.py --config configs/eval/default.yaml --checkpoint outputs/train/<ckpt>
```

On the TU cluster, datasets live in `/home/space/datasets/<dataset>` (make them readable with
`chmod -R a+r`). No Jupyter notebooks: everything runs as scripts.

## Training

```bash
python scripts/train.py --config configs/train/debug.yaml     # ~1 min on a laptop, sanity check
python scripts/train.py --config configs/train/default.yaml   # ~3 h on one A100
```

Outputs go to `train.out_dir` of the config: `metrics.csv` (training and validation NLL, outcome
and CATE MSE of the model and of the difference of group means), `loss_curve.png`,
`checkpoint_latest.pt` and `checkpoint_best.pt`. Load a trained model with
`dopfn.model.Regressor.from_checkpoint(path)`.

### On the Hydra cluster

Step by step in [cluster/README.md](cluster/README.md). In short:

```bash
bash cluster/hydra.sh setup     # once: build the container
bash cluster/hydra.sh test      # pytest on a CPU node
bash cluster/hydra.sh data      # case studies -> /home/space/datasets/dopfn_case_studies
bash cluster/hydra.sh smoke     # 10-minute GPU training test
bash cluster/hydra.sh train     # full run; run again to continue after the 5 h limit
bash cluster/hydra.sh status    # jobs and newest log
```

## Docs

- [docs/paper-notes.md](docs/paper-notes.md): the paper, summarized for implementation
- [docs/decisions.md](docs/decisions.md): open and agreed team decisions
- [docs/resources.md](docs/resources.md): external repos and literature
- [docs/milestones/](docs/milestones/): one folder per milestone (requirements, plan, status)

Milestone submissions are tagged `ms1`, `ms2`, ...
