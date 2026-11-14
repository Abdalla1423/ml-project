"""Generate prior datasets / convert the Do-PFN case studies to plain tensor files (owner: A).

Usage:
    # Held-out prior datasets (e.g. a fixed test set), one .pt file per dataset:
    python scripts/generate_data.py prior --config configs/data/prior.yaml \
        --out /home/space/datasets/dopfn_prior_test --n-datasets 256 --seed 1000

    # Case studies: convert Do-PFN's pickles (needs `pip install -e ".[case-studies]"` and a clone
    # of https://github.com/jr2021/Do-PFN for its data/ folder):
    python scripts/generate_data.py case-studies --source <Do-PFN>/data/prior_sampling \
        --out /home/space/datasets/dopfn_case_studies

On the cluster, make the output readable for everyone: ``chmod -R a+r <out>``.
"""

import argparse
from pathlib import Path

import torch

from dopfn.common.config import load_yaml
from dopfn.data.case_studies import (
    CASE_STUDY_DIRECTORIES,
    convert_do_pfn_pickle,
    save_case_study_file,
)
from dopfn.data.load import load_data
from dopfn.data.prior import PriorConfig


def generate_prior(config_path: Path, out: Path, n_datasets: int, seed: int | None) -> None:
    """Write ``n_datasets`` prior datasets as ``prior_<i>.pt`` (dicts of ``CausalDataset`` fields)."""
    raw_config = load_yaml(config_path)
    seed = raw_config.get("seed", 0) if seed is None else seed
    config = PriorConfig.from_dict(raw_config)
    out.mkdir(parents=True, exist_ok=True)
    for i, data in enumerate(load_data("prior", seed=seed, n_datasets=n_datasets, config=config)):
        torch.save(data._asdict(), out / f"prior_{i}.pt")
    print(f"Wrote {n_datasets} prior datasets (seed {seed}) to {out}")


def convert_case_studies(source: Path, out: Path) -> None:
    """Convert every ``<Case_Study>/<Case_Study>_<i>.pkl`` under ``source`` to ``.pt``."""
    for name, directory in CASE_STUDY_DIRECTORIES.items():
        files = sorted((source / directory).glob("*.pkl"))
        if not files:
            print(f"Skipping {name}: no pickles in {source / directory}")
            continue
        for pkl in files:
            save_case_study_file(convert_do_pfn_pickle(pkl), out / directory / f"{pkl.stem}.pt")
        print(f"Converted {len(files)} files of {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    prior = sub.add_parser("prior", help="Sample prior datasets and save them.")
    prior.add_argument("--config", type=Path, default=Path("configs/data/prior.yaml"))
    prior.add_argument("--out", type=Path, required=True, help="Output directory.")
    prior.add_argument("--n-datasets", type=int, required=True)
    prior.add_argument("--seed", type=int, default=None, help="Base seed (default: from config).")

    cases = sub.add_parser("case-studies", help="Convert Do-PFN case-study pickles.")
    cases.add_argument("--source", type=Path, required=True, help="<Do-PFN>/data/prior_sampling")
    cases.add_argument("--out", type=Path, required=True, help="Output root directory.")

    args = parser.parse_args()
    if args.command == "prior":
        generate_prior(args.config, args.out, args.n_datasets, args.seed)
    else:
        convert_case_studies(args.source, args.out)


if __name__ == "__main__":
    main()
