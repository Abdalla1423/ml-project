"""Pre-train the Do-PFN prototype on prior data (owner: B).

Usage:
    python scripts/train.py --config configs/train/default.yaml
    python scripts/train.py --config configs/train/default.yaml --resume --max-minutes 290

Writes metrics.csv, loss_curve.png, checkpoint_latest.pt and checkpoint_best.pt to the output
directory (``train.out_dir`` in the config, or ``--out-dir``).
"""

import argparse
from dataclasses import asdict

import yaml

from dopfn.model import Model
from dopfn.training import load_train_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    parser.add_argument("--steps", type=int, default=None, help="Override the number of steps.")
    parser.add_argument("--device", default=None, help="auto, cuda, mps or cpu.")
    parser.add_argument("--out-dir", default=None, help="Override the output directory.")
    parser.add_argument("--num-workers", type=int, default=None, help="Data worker processes.")
    parser.add_argument("--max-minutes", type=float, default=None, help="Stop cleanly after this time.")
    parser.add_argument(
        "--target", choices=["interventional", "observational"], default=None,
        help="Training target: y_in (Do-PFN) or the factual y of the query rows (sanity check).",
    )
    parser.add_argument("--resume", action="store_true", help="Continue from checkpoint_latest.pt.")
    args = parser.parse_args()

    model_config, config = load_train_config(
        args.config,
        seed=args.seed,
        steps=args.steps,
        device=args.device,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        max_minutes=args.max_minutes,
        target=args.target,
    )
    model = Model.from_config(model_config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params / 1e6:.2f}M parameters, {asdict(model_config)}")
    print(yaml.safe_dump({"train": {k: v for k, v in asdict(config).items() if k != "prior"}}, sort_keys=False))
    train(model, config, resume=args.resume)


if __name__ == "__main__":
    main()
