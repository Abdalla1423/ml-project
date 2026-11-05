"""Pre-train the Do-PFN prototype on prior data (owner: B).

Usage: python scripts/train.py --config configs/train/default.yaml
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/default.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    args = parser.parse_args()
    raise NotImplementedError(f"training not implemented yet (config={args.config})")


if __name__ == "__main__":
    main()
