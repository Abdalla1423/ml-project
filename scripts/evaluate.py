"""Evaluate the prototype and baselines on prior data and case studies (owner: C).

Usage: python scripts/evaluate.py --config configs/eval/default.yaml --checkpoint <path>
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Trained model checkpoint.")
    args = parser.parse_args()
    raise NotImplementedError(f"evaluation not implemented yet (config={args.config})")


if __name__ == "__main__":
    main()
