"""Generate prior datasets / prepare case studies on disk (owner: A).

Usage: python scripts/generate_data.py --config configs/data/prior.yaml --out /home/space/datasets/<name>
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/prior.yaml")
    parser.add_argument("--out", required=True, help="Output directory.")
    args = parser.parse_args()
    raise NotImplementedError(f"data generation not implemented yet (config={args.config})")


if __name__ == "__main__":
    main()
