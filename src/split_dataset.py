"""
Add a train/val 'split' column to data/manifest.csv (90/10 by default,
fixed seed for reproducibility).

Usage:
    python -m src.split_dataset
    python -m src.split_dataset --val-frac 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .utils import MANIFEST_PATH, setup_logging

log = logging.getLogger("split")


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.manifest.exists():
        log.error("Manifest not found: %s. Run transcribe.py first.", args.manifest)
        return

    df = pd.read_csv(args.manifest)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n_val = max(1, int(len(df) * args.val_frac))
    df["split"] = "train"
    df.loc[: n_val - 1, "split"] = "val"
    df.to_csv(args.manifest, index=False)
    log.info("Wrote split: %d train / %d val (total %d)",
             (df["split"] == "train").sum(), n_val, len(df))


if __name__ == "__main__":
    main()
