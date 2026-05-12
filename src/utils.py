"""Shared helpers: device selection, paths, logging."""
from __future__ import annotations

import logging
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CHUNKS_DIR = DATA_DIR / "chunks"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
MODELS_DIR = REPO_ROOT / "models"
FINETUNED_DIR = MODELS_DIR / "finetuned"
EVAL_DIR = REPO_ROOT / "evaluation"
FIGURES_DIR = EVAL_DIR / "figures"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def pick_torch_device() -> str:
    """Return the best available torch device string: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_dirs() -> None:
    for d in (RAW_DIR, CHUNKS_DIR, MODELS_DIR, FINETUNED_DIR, EVAL_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
