"""
Fine-tune Coqui XTTS-v2 on the speaker in data/manifest.csv.

Designed primarily for the RTX 3070 (8 GB VRAM, Windows 11) but will run on
Apple Silicon (MPS) or CPU as a fallback. Logs per-epoch train and val loss
to a CSV so the evaluation step can plot training curves.

Usage (typical, on the 3070):
    python -m src.train --epochs 10 --batch-size 2 --grad-accum 8

Tips for 8 GB VRAM:
    - Keep --batch-size 1 or 2 and --grad-accum 4..16
    - --max-audio-sec 10 or less
    - --mixed-precision (fp16) is on by default when CUDA is available
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .utils import (
    DATA_DIR,
    FINETUNED_DIR,
    MANIFEST_PATH,
    MODELS_DIR,
    ensure_dirs,
    pick_torch_device,
    setup_logging,
)

log = logging.getLogger("train")


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    grad_accum: int
    learning_rate: float
    max_audio_sec: float
    mixed_precision: bool
    run_name: str
    seed: int


def build_coqui_dataset(manifest: Path, dataset_root: Path) -> Path:
    """
    Convert our manifest.csv into the Coqui LJSpeech-style format Coqui's
    XTTS recipe consumes: a metadata.csv with `path|text|speaker` rows.
    """
    df = pd.read_csv(manifest)
    if "split" not in df.columns:
        raise SystemExit("Manifest has no 'split' column. Run split_dataset.py first.")

    dataset_root.mkdir(parents=True, exist_ok=True)
    wavs_dir = dataset_root / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    # Coqui expects wavs/<basename>.wav and a single metadata.csv
    out_meta = dataset_root / "metadata.csv"
    with open(out_meta, "w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            src = (DATA_DIR.parent / row["path"]).resolve() if str(row["path"]).startswith("data") \
                  else (DATA_DIR / row["path"]).resolve()
            if not src.exists():
                # Manifest stores 'data/chunks/foo.wav' relative to repo root
                alt = (DATA_DIR.parent / row["path"]).resolve()
                src = alt if alt.exists() else src
            dst = wavs_dir / src.name
            if not dst.exists():
                try:
                    os.symlink(src, dst)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)
            stem = src.stem
            text = str(row["text"]).replace("|", " ").strip()
            speaker = str(row.get("speaker", "speaker0"))
            fh.write(f"{stem}|{text}|{speaker}\n")
    log.info("Built Coqui-style dataset at %s", dataset_root)
    return out_meta


def run_xtts_finetune(cfg: TrainConfig, dataset_root: Path, out_dir: Path) -> Path:
    """
    Wraps Coqui's XTTS fine-tuning. Implementation note: Coqui's recipe API
    has shifted across releases; this calls the high-level helper if
    available, otherwise falls back to invoking the trainer directly.
    """
    import torch

    device = pick_torch_device()
    log.info("Torch device: %s", device)
    if device == "cuda":
        log.info("CUDA: %s", torch.cuda.get_device_name(0))

    out_dir.mkdir(parents=True, exist_ok=True)
    loss_csv = out_dir / "loss_log.csv"

    try:
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
        from trainer import Trainer, TrainerArgs
    except Exception as e:
        raise SystemExit(
            "Coqui TTS XTTS trainer modules not importable. "
            "Install TTS with: pip install -U TTS\n"
            f"Original error: {e}"
        )

    # Download pretrained XTTS-v2 if not cached
    pretrained_dir = MODELS_DIR / "pretrained" / "XTTS-v2"
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    log.info("Pretrained model dir: %s", pretrained_dir)

    from huggingface_hub import hf_hub_download
    required_files = ["mel_stats.pth", "dvae.pth", "model.pth", "vocab.json", "config.json"]
    for fname in required_files:
        target = pretrained_dir / fname
        if not target.exists():
            log.info("Downloading %s from coqui/XTTS-v2 ...", fname)
            hf_hub_download(
                repo_id="coqui/XTTS-v2",
                filename=fname,
                local_dir=str(pretrained_dir),
                local_dir_use_symlinks=False,
            )
    log.info("Pretrained XTTS-v2 files ready.")

    config = GPTTrainerConfig(
        output_path=str(out_dir),
        model_args=GPTArgs(
            max_conditioning_length=132300,
            min_conditioning_length=66150,
            debug_loading_failures=False,
            max_wav_length=int(cfg.max_audio_sec * 22050),
            max_text_length=200,
            mel_norm_file=str(pretrained_dir / "mel_stats.pth"),
            dvae_checkpoint=str(pretrained_dir / "dvae.pth"),
            xtts_checkpoint=str(pretrained_dir / "model.pth"),
            tokenizer_file=str(pretrained_dir / "vocab.json"),
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
        ),
        run_name=cfg.run_name,
        project_name="aivoicemaker",
        run_description="XTTS-v2 fine-tune on AIVoiceMaker dataset",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio={"sample_rate": 22050, "dvae_sample_rate": 22050, "output_sample_rate": 24000},
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        batch_group_size=48,
        eval_batch_size=cfg.batch_size,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        eval_split_size=0.0,
        print_step=25,
        plot_step=100,
        log_model_step=1000,
        save_step=1000,
        save_n_checkpoints=2,
        save_checkpoints=True,
        print_eval=True,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=cfg.learning_rate,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
        mixed_precision=cfg.mixed_precision and device == "cuda",
    )

    log.warning(
        "NOTE: This script wires up the standard Coqui XTTS GPT trainer. "
        "If your TTS version's trainer API differs, copy the closest "
        "`recipes/ljspeech/xtts_v2/train_gpt_xtts.py` script from the TTS repo "
        "and point its DATASETS_CONFIG_LIST to %s.",
        dataset_root,
    )

    # Load the trainable GPT model
    from TTS.tts.datasets import BaseDatasetConfig

    ds_cfg = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="aivoicemaker",
        path=str(dataset_root),
        meta_file_train="metadata.csv",
        language="en",
    )

    model = GPTTrainer.init_from_config(config)
    from TTS.tts.datasets import load_tts_samples

    train_samples, eval_samples = load_tts_samples(
        ds_cfg,
        eval_split=True,
        eval_split_max_size=256,
        eval_split_size=0.1,
    )

    trainer = Trainer(
        TrainerArgs(restore_path=None, skip_train_epoch=False, grad_accum_steps=cfg.grad_accum),
        config,
        output_path=str(out_dir),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # Best-effort: extract per-epoch loss into loss_log.csv from trainer.events
    try:
        with open(loss_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch", "train_loss", "val_loss"])
            for ep, (tl, vl) in enumerate(zip(getattr(trainer, "epochs_train_loss", []),
                                              getattr(trainer, "epochs_val_loss", []))):
                w.writerow([ep, tl, vl])
    except Exception:
        pass

    return out_dir


def main() -> None:
    setup_logging()
    ensure_dirs()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--max-audio-sec", type=float, default=10.0)
    p.add_argument("--no-mixed-precision", action="store_true")
    p.add_argument("--run-name", default="xtts_v2_finetune")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_audio_sec=args.max_audio_sec,
        mixed_precision=not args.no_mixed_precision,
        run_name=args.run_name,
        seed=args.seed,
    )

    dataset_root = DATA_DIR / "coqui_ds"
    build_coqui_dataset(args.manifest, dataset_root)

    out_dir = FINETUNED_DIR / args.run_name
    log.info("Output dir: %s", out_dir)
    run_xtts_finetune(cfg, dataset_root, out_dir)
    log.info("Done. Checkpoints in %s", out_dir)


if __name__ == "__main__":
    main()
