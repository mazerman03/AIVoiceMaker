"""
Evaluate the fine-tuned model with three objective metrics:

  1. Loss curves   -- read from training run's loss_log.csv (or trainer logs).
  2. Speaker similarity -- Resemblyzer cosine between real and synthesized clips.
  3. WER (round-trip)   -- synthesize known sentences, transcribe with Whisper,
                            compare to the original text.

All figures + a summary CSV are written to evaluation/figures/.

Usage:
    python -m src.evaluate
    python -m src.evaluate --checkpoint models/finetuned/xtts_v2_finetune
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .utils import (
    CHUNKS_DIR,
    EVAL_DIR,
    FIGURES_DIR,
    FINETUNED_DIR,
    MANIFEST_PATH,
    ensure_dirs,
    setup_logging,
)

log = logging.getLogger("evaluate")

# Held-out prompts (the speaker did NOT say these in training).
HELD_OUT_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is reshaping how we create media.",
    "I am a synthetic voice trained on YouTube recordings.",
    "Today's weather is unusually pleasant for this time of year.",
    "Please save your work before closing the application.",
]


def plot_loss_curves(checkpoint_dir: Path) -> None:
    import matplotlib.pyplot as plt

    csv_path = checkpoint_dir / "loss_log.csv"
    if not csv_path.exists():
        log.warning("No loss_log.csv at %s; skipping loss-curve plot.", csv_path)
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        log.warning("loss_log.csv is empty; skipping plot.")
        return
    plt.figure(figsize=(7, 4))
    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training / validation loss")
    plt.legend()
    plt.grid(alpha=0.3)
    out = FIGURES_DIR / "loss_curves.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("Wrote %s", out)


def speaker_similarity(real_wavs: List[Path], synth_wavs: List[Path]) -> float:
    from resemblyzer import VoiceEncoder, preprocess_wav

    enc = VoiceEncoder()
    real_emb = np.mean([enc.embed_utterance(preprocess_wav(str(p))) for p in real_wavs], axis=0)
    sims = []
    for p in synth_wavs:
        emb = enc.embed_utterance(preprocess_wav(str(p)))
        sims.append(float(np.dot(real_emb, emb) /
                          (np.linalg.norm(real_emb) * np.linalg.norm(emb) + 1e-12)))
    return float(np.mean(sims)) if sims else float("nan")


def round_trip_wer(synth_dir: Path, prompts: List[str]) -> float:
    from faster_whisper import WhisperModel
    import jiwer

    model = WhisperModel("medium", device="cpu", compute_type="int8")
    refs, hyps = [], []
    for i, prompt in enumerate(prompts):
        wav = synth_dir / f"prompt_{i:02d}.wav"
        if not wav.exists():
            continue
        segments, _ = model.transcribe(str(wav), language="en", beam_size=5)
        text = " ".join(s.text for s in segments).strip().lower()
        refs.append(prompt.lower())
        hyps.append(text)
    if not refs:
        return float("nan")
    return jiwer.wer(refs, hyps)


def main() -> None:
    setup_logging()
    ensure_dirs()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--num-val-samples", type=int, default=10)
    args = p.parse_args()

    from .infer import _find_default_checkpoint, synthesize
    import soundfile as sf

    ckpt = args.checkpoint or _find_default_checkpoint()
    if ckpt is None:
        raise SystemExit("No fine-tuned checkpoint found.")
    log.info("Evaluating checkpoint: %s", ckpt)

    plot_loss_curves(ckpt)

    # 1) Synthesize held-out prompts
    synth_dir = EVAL_DIR / "synth"
    synth_dir.mkdir(parents=True, exist_ok=True)
    log.info("Synthesizing %d held-out prompts ...", len(HELD_OUT_PROMPTS))
    for i, prompt in enumerate(HELD_OUT_PROMPTS):
        sr, wav = synthesize(prompt, checkpoint_dir=ckpt)
        sf.write(synth_dir / f"prompt_{i:02d}.wav", wav, sr)

    # 2) Speaker similarity (real reference clips vs. synthesized clips)
    real_wavs = sorted(CHUNKS_DIR.glob("*.wav"))[: args.num_val_samples]
    synth_wavs = sorted(synth_dir.glob("prompt_*.wav"))
    if not real_wavs:
        log.warning("No real reference clips in %s; skipping similarity.", CHUNKS_DIR)
        sim = float("nan")
    else:
        sim = speaker_similarity(real_wavs, synth_wavs)
    log.info("Speaker similarity (cosine, higher is better): %.4f", sim)

    # 3) Round-trip WER on held-out prompts
    wer = round_trip_wer(synth_dir, HELD_OUT_PROMPTS)
    log.info("Round-trip WER (lower is better): %.4f", wer)

    # 4) Validation-split round-trip WER (text the model DID train on but we held back)
    val_wer = float("nan")
    if args.manifest.exists():
        df = pd.read_csv(args.manifest)
        if "split" in df.columns:
            val = df[df["split"] == "val"].head(args.num_val_samples)
            if len(val):
                val_synth_dir = EVAL_DIR / "val_synth"
                val_synth_dir.mkdir(parents=True, exist_ok=True)
                prompts = []
                for i, (_, row) in enumerate(val.iterrows()):
                    sr, wav = synthesize(str(row["text"]), checkpoint_dir=ckpt)
                    sf.write(val_synth_dir / f"prompt_{i:02d}.wav", wav, sr)
                    prompts.append(str(row["text"]))
                val_wer = round_trip_wer(val_synth_dir, prompts)
                log.info("Validation-split round-trip WER: %.4f", val_wer)

    out_csv = EVAL_DIR / "metrics.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["speaker_similarity_cosine", f"{sim:.4f}"])
        w.writerow(["wer_held_out", f"{wer:.4f}"])
        w.writerow(["wer_validation_split", f"{val_wer:.4f}"])
    log.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
