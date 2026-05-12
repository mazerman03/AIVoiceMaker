"""
Inference: turn text into a WAV using the fine-tuned XTTS-v2 checkpoint.

Usage (CLI):
    python -m src.infer --text "Hello world" --out out.wav
    python -m src.infer --text "..." --reference data/chunks/some_chunk.wav

The Gradio app (app/gradio_app.py) imports `synthesize` from this module.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .utils import CHUNKS_DIR, FINETUNED_DIR, pick_torch_device, setup_logging

log = logging.getLogger("infer")

_MODEL_CACHE: dict = {}


def _find_default_checkpoint() -> Optional[Path]:
    """Most-recently-modified subdirectory of models/finetuned/."""
    if not FINETUNED_DIR.exists():
        return None
    runs = [d for d in FINETUNED_DIR.iterdir() if d.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def _find_default_reference() -> Optional[Path]:
    """A short, clean chunk to condition the voice on."""
    if not CHUNKS_DIR.exists():
        return None
    wavs = sorted(CHUNKS_DIR.glob("*.wav"))
    return wavs[0] if wavs else None


def load_model(checkpoint_dir: Optional[Path] = None):
    """Lazy-load XTTS-v2 from a fine-tuned checkpoint folder."""
    key = str(checkpoint_dir or "default")
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    ckpt = checkpoint_dir or _find_default_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(
            f"No fine-tuned checkpoint found under {FINETUNED_DIR}. "
            "Train first or pass --checkpoint."
        )

    config_path = ckpt / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {ckpt}")

    log.info("Loading XTTS-v2 from %s ...", ckpt)
    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(ckpt), use_deepspeed=False)
    device = pick_torch_device()
    if device == "cuda":
        model.cuda()
    log.info("Model loaded on %s", device)
    _MODEL_CACHE[key] = (model, config)
    return model, config


def synthesize(
    text: str,
    reference_wav: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    language: str = "en",
) -> Tuple[int, np.ndarray]:
    """Return (sample_rate, waveform) for the given text."""
    model, config = load_model(checkpoint_dir)
    ref = reference_wav or _find_default_reference()
    if ref is None:
        raise FileNotFoundError(
            "No reference audio available. Place at least one .wav in data/chunks/ "
            "or pass --reference."
        )

    log.info("Synthesizing %d chars (ref=%s)", len(text), ref.name)
    out = model.synthesize(
        text=text,
        config=config,
        speaker_wav=str(ref),
        language=language,
    )
    wav = out["wav"] if isinstance(out, dict) else out
    sr = getattr(config, "output_sample_rate", 24000)
    return sr, np.asarray(wav, dtype=np.float32)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", required=True)
    p.add_argument("--out", type=Path, default=Path("out.wav"))
    p.add_argument("--reference", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--language", default="en")
    args = p.parse_args()

    import soundfile as sf

    sr, wav = synthesize(args.text, args.reference, args.checkpoint, args.language)
    sf.write(str(args.out), wav, sr)
    log.info("Wrote %s (%.2fs @ %d Hz)", args.out, len(wav) / sr, sr)


if __name__ == "__main__":
    main()
