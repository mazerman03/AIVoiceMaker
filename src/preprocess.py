"""
Preprocess raw MP3s in data/raw/ into clean, VAD-segmented WAV chunks
suitable for TTS fine-tuning.

Output: data/chunks/<source-stem>_<idx>.wav (mono, 24 kHz, loudness-normalized).

Usage:
    python -m src.preprocess
    python -m src.preprocess --target-sr 24000 --min-sec 2.5 --max-sec 12.0

Run this on the Mac before sending data/chunks/ to the PC.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .utils import CHUNKS_DIR, RAW_DIR, ensure_dirs, setup_logging

log = logging.getLogger("preprocess")

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    """Load any audio file, downmix to mono, resample to target_sr."""
    import librosa

    wav, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype(np.float32)


def load_silero_vad():
    log.info("Loading Silero VAD via torch.hub ...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


def vad_segments(
    wav: np.ndarray,
    sr: int,
    model,
    utils,
    min_sec: float,
    max_sec: float,
) -> list[tuple[int, int]]:
    """Return list of (start_sample, end_sample) speech segments."""
    (get_speech_timestamps, _, _, _, _) = utils
    # Silero expects 16kHz tensor
    import librosa

    wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    tensor = torch.from_numpy(wav16)
    ts = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=int(min_sec * 1000),
        max_speech_duration_s=max_sec,
        min_silence_duration_ms=300,
        speech_pad_ms=120,
    )
    # Map back to original sample rate
    scale = sr / 16000
    return [(int(t["start"] * scale), int(t["end"] * scale)) for t in ts]


def loudness_normalize(wav: np.ndarray, sr: int, target_lufs: float = -23.0) -> np.ndarray:
    import pyloudnorm as pyln

    meter = pyln.Meter(sr)
    try:
        loudness = meter.integrated_loudness(wav)
    except ValueError:
        # Too short for ITU loudness; fall back to peak normalization
        peak = float(np.max(np.abs(wav)) or 1.0)
        return wav * (0.9 / peak)
    out = pyln.normalize.loudness(wav, loudness, target_lufs)
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def process_file(
    src: Path,
    out_dir: Path,
    target_sr: int,
    min_sec: float,
    max_sec: float,
    vad_model,
    vad_utils,
) -> int:
    log.info("Loading %s", src.name)
    wav = load_audio_mono(src, target_sr)
    segs = vad_segments(wav, target_sr, vad_model, vad_utils, min_sec, max_sec)
    log.info("  %d speech segments", len(segs))
    written = 0
    for i, (start, end) in enumerate(segs):
        clip = wav[start:end]
        if len(clip) < int(min_sec * target_sr):
            continue
        clip = loudness_normalize(clip, target_sr)
        out = out_dir / f"{src.stem}_{i:05d}.wav"
        sf.write(str(out), clip, target_sr, subtype="PCM_16")
        written += 1
    log.info("  wrote %d chunks", written)
    return written


def main() -> None:
    setup_logging()
    ensure_dirs()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=RAW_DIR)
    p.add_argument("--output-dir", type=Path, default=CHUNKS_DIR)
    p.add_argument("--target-sr", type=int, default=24000)
    p.add_argument("--min-sec", type=float, default=2.5)
    p.add_argument("--max-sec", type=float, default=12.0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        f for f in args.input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS
    )
    if not files:
        log.error("No audio files found in %s", args.input_dir)
        return

    log.info("Found %d source files", len(files))
    vad_model, vad_utils = load_silero_vad()

    total = 0
    for f in files:
        total += process_file(
            f, args.output_dir, args.target_sr, args.min_sec, args.max_sec,
            vad_model, vad_utils,
        )
    log.info("Done. Wrote %d total chunks to %s", total, args.output_dir)


if __name__ == "__main__":
    main()
