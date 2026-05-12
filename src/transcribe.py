"""
Auto-transcribe every WAV in data/chunks/ with faster-whisper and write
data/manifest.csv with columns: path,text,speaker,duration.

Auto-detects CUDA (RTX 3070) vs CPU/MPS. On the 3070 this runs at large-v3
in ~15-40 min for ~7 h of audio. On the Mac, prefer --model medium.

Usage:
    # On the 3070 (Windows):
    python -m src.transcribe --model large-v3 --compute-type int8_float16

    # On the Mac (fallback):
    python -m src.transcribe --model medium --compute-type int8

    # Force a language (default: auto-detect):
    python -m src.transcribe --language en
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import soundfile as sf

from .utils import CHUNKS_DIR, MANIFEST_PATH, ensure_dirs, setup_logging

log = logging.getLogger("transcribe")


def clean_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> None:
    setup_logging()
    ensure_dirs()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunks-dir", type=Path, default=CHUNKS_DIR)
    p.add_argument("--out", type=Path, default=MANIFEST_PATH)
    p.add_argument("--model", default="large-v3",
                   help="faster-whisper model name (tiny|base|small|medium|large-v3)")
    p.add_argument("--compute-type", default=None,
                   help="auto by default. Try int8_float16 on CUDA, int8 on CPU.")
    p.add_argument("--device", default=None,
                   help="cuda | cpu | auto (default: auto)")
    p.add_argument("--language", default=None,
                   help="ISO 639-1 code (e.g. en). Default: auto-detect.")
    p.add_argument("--speaker", default="speaker0",
                   help="Speaker label written to manifest.")
    args = p.parse_args()

    from faster_whisper import WhisperModel

    device = args.device
    if device in (None, "auto"):
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    compute_type = args.compute_type or ("int8_float16" if device == "cuda" else "int8")
    log.info("Loading faster-whisper %s on %s (%s) ...", args.model, device, compute_type)
    model = WhisperModel(args.model, device=device, compute_type=compute_type)

    wavs = sorted(args.chunks_dir.glob("*.wav"))
    if not wavs:
        log.error("No WAV chunks found in %s. Run preprocess.py first.", args.chunks_dir)
        return
    log.info("Transcribing %d chunks ...", len(wavs))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "text", "speaker", "duration", "language"])
        for i, wav in enumerate(wavs, 1):
            info = sf.info(str(wav))
            duration = info.frames / float(info.samplerate)
            try:
                segments, det = model.transcribe(
                    str(wav),
                    language=args.language,
                    beam_size=5,
                    vad_filter=False,
                )
                text = clean_text(" ".join(s.text for s in segments))
                lang = det.language
            except Exception as e:
                log.warning("  %s failed: %s", wav.name, e)
                continue
            if not text:
                continue
            w.writerow([
                str(wav.relative_to(args.chunks_dir.parent)),
                text, args.speaker, f"{duration:.3f}", lang,
            ])
            if i % 25 == 0 or i == len(wavs):
                log.info("  %d/%d done", i, len(wavs))

    log.info("Wrote manifest: %s", args.out)


if __name__ == "__main__":
    main()
