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
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .utils import CHUNKS_DIR, FINETUNED_DIR, pick_torch_device, setup_logging

log = logging.getLogger("infer")

_MODEL_CACHE: dict = {}
_REF_CACHE: dict = {}

# Default XTTS-v2 generation parameters known to work well for voice
# cloning. See https://docs.coqui.ai/en/latest/models/xtts.html
DEFAULT_GEN_PARAMS = {
    "temperature": 0.7,
    "length_penalty": 1.0,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
    "speed": 1.0,
    "enable_text_splitting": True,
}


def _find_default_checkpoint() -> Optional[Path]:
    """Most-recently-modified subdirectory of models/finetuned/."""
    if not FINETUNED_DIR.exists():
        return None
    runs = [d for d in FINETUNED_DIR.iterdir() if d.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def list_reference_chunks(min_seconds: float = 6.0, limit: int = 30) -> List[Path]:
    """Return the longest chunks (>= min_seconds) — good voice references.

    XTTS-v2 conditions on up to 6 s of reference audio per clip; longer
    clips give the model more prosodic context to imitate, which sounds
    less synthetic than a 3-second reference.
    """
    cache_key = (min_seconds, limit)
    if cache_key in _REF_CACHE:
        return _REF_CACHE[cache_key]

    if not CHUNKS_DIR.exists():
        _REF_CACHE[cache_key] = []
        return []

    import soundfile as sf

    candidates = []
    for p in CHUNKS_DIR.glob("*.wav"):
        try:
            dur = sf.info(str(p)).duration
        except Exception:
            continue
        if dur >= min_seconds:
            candidates.append((dur, p))
    candidates.sort(key=lambda x: -x[0])
    chosen = [p for _, p in candidates[:limit]]
    _REF_CACHE[cache_key] = chosen
    return chosen


def _find_default_reference() -> Optional[Path]:
    """A longish, clean chunk to condition the voice on."""
    refs = list_reference_chunks(min_seconds=8.0, limit=1)
    if refs:
        return refs[0]
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
    # Pick the actual fine-tuned checkpoint file (Coqui trainer saves
    # `best_model.pth`, while plain XTTS releases use `model.pth`).
    weight_candidates = [
        ckpt / "best_model.pth",
        ckpt / "model.pth",
    ]
    weight_candidates += sorted(ckpt.glob("checkpoint_*.pth"), reverse=True)
    weight_path = next((p for p in weight_candidates if p.exists()), None)
    if weight_path is None:
        raise FileNotFoundError(f"No model.pth/best_model.pth/checkpoint_*.pth in {ckpt}")
    vocab_path = ckpt / "vocab.json"
    model.load_checkpoint(
        config,
        checkpoint_path=str(weight_path),
        vocab_path=str(vocab_path) if vocab_path.exists() else None,
        use_deepspeed=False,
    )
    device = pick_torch_device()
    if device == "cuda":
        model.cuda()
    log.info("Model loaded on %s", device)
    _MODEL_CACHE[key] = (model, config)
    return model, config


PathOrPaths = Union[Path, str, Sequence[Union[Path, str]], None]


def _coerce_refs(reference_wav: PathOrPaths) -> List[str]:
    if reference_wav is None:
        return []
    if isinstance(reference_wav, (str, Path)):
        return [str(reference_wav)]
    return [str(p) for p in reference_wav]


def synthesize(
    text: str,
    reference_wav: PathOrPaths = None,
    checkpoint_dir: Optional[Path] = None,
    language: str = "en",
    *,
    temperature: float = DEFAULT_GEN_PARAMS["temperature"],
    length_penalty: float = DEFAULT_GEN_PARAMS["length_penalty"],
    repetition_penalty: float = DEFAULT_GEN_PARAMS["repetition_penalty"],
    top_k: int = DEFAULT_GEN_PARAMS["top_k"],
    top_p: float = DEFAULT_GEN_PARAMS["top_p"],
    speed: float = DEFAULT_GEN_PARAMS["speed"],
    enable_text_splitting: bool = DEFAULT_GEN_PARAMS["enable_text_splitting"],
) -> Tuple[int, np.ndarray]:
    """Return (sample_rate, waveform) for the given text.

    `reference_wav` accepts either a single path or a list of paths.
    Passing several references gives XTTS a richer speaker embedding and
    typically sounds noticeably less synthetic.
    """
    model, config = load_model(checkpoint_dir)
    refs = _coerce_refs(reference_wav)
    if not refs:
        default = _find_default_reference()
        if default is None:
            raise FileNotFoundError(
                "No reference audio available. Place at least one .wav in data/chunks/ "
                "or pass --reference."
            )
        refs = [str(default)]

    log.info(
        "Synthesizing %d chars (refs=%d, T=%.2f, top_p=%.2f)",
        len(text), len(refs), temperature, top_p,
    )
    out = model.synthesize(
        text=text,
        config=config,
        speaker_wav=refs if len(refs) > 1 else refs[0],
        language=language,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        speed=speed,
        enable_text_splitting=enable_text_splitting,
    )
    wav = out["wav"] if isinstance(out, dict) else out
    sr = getattr(config, "output_sample_rate", 24000)
    return sr, np.asarray(wav, dtype=np.float32)


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", required=True)
    p.add_argument("--out", type=Path, default=Path("out.wav"))
    p.add_argument(
        "--reference",
        type=Path,
        nargs="+",
        default=None,
        help="One or more reference WAVs. If omitted, a long clip from data/chunks/ is used.",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--language", default="en")
    p.add_argument("--temperature", type=float, default=DEFAULT_GEN_PARAMS["temperature"])
    p.add_argument("--top-p", type=float, default=DEFAULT_GEN_PARAMS["top_p"])
    p.add_argument("--top-k", type=int, default=DEFAULT_GEN_PARAMS["top_k"])
    p.add_argument("--repetition-penalty", type=float, default=DEFAULT_GEN_PARAMS["repetition_penalty"])
    p.add_argument("--speed", type=float, default=DEFAULT_GEN_PARAMS["speed"])
    args = p.parse_args()

    import soundfile as sf

    sr, wav = synthesize(
        args.text,
        reference_wav=args.reference,
        checkpoint_dir=args.checkpoint,
        language=args.language,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        speed=args.speed,
    )
    sf.write(str(args.out), wav, sr)
    log.info("Wrote %s (%.2fs @ %d Hz)", args.out, len(wav) / sr, sr)


if __name__ == "__main__":
    main()
