# AIVoiceMaker

End-to-end voice-cloning TTS pipeline. Drop in MP3 voice recordings of one
speaker, the system auto-transcribes them with Whisper, fine-tunes a
pretrained TTS model (Coqui XTTS-v2) on that voice, and exposes a local
Gradio web app where you type text and hear it spoken in the cloned voice.

## Hardware split

- **MacBook (M1 Pro, 32 GB)** — preprocessing, evaluation, the Gradio demo.
- **Gaming PC (Windows 11, RTX 3070 8 GB)** — Whisper transcription + the
  XTTS-v2 fine-tune. ~5–10× faster than the Mac for training.

The two machines share code via this repo; data and checkpoints move
manually. See [`docs/HANDOFF.md`](docs/HANDOFF.md) for the exact workflow.

## Quickstart

```bash
# Mac
pip install -r requirements.txt
# Drop MP3s into data/raw/, then:
python -m src.preprocess

# --- copy data/chunks/ to the PC, then on the PC: ---
pip install -r requirements.txt
pip install -r requirements-cuda.txt
python -m src.transcribe --model large-v3 --compute-type int8_float16
python -m src.split_dataset
python -m src.train --epochs 10 --batch-size 2 --grad-accum 8

# --- copy models/finetuned/ + data/manifest.csv back to the Mac, then: ---
python -m src.evaluate
python app/gradio_app.py
```

## Layout

```
src/
  preprocess.py      # mp3 -> mono 24 kHz VAD-chunked wavs
  transcribe.py      # faster-whisper -> data/manifest.csv
  split_dataset.py   # 90/10 train/val split
  train.py           # XTTS-v2 fine-tune (CUDA preferred, MPS/CPU fallback)
  evaluate.py        # loss curves, speaker similarity, round-trip WER
  infer.py           # text -> wav using the fine-tuned checkpoint
  utils.py           # paths, device picker, logging
app/
  gradio_app.py      # local web UI
docs/
  HANDOFF.md         # Mac <-> PC file movement
data/                # gitignored: raw/, chunks/, manifest.csv
models/              # gitignored: pretrained/, finetuned/
evaluation/          # generated metrics + figures
```

## Estimated run times

For ~7 hours of single-speaker audio:

| Stage | Where | Time |
|---|---|---|
| Preprocess + VAD | Mac | 15–40 min |
| Whisper `large-v3` | PC (3070) | 15–40 min |
| Fine-tune XTTS-v2, 10 epochs | PC (3070) | 1.5–4 h |
| Evaluation | Mac | 10–30 min |
| Inference per sentence | Mac | 2–6 s |

End-to-end is realistically a single afternoon.

## License

Code: see [LICENSE](LICENSE). The Coqui XTTS-v2 model is non-commercial
(CPML); fine for personal/research/homework use.

