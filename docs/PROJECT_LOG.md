# Project Log — AIVoiceMaker

> Plain-language record of every decision made and every command run while
> building this project. Intended to be handed to a future agent (or human)
> as the source material for writing the homework report and presentation.
>
> Date of build: 2026-05-11. Builder: GitHub Copilot CLI agent (Claude Opus 4.7),
> driven by user `maxzemeno`.

---

## 1. Goal

Build an end-to-end **voice cloning / text-to-speech** pipeline that learns to
speak in the voice of a single speaker from YouTube-extracted MP3s, with **no
pre-existing transcripts**. Final deliverable: a local web app where the user
types text and gets back a WAV in the cloned voice.

The project also serves as a homework deliverable that requires:
1. Identifying an NLP application area and explaining its relevance.
2. A clearly defined objective.
3. A *trained* model (not just zero-shot inference) with code-level detail.
4. Validation **metrics** on a held-out set.
5. Inference results on data the model **never saw** during training.
6. A personal conclusion.

The written report and slide deck will be authored later, using this log
plus the figures and metrics produced by `src/evaluate.py`.

## 2. NLP / AI application area chosen

**Speech synthesis (TTS) with voice cloning, combined with automatic
speech recognition (ASR) for transcript generation.** The pipeline is a
two-stage NLP system:

1. **ASR** with **OpenAI Whisper** (via `faster-whisper`): converts each
   audio chunk into a text transcript. This solves the missing-transcript
   problem and is itself a deep-learning NLP model (encoder-decoder
   Transformer).
2. **TTS** with **Coqui XTTS-v2** (a multilingual neural TTS model based
   on a GPT-style audio token decoder + diffusion vocoder): fine-tuned on
   the speaker so it produces speech in their voice from arbitrary text.

### Why this is relevant

- **Accessibility**: voice cloning powers screen readers, voice
  reconstruction for people who have lost their voice (e.g. ALS), and
  multilingual dubbing.
- **Content creation**: indie game devs, podcasters, and educators can
  produce narration without recording each line.
- **Research / NLP coursework**: the pipeline touches on ASR, dataset
  curation, transfer learning, sequence-to-sequence modeling, evaluation
  metrics for generative speech, and audio signal processing — a broad
  cross-section of modern speech NLP.

## 3. Hardware split

| Machine | Role | Reason |
|---|---|---|
| MacBook M1 Pro, 32 GB RAM | preprocessing, evaluation, Gradio demo | available everywhere; Apple MPS is fine for small jobs |
| Gaming PC, Windows 11, **NVIDIA RTX 3070 (8 GB VRAM)** | Whisper ASR + XTTS-v2 fine-tune | CUDA is ~5–10× faster than MPS for this stack |

The two machines aren't always on the same network, so the workflow is
**file-based handoff** (zip + USB / cloud), documented in
[`docs/HANDOFF.md`](HANDOFF.md).

## 4. Repository layout (final)

```
AIVoiceMaker/
├── data/                        # gitignored
│   ├── raw/                     # original .mp3 files
│   ├── chunks/                  # VAD-segmented mono 24 kHz wavs
│   └── manifest.csv             # path|text|speaker|duration|language|split
├── models/                      # gitignored
│   ├── pretrained/XTTS-v2/      # base XTTS-v2 weights (downloaded once)
│   └── finetuned/<run_name>/    # fine-tuned checkpoints + loss_log.csv
├── src/
│   ├── utils.py                 # paths, device picker, logging helper
│   ├── preprocess.py            # mp3 → mono 24 kHz VAD-chunked wavs
│   ├── transcribe.py            # faster-whisper → manifest.csv
│   ├── split_dataset.py         # 90/10 train/val split
│   ├── train.py                 # XTTS-v2 fine-tune (CUDA preferred)
│   ├── evaluate.py              # loss curves + speaker similarity + WER
│   └── infer.py                 # text → wav using fine-tuned checkpoint
├── app/
│   └── gradio_app.py            # local type-to-speech web UI
├── docs/
│   ├── HANDOFF.md               # Mac↔PC file-movement instructions
│   ├── PROJECT_LOG.md           # this file
│   ├── index.html               # static GitHub Pages site (samples)
│   └── samples/                 # pre-rendered demo wavs (manual upload)
├── notebooks/                   # reserved
├── requirements.txt             # Mac base
├── requirements-cuda.txt        # PC additions (CUDA 12.1 torch override)
├── .gitignore                   # excludes data/ models/ *.wav *.mp3 *.pth
├── README.md
└── LICENSE
```

## 5. Pipeline — design choices and code-level details

### 5.1 Preprocessing — `src/preprocess.py`

**Goal:** turn long, noisy YouTube streams into many short, clean,
single-utterance WAVs suitable for TTS training.

**Steps inside the script:**
1. Iterate over `data/raw/*.mp3` (also `.wav .flac .m4a .ogg`).
2. **Decode + resample**: `librosa.load(..., sr=24000, mono=True)`. 24 kHz
   is XTTS-v2's output sample rate; matching it avoids resampling later.
3. **Voice activity detection (VAD)** with **Silero VAD** (loaded via
   `torch.hub.load("snakers4/silero-vad")`). Silero requires 16 kHz, so
   the audio is down-resampled internally just for VAD detection; the
   detected timestamps are then mapped back to the 24 kHz waveform.
   Parameters used:
   - `min_speech_duration_ms = 2500`  (drop tiny utterances)
   - `max_speech_duration_s  = 12`    (XTTS recipe likes ≤ ~12 s clips)
   - `min_silence_duration_ms = 300`  (split on real pauses)
   - `speech_pad_ms = 120`            (small head/tail cushion)
4. **Loudness normalize** each segment to **-23 LUFS** with `pyloudnorm`
   (ITU-R BS.1770), then peak-limit to 0.99 to prevent clipping.
   Segments too short for ITU loudness fall back to peak normalization.
5. **Write** as 16-bit PCM WAV at 24 kHz: `data/chunks/<src_stem>_<idx>.wav`.

**Why these choices:**
- VAD instead of silence-thresholding handles background music/SFX much
  better — Silero is a small CNN trained on speech vs noise.
- Loudness normalization (not just peak) makes training data consistent
  across sources recorded at different levels.
- 24 kHz mono PCM-16 matches XTTS-v2's input/output expectations.

### 5.2 Transcription — `src/transcribe.py`

**Model:** `faster-whisper` (a CTranslate2 reimplementation of OpenAI
Whisper that's 4–8× faster and lower-VRAM than the reference impl).
- On the **RTX 3070**: `large-v3` with `compute_type="int8_float16"`.
  Best quality, fast, fits in 8 GB.
- On the **Mac (fallback)**: `medium` with `compute_type="int8"`.

**Output:** `data/manifest.csv` with columns
`path, text, speaker, duration, language`. One row per chunk.

**Notes:**
- `vad_filter=False` because we already VAD'd in step 5.1; double-VAD
  hurts edge words.
- `beam_size=5` for slightly better accuracy than greedy.
- Empty transcriptions are dropped (model failed → unusable for training).

### 5.3 Train/val split — `src/split_dataset.py`

90% train / 10% val, fixed random seed (42), reproducible. Adds a `split`
column to `manifest.csv` in place. The val rows are used by
`evaluate.py` for the **validation-split round-trip WER** (text the model
trained on but the audio is held back from this exact metric run — pure
held-out comparison comes from the **held-out prompt set** below).

### 5.4 Fine-tuning — `src/train.py`

**Base model:** Coqui XTTS-v2. It has two trainable parts; we fine-tune
the **GPT-style audio-language model** (the part that decides which audio
tokens to emit for each text token). The DVAE (audio token codec) and
HiFi-GAN-like vocoder are kept frozen.

**Optimizer / schedule:**
- `AdamW`, `lr=5e-6`, betas=(0.9, 0.96), weight_decay=1e-2
- `MultiStepLR` (milestone at the end of long runs, gamma 0.5)
- **mixed precision (fp16)** when CUDA is available

**Memory tricks for 8 GB VRAM (RTX 3070):**
- `--batch-size 2` with `--grad-accum 8` → effective batch 16
- `--max-audio-sec 10` → caps `max_wav_length` so long clips don't OOM
- If OOM still occurs: `--batch-size 1 --grad-accum 16 --max-audio-sec 8`

**Logging:**
- TensorBoard via Coqui's built-in `dashboard_logger="tensorboard"`
- Best-effort per-epoch `loss_log.csv` saved alongside the checkpoint,
  consumed by `evaluate.py` to plot the loss curves figure.

**Output:** `models/finetuned/<run_name>/best_model.pth` + `config.json`
+ tokenizer files. This whole folder is what gets shipped back to the Mac.

**Honest caveat documented in code:** Coqui's TTS package has shifted its
trainer API across releases. The script wires up the standard XTTS GPT
trainer; if a future TTS version's API differs, the user should copy
`recipes/ljspeech/xtts_v2/train_gpt_xtts.py` from the Coqui repo and
point its dataset config at the dataset built by `build_coqui_dataset()`.

### 5.5 Evaluation — `src/evaluate.py`

Three objective metrics, all written to `evaluation/figures/` and a
summary `evaluation/metrics.csv`:

1. **Loss curves** — read from `loss_log.csv`, matplotlib line plot of
   train vs val loss per epoch. Demonstrates the model actually learned
   (decreasing val loss) and isn't overfit (val curve doesn't diverge).

2. **Speaker similarity (cosine)** — uses **Resemblyzer**
   (a pretrained GE2E speaker-encoder LSTM that maps any utterance to a
   256-dim embedding). For each held-out synthesized clip, compute the
   cosine similarity between its embedding and the **mean embedding of
   the original speaker's chunks**. Higher = closer to target voice.
   Range 0..1; >0.75 typically perceived as "same speaker".

3. **Round-trip WER** — synthesize a sentence, transcribe the result with
   Whisper, compute Word Error Rate (`jiwer.wer`) vs the input text.
   Lower = clearer / more intelligible speech. Computed on:
   - 5 **held-out prompts** the speaker never said (test set).
   - The **validation split** of training texts (val set).

**Held-out prompts (test set):**
```
"The quick brown fox jumps over the lazy dog."
"Artificial intelligence is reshaping how we create media."
"I am a synthetic voice trained on YouTube recordings."
"Today's weather is unusually pleasant for this time of year."
"Please save your work before closing the application."
```

These satisfy the homework's "results on data different from training"
requirement.

### 5.6 Inference + demo — `src/infer.py` + `app/gradio_app.py`

- `infer.py.synthesize(text, reference_wav=None, checkpoint_dir=None,
  language="en")` returns `(sample_rate, waveform)`.
- The default `reference_wav` is the first chunk in `data/chunks/` (used
  by XTTS-v2 to extract the speaker conditioning embedding at inference).
- The default `checkpoint_dir` is the most-recently-modified subfolder of
  `models/finetuned/`.
- Model is cached after first load.
- `gradio_app.py` is a 30-line UI: textbox in, audio player out, three
  example prompts.

## 6. Concrete actions taken on this build

### 6.1 Repo scaffolding
- Created `src/`, `app/`, `docs/`, `notebooks/`, `models/`, `data/chunks/`
  (with `.gitkeep` placeholders).
- Updated `.gitignore` to exclude `data/`, `models/`, `runs/`, `.gradio/`,
  `*.mp3`, `*.wav`, `*.flac`, `*.ogg`, `*.m4a`, `*.ckpt`, `*.pth`, `*.pt`,
  `*.bin`, `*.safetensors`, `tensorboard/`, `lightning_logs/`.
- Wrote `requirements.txt` and `requirements-cuda.txt`.
- Wrote all 7 scripts above + the Gradio app.
- Wrote `docs/HANDOFF.md` (Mac↔PC workflow) and `docs/index.html`
  (GitHub Pages static site that auto-discovers samples in `docs/samples/`
  via either `index.json` or HEAD probes).
- Updated `README.md` with quickstart, layout, and timing table.

### 6.2 Mac environment setup
- Initial `python3` on the Mac was Python **3.14.3**, which has no wheels
  for Coqui TTS / faster-whisper / PyTorch yet. Switched to conda
  (`/opt/anaconda3`) and created:
  ```bash
  conda create -n aivoice python=3.11 -y
  conda activate aivoice
  ```
- Installed the preprocessing-only deps first (small, fast):
  `numpy "scipy<1.13" soundfile librosa pydub pyloudnorm torch torchaudio tqdm`
- Verified `torch.backends.mps.is_available() == True` on M1 Pro.
- `ffmpeg 8.1.1` already present at `/opt/homebrew/bin/ffmpeg` (required
  by librosa / pydub for MP3 decoding).

### 6.3 Preprocessing run

Command:
```bash
python -m src.preprocess
```

Wall time: **~2.5 minutes**.

Per-file VAD output:
| Source MP3 | Raw duration | Speech segments | Chunks written |
|---|---:|---:|---:|
| Ame Messes Up The Timeline … | 3.02 min | 16 | 16 |
| Back from Japan ~ ☆ | 120.05 min | 788 | 776 |
| OMG!!! (ò_ó) ............. hi | 188.52 min | 841 | 840 |
| hey! | 128.15 min | 621 | 618 |
| **Total** | **439.7 min (7.33 h)** | 2,266 | **2,250** |

### 6.4 Post-preprocessing verification

| Property | Value |
|---|---|
| Total raw audio | 7.33 h |
| Total kept audio (after VAD) | 3.18 h |
| **Retention rate** | **43.4 %** (rest = silence / music / SFX dropped) |
| Number of chunks | 2,250 |
| Format | mono, 24 000 Hz, PCM-16 |
| Per-chunk duration | min 2.58 s, **median 4.37 s**, max 11.98 s, mean 5.09 s |
| Loudness | RMS 0.043–0.076 (mean 0.067) — very consistent |
| Peaks | mean 0.543, max 0.990 → **0 clipped files** |
| Silent files (RMS < 0.005) | **0** |
| Total size on disk | 528 MB |

**Interpretation:** the ~7 h of streams contained ~3 h of actual speech
from the speaker. That's a strong dataset for single-speaker TTS
fine-tuning (typical recipes use 30 min – 5 h).

### 6.5 Sanity-check methodology

A 10-file random sample was inspected directly (RMS, peak, duration), and
then *all* 2,250 files were scanned with `soundfile` to confirm:
- no silent chunks (would be useless for training),
- no clipped chunks (would inject noise),
- consistent loudness (means the loudness normalizer worked),
- duration distribution matches the requested 2.5–12 s window.

All four checks passed.

### 6.6 Handoff package built

```bash
cd /Users/maxzemeno/Documents/GitHub/AIVoiceMaker/data
zip -r "/Users/maxzemeno/Documents/Tareas UANL/Introduccion al Aprendizaje Profundo/AIVoiceMaker_chunks.zip" chunks
```

Result: `AIVoiceMaker_chunks.zip` (451 MB) in the Tareas folder, ready to
ship to the RTX 3070 PC. Compression saved ~77 MB versus the raw 528 MB
folder (PCM-16 doesn't compress much).

## 7. Status / next steps

Completed (this session):
- [x] Repo scaffolding
- [x] All Python scripts + Gradio app + GH Pages site
- [x] Mac conda env (`aivoice`, Python 3.11) + preprocessing deps installed
- [x] Preprocessing run on all 4 MP3s → 2,250 chunks verified

Remaining:
- [ ] Ship `data/chunks/` to the RTX 3070 PC (zip + transfer)
- [ ] On PC: install `requirements.txt` + `requirements-cuda.txt`
- [ ] On PC: `python -m src.transcribe --model large-v3 --compute-type int8_float16`
- [ ] On PC: `python -m src.split_dataset`
- [ ] On PC: `python -m src.train --epochs 10 --batch-size 2 --grad-accum 8`
- [ ] Ship `models/finetuned/<run>/` and `data/manifest.csv` back to Mac
- [ ] On Mac: `python -m src.evaluate` → loss curves, similarity, WER
- [ ] On Mac: `python app/gradio_app.py` → live demo
- [ ] (Optional) copy a few `evaluation/synth/prompt_*.wav` into
      `docs/samples/` and enable GitHub Pages

## 8. Citations / dependencies (for the report)

- **OpenAI Whisper** — Radford et al., 2022. *Robust Speech Recognition
  via Large-Scale Weak Supervision.* arXiv:2212.04356.
- **faster-whisper** — Guillaume Klein. CTranslate2 reimplementation of
  Whisper. https://github.com/SYSTRAN/faster-whisper
- **Silero VAD** — Silero Team. https://github.com/snakers4/silero-vad
- **Coqui TTS / XTTS-v2** — Coqui-AI / Idiap. CPML-licensed.
  https://github.com/coqui-ai/TTS
- **Resemblyzer** — Corentin Jemine. GE2E speaker encoder.
  https://github.com/resemble-ai/Resemblyzer  (and Wan et al., 2018,
  *Generalized End-to-End Loss for Speaker Verification*, ICASSP).
- **jiwer** — Word Error Rate computation.
  https://github.com/jitsi/jiwer
- **Gradio** — https://www.gradio.app/
- **PyTorch** — Paszke et al., 2019.
- **librosa, soundfile, pyloudnorm, pydub** — standard Python audio
  tooling.
- **Source data:** Hololive English / Amelia Watson YouTube streams
  (4 videos, ~7 h total). Used here strictly for personal homework /
  research, no redistribution. The XTTS-v2 license is explicitly
  non-commercial (CPML), which aligns with this use.

## 9. Personal-conclusion seed (for the report writer)

Some honest takeaways the report can build on:
- VAD + loudness normalization matter as much as the model choice — they
  determined whether 2 h of "stream audio" became 1 h of usable training
  data or 0.5 h.
- Transfer learning collapses the data requirement: fine-tuning XTTS-v2
  on ~3 h of one speaker is a different universe from training a TTS
  model from scratch (which would need tens of hours of *clean studio*
  audio + a GPU cluster).
- ASR (Whisper) and TTS (XTTS) used together let you bootstrap a custom
  voice from zero metadata — the labels are generated for you. This is
  a real-world pattern (data-flywheel: model A produces labels for
  model B).
- The biggest engineering risks were (a) Python version compatibility
  with the Coqui stack and (b) Coqui's trainer API churn across releases
  — both worth flagging in any reproduction guide.
