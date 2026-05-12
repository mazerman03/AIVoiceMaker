# Theoretical Basis of AIVoiceMaker

A plain-English walkthrough of everything this project relies on, going
from "what is AI" down to "what exactly is happening when you press
*train*". Read this once and you'll be able to explain the project to
anyone, including yourself in three months.

---

## 1. The big picture: what kind of AI is this?

This project is a **Text-to-Speech (TTS)** system. You type a sentence,
and the computer speaks it out loud — but it speaks in a *specific
person's voice* that you taught it from audio recordings.

That's two problems glued together:

1. **Speech synthesis**: turn text into audio that sounds like a human
   talking. (A computer making any human-sounding voice.)
2. **Voice cloning**: make that audio sound like *one specific person*,
   not a generic robot voice.

Modern AI does both at once with the same model.

---

## 2. What is a neural network, really?

A neural network is just a giant mathematical function:

```
output = f(input, parameters)
```

- `input` is whatever you feed it (text, audio, an image, …).
- `parameters` (also called *weights*) are millions or billions of
  numbers stored in the model.
- `f` is a fixed recipe of additions, multiplications, and one
  non-linear "squish" step (called an *activation function*), repeated
  in many *layers*.

When you **train** a neural network, you don't write the recipe — you
write the *initial* parameters as random numbers, then iteratively
nudge them so that for every example in your dataset, the output gets
closer to the correct answer.

The "nudging" is done by:

1. Computing how wrong the output is — a number called the **loss**.
2. Asking calculus (specifically, the *chain rule* — implemented as
   **backpropagation**) which direction to push each parameter to make
   the loss smaller.
3. Pushing each parameter a tiny step in that direction. That tiny step
   size is called the **learning rate**.

Repeat this billions of times across billions of examples and the
parameters end up encoding something useful — like "how English speech
sounds".

---

## 3. Two key tricks this project uses

### 3.1 Transfer learning (a.k.a. fine-tuning)

Training a TTS model **from scratch** would need:

- Tens of hours of studio-quality recordings of one person,
- Professional text transcripts,
- A small GPU cluster running for weeks.

We don't have any of that. So we don't start from scratch — we start
from a model someone else **already trained** on thousands of hours of
audio from many speakers, and we just *adjust* it to sound like our
target speaker.

This is called **transfer learning**. The pretrained model already knows
how human speech works (phonemes, rhythm, prosody). We're just teaching
it the *flavor* of one new voice. It's the difference between teaching
someone Spanish (months) and teaching a fluent speaker the Argentinian
accent (weeks).

The pretrained model we're starting from is called **XTTS-v2**, released
by Coqui AI. It was trained on thousands of hours of multilingual
speech, supports 17 languages, and can already imitate a speaker after
hearing only a 6-second sample of their voice. We're going further than
that 6-second imitation — we're actually rewriting some of its
parameters with full fine-tuning so it gets really good at our voice
specifically.

### 3.2 Self-supervised data labeling

The second trick solves a *data* problem. The Hololive Amelia Watson
recordings I started with are just MP3 streams from YouTube. There are
no text transcripts paired with each second of audio.

TTS training needs `(text, audio)` pairs — the text says *what* to say
and the audio says *how it should sound*. Without transcripts we'd be
stuck.

The solution: use a *different* AI model — **Whisper** by OpenAI — to
**listen to the audio and write the transcripts for us automatically**.
Whisper is a state-of-the-art Automatic Speech Recognition (ASR) model.
It hears audio and writes down the words. Its transcripts are not
perfect, but on clean-ish English speech it makes only a handful of
mistakes per hundred words, which is good enough for our purposes.

This pattern — using one model to label data for another model — is
called **pseudo-labeling** or a **data flywheel**. It is one of the
most important practical techniques in modern deep learning.

---

## 4. The full pipeline, end-to-end

Here is the entire chain of things that happens, in order. Every step
exists for a real reason — none of this is busywork.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Raw input:  4 long MP3 files (~7.3 hours of YouTube streams)        │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 1 — PREPROCESSING   (Mac, src/preprocess.py)                   │
│                                                                      │
│  • Convert MP3 → mono 24 kHz WAV (the format XTTS expects).          │
│  • Run a Voice Activity Detector (Silero VAD) to find segments       │
│    where someone is actually speaking. Cuts away laughter, music,    │
│    background, silence between sentences.                            │
│  • Slice the long files into 2.5–12 s chunks, each containing one    │
│    coherent utterance.                                               │
│  • Loudness-normalize each chunk to a standard volume (-23 LUFS),    │
│    so the model isn't confused by loud parts vs quiet parts.         │
│                                                                      │
│  Result: 2,250 small WAV files, ~3.18 h of clean speech.             │
│  (We threw away 56% of the raw audio — most of it was silence,       │
│   music, or noise. This is normal.)                                  │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 2 — TRANSCRIPTION   (PC w/ RTX 3070, src/transcribe.py)        │
│                                                                      │
│  • For each of the 2,250 WAVs, run Whisper (large-v3 model).         │
│  • Whisper "listens" and writes down the English transcript.         │
│  • Output: one CSV file, `data/manifest.csv`, with columns           │
│    (wav_path, text). One row per chunk.                              │
│                                                                      │
│  Took ~40 minutes on the GPU.  This is the (audio, text) dataset.    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 3 — SPLIT   (src/split_dataset.py)                             │
│                                                                      │
│  • Randomly split the 2,250 chunks into ~95% training, ~5% held-out  │
│    validation.  The validation chunks are NEVER shown to the model   │
│    during training — they are the "exam" used to check that the     │
│    model is learning generalizable patterns, not just memorizing.    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 4 — FINE-TUNING   (PC w/ RTX 3070, src/train.py)               │
│                                                                      │
│  • Load the pretrained XTTS-v2 weights (~2 GB).                      │
│  • For each (audio, text) pair in the training set:                  │
│       a) Feed the text + a short snippet of *our* voice into the     │
│          model.  Ask it to predict the rest of the audio.            │
│       b) Compare its prediction to the real audio → loss.            │
│       c) Backpropagate and nudge the parameters.                     │
│  • Do this for 10 epochs (each epoch = one full pass through         │
│    the 2,250 chunks).                                                │
│  • Every so often, check loss on the validation set to make sure     │
│    the model is actually improving, not overfitting.                 │
│                                                                      │
│  At the end: a new "AIVoiceMaker" checkpoint that sounds like our    │
│  target speaker.                                                     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 5 — EVALUATION   (Mac, src/evaluate.py)                        │
│                                                                      │
│  Three objective metrics:                                            │
│                                                                      │
│  • Loss curves — should go down over epochs (visual sanity check).   │
│  • Speaker similarity — synthesize a sentence with the fine-tuned    │
│    model, then use a *third* model (Resemblyzer, a speaker-          │
│    verification network) to compute the cosine similarity between    │
│    the synthesized voice and the real voice. Higher = better.        │
│  • Round-trip WER — feed the synthesized audio back into Whisper.    │
│    Whisper transcribes it. Compare its transcript to the text we    │
│    *asked* the model to say. Word Error Rate near 0% = the model    │
│    produces intelligible speech.                                     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Step 6 — INFERENCE / DEMO   (Mac, app/gradio_app.py)                │
│                                                                      │
│  A simple web UI (Gradio) where you type text → it plays audio.      │
│  This is the final user-facing product.                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Zooming in on the model: how does XTTS-v2 actually work?

XTTS-v2 is the model doing the heavy lifting. Internally it has three
neural sub-networks that work like an assembly line:

1. **Tokenizer / text encoder.** Splits the input text into small units
   (~characters / sub-words) and turns each one into a vector of
   numbers. So `"Hello world"` becomes a list of vectors.

2. **GPT-style autoregressive decoder.**  This is *the* big component —
   essentially the same architecture as ChatGPT, but it predicts
   **discrete audio tokens** instead of text tokens.  It reads the text
   vectors *and* a 6-second snippet of the target speaker (used as
   "conditioning" — this is how the speaker identity gets injected),
   then generates a sequence of audio tokens one at a time. Each token
   represents about 25 milliseconds of speech.

   This is the part we are fine-tuning. Of the ~500 million parameters
   in the model, the GPT decoder is most of them.

3. **HiFi-GAN vocoder.** Takes the sequence of audio tokens from the
   decoder and turns them back into an actual waveform you can play
   through speakers. We don't fine-tune this — it's already very good
   at the "tokens → waveform" job for any voice.

So one inference (one "say this sentence") looks like:

```
text → tokenizer → text vectors ───┐
speaker WAV (6s reference) ────────┤
                                   ▼
                              GPT decoder
                                   │
                                   ▼
                          discrete audio tokens
                                   │
                                   ▼
                            HiFi-GAN vocoder
                                   │
                                   ▼
                              24 kHz waveform ─→ speaker
```

During training, the loss measures how close the GPT's predicted audio
tokens are to the real audio tokens of the chunk we're training on.
Two losses are combined:

- `loss_mel_ce` — does the predicted audio match the real mel
  spectrogram of the chunk? (Most of the signal.)
- `loss_text_ce` — does the model attend to the right pieces of the
  text? (Keeps it from drifting.)

---

## 6. Why split the work across Mac and PC?

Two physical machines, two very different jobs:

| Machine | What it's good at | What it does here |
|---------|-------------------|-------------------|
| MacBook Pro (M1 Pro) | Solid general-purpose CPU, partial GPU acceleration via Apple's MPS backend, decent audio tooling | Preprocessing (it's CPU-bound), evaluation, and the final Gradio demo |
| Windows desktop + NVIDIA RTX 3070 (8 GB VRAM) | Has CUDA-compatible NVIDIA GPU. The PyTorch + Coqui-TTS stack is best supported on CUDA. | Whisper transcription + XTTS-v2 fine-tuning |

GPU memory is the limit. With 8 GB of VRAM we can only fit batches of
~2 audio chunks at a time during training, which is why we use a trick
called **gradient accumulation**: process 2 chunks, *remember* the
gradients but don't update the model yet, process the next 2, add their
gradients to the accumulator, …, and after 8 rounds finally update.
This simulates a batch of size 16 with the memory of size 2 — slower
but mathematically equivalent.

---

## 7. Glossary of the jargon used in this repo

- **Epoch**: one complete pass over the entire training set.
- **Step**: one parameter update (one batch processed and weights
  nudged).
- **Loss**: a single number representing how wrong the model currently
  is on this batch. Lower is better.
- **Learning rate**: how big a step we take when nudging parameters.
  Too big → overshoots; too small → trains forever. We use 5e-6.
- **Batch size**: how many examples are processed together before one
  gradient update.
- **Gradient accumulation**: simulating a larger batch on a small GPU
  by adding gradients from several mini-batches before updating.
- **Mixed precision (fp16)**: doing math in 16-bit floats instead of
  32-bit when safe. Roughly 2× faster on modern GPUs and uses half the
  memory.
- **Checkpoint**: a snapshot of the model's parameters saved to disk
  during training. The "model" we ship is just the latest checkpoint.
- **Inference**: using the model after training (i.e. generating
  speech, as opposed to training).
- **Validation set**: examples held out from training, used only to
  measure progress on data the model hasn't seen.
- **Overfitting**: when the model memorizes the training set instead
  of learning general patterns. Signal: training loss keeps going down
  but validation loss starts going up.
- **VAD (Voice Activity Detection)**: deciding which slices of audio
  contain speech vs silence/music.
- **Mel spectrogram**: a 2D image-like representation of audio (time
  on one axis, frequency on the other) that's more useful for ML than
  the raw waveform. Most TTS models internally work in this space.
- **ASR (Automatic Speech Recognition)**: audio → text. Whisper.
- **TTS (Text-to-Speech)**: text → audio. XTTS-v2.
- **WER (Word Error Rate)**: % of words wrong when comparing a
  transcript to a reference. Lower = better.
- **Cosine similarity**: a number between –1 and 1 saying how similar
  two vectors are (1 = identical direction). We use it on
  speaker-embedding vectors to ask "how alike do these two voices
  sound?".

---

## 8. One-paragraph summary you can recite

> *AIVoiceMaker is a voice-cloning text-to-speech system. We start
> with a few hours of YouTube audio of one speaker, automatically
> transcribe it with OpenAI's Whisper, clean and slice it into
> several thousand short utterances, and then fine-tune Coqui's
> pretrained XTTS-v2 model on those `(text, audio)` pairs. The result
> is a model that, given any English sentence, generates speech in
> that specific person's voice. We evaluate it with three objective
> metrics — training/validation loss curves, speaker-similarity
> cosine, and round-trip word error rate — and expose it through a
> small Gradio web app for live use.*

That paragraph alone is your homework defense. The rest of this doc
exists so you can answer follow-up questions about *why* every step is
there.
