# Mac ↔ PC Handoff

Two machines share this project, but **only the code** is in git. The data
and model checkpoints move manually (USB / cloud / scp). This doc lists
exactly what to copy in each direction.

| Machine | Role |
|---|---|
| **MacBook M1 Pro (32 GB)** | Preprocessing, evaluation, Gradio demo, presentation |
| **Gaming PC (Win 11, RTX 3070, 8 GB VRAM)** | Whisper transcription + XTTS-v2 fine-tune |

---

## One-time setup on each machine

### Mac

```bash
git clone <repo> AIVoiceMaker
cd AIVoiceMaker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Windows PC (PowerShell)

```powershell
git clone <repo> AIVoiceMaker
cd AIVoiceMaker
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-cuda.txt
```

Sanity-check CUDA on the PC:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True NVIDIA GeForce RTX 3070
```

---

## Workflow

### Step 1 — On the Mac: preprocess

Drop the YouTube MP3s into `data/raw/`, then:

```bash
python -m src.preprocess
```

Produces `data/chunks/*.wav` (mono, 24 kHz, VAD-segmented, loudness-normalized).
Expected size for ~7 h of audio: roughly **1–2 GB**.

### Step 2 — Mac → PC: ship the chunks

Zip `data/chunks/` and move it to the PC by whatever's convenient:
USB stick, Google Drive, Dropbox, or scp. On the PC, place it back at
`AIVoiceMaker/data/chunks/`.

```bash
# Mac
cd AIVoiceMaker/data
zip -r chunks.zip chunks
# ... transfer chunks.zip to the PC ...
```

```powershell
# PC
cd AIVoiceMaker\data
Expand-Archive chunks.zip -DestinationPath .
```

### Step 3 — On the PC: transcribe → split → fine-tune

```powershell
python -m src.transcribe --model large-v3 --compute-type int8_float16
python -m src.split_dataset
python -m src.train --epochs 10 --batch-size 2 --grad-accum 8
```

Outputs:
- `data/manifest.csv`   (~ tens of KB)
- `models/finetuned/xtts_v2_finetune/`  (checkpoints, ~1.5–2.5 GB total;
  the `best_model.pth` and `config.json` are what you need on the Mac)

If you hit a CUDA OOM:

```powershell
python -m src.train --epochs 10 --batch-size 1 --grad-accum 16 --max-audio-sec 8
```

### Step 4 — PC → Mac: ship the checkpoint

Send back the **whole `models/finetuned/xtts_v2_finetune/` folder** (or just
`best_model.pth` + `config.json` + any tokenizer files in the run directory).

```powershell
# PC
cd AIVoiceMaker\models\finetuned
Compress-Archive xtts_v2_finetune xtts_v2_finetune.zip
# ... transfer to Mac ...
```

```bash
# Mac
cd AIVoiceMaker/models/finetuned
unzip xtts_v2_finetune.zip
```

Also copy back **`data/manifest.csv`** so the Mac can compute the
validation-split WER.

### Step 5 — On the Mac: evaluate + demo

```bash
python -m src.evaluate
python app/gradio_app.py
```

`evaluate.py` writes `evaluation/figures/loss_curves.png` and
`evaluation/metrics.csv`. The Gradio app opens at <http://127.0.0.1:7860>.

---

## What is and isn't in git

In git: code (`src/`, `app/`, `docs/`), `requirements*.txt`, `README.md`,
`.gitignore`, `LICENSE`, empty `.gitkeep` placeholders.

**Never** in git (gitignored): `data/raw/`, `data/chunks/`,
`data/manifest.csv`, `models/`, `runs/`, `.gradio/`, any `*.wav` / `*.mp3` /
`*.pth` / `*.ckpt` files anywhere in the tree.

If you ever stage one accidentally, `git restore --staged <path>` and check
`.gitignore` covers it.
