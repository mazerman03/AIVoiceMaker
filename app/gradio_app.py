"""
Local Gradio demo: type text, hear it spoken in the cloned voice.

Usage:
    python app/gradio_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `from src...` when launched as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from src.infer import synthesize  # noqa: E402
from src.utils import setup_logging  # noqa: E402

EXAMPLES = [
    "Hello! This is my AI-cloned voice talking to you.",
    "The model was fine-tuned on YouTube recordings.",
    "Type any sentence above and click Generate.",
]


def tts(text: str):
    text = (text or "").strip()
    if not text:
        return None
    sr, wav = synthesize(text)
    return (sr, wav)


def main() -> None:
    setup_logging()
    with gr.Blocks(title="AIVoiceMaker") as demo:
        gr.Markdown("# AIVoiceMaker\nType text → hear the cloned voice.")
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", lines=4, placeholder="Type something...")
                btn = gr.Button("Generate", variant="primary")
                gr.Examples(EXAMPLES, inputs=text)
            with gr.Column():
                audio = gr.Audio(label="Generated speech", type="numpy")
        btn.click(tts, inputs=text, outputs=audio)
    demo.launch()


if __name__ == "__main__":
    main()
