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

from src.infer import (  # noqa: E402
    DEFAULT_GEN_PARAMS,
    list_reference_chunks,
    synthesize,
)
from src.utils import setup_logging  # noqa: E402

EXAMPLES = [
    "Hello! This is my AI-cloned voice talking to you.",
    "The model was fine-tuned on YouTube recordings using XTTS version 2.",
    "I can read full sentences with natural rhythm and intonation.",
    "Type any sentence above and click Generate to hear it.",
    "Did you know that Hololive is a Japanese virtual YouTuber agency?",
    "Today's weather is unusually pleasant for this time of year.",
    "Artificial intelligence is reshaping how we create digital media.",
    "I am a synthetic voice trained on roughly three hours of clean audio.",
    "Please save your work before closing the application.",
    "Once upon a time, in a land far, far away, a small fox went on an adventure.",
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore on sunny summer afternoons.",
]


def _ref_choices() -> list[str]:
    refs = list_reference_chunks(min_seconds=6.0, limit=20)
    return [str(p) for p in refs]


def _initial_refs() -> list[str]:
    # Pick a few different reference clips by default — XTTS-v2 averages
    # speaker embeddings across them, which sounds noticeably less robotic
    # than a single short reference.
    return _ref_choices()[:3]


def tts(
    text: str,
    refs: list[str],
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    speed: float,
):
    text = (text or "").strip()
    if not text:
        return None
    chosen = refs or _initial_refs()
    sr, wav = synthesize(
        text,
        reference_wav=chosen,
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k),
        repetition_penalty=repetition_penalty,
        speed=speed,
    )
    return (sr, wav)


def main() -> None:
    setup_logging()
    ref_options = _ref_choices()
    initial = _initial_refs()
    with gr.Blocks(title="AIVoiceMaker") as demo:
        gr.Markdown(
            "# AIVoiceMaker\n"
            "Type text → hear it spoken in the fine-tuned voice. "
            "Pick one or more reference clips (longer + multiple = sounds less synthetic). "
            "Tweak the generation parameters for more or less expressive delivery."
        )
        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(
                    label="Text",
                    lines=4,
                    placeholder="Type something...",
                    value=EXAMPLES[0],
                )
                btn = gr.Button("Generate", variant="primary", size="lg")
                gr.Examples(EXAMPLES, inputs=text, label="Example sentences (click to load)")
            with gr.Column(scale=1):
                audio = gr.Audio(label="Generated speech", type="numpy", autoplay=True)
                with gr.Accordion("Voice references (used to clone the voice)", open=False):
                    refs = gr.Dropdown(
                        label="Reference clips",
                        choices=ref_options,
                        value=initial,
                        multiselect=True,
                        info=(
                            "XTTS-v2 averages the speaker embedding across these. "
                            "More + longer clips → richer voice."
                        ),
                    )
                with gr.Accordion("Generation parameters", open=False):
                    temperature = gr.Slider(
                        0.1, 1.5, value=DEFAULT_GEN_PARAMS["temperature"],
                        step=0.05, label="Temperature",
                        info="Higher = more variation/expression, lower = more monotone/safe.",
                    )
                    top_p = gr.Slider(
                        0.1, 1.0, value=DEFAULT_GEN_PARAMS["top_p"],
                        step=0.05, label="top_p",
                        info="Nucleus sampling cutoff.",
                    )
                    top_k = gr.Slider(
                        0, 200, value=DEFAULT_GEN_PARAMS["top_k"],
                        step=1, label="top_k",
                        info="Limits the vocabulary at each step (0 = off).",
                    )
                    repetition_penalty = gr.Slider(
                        1.0, 10.0, value=DEFAULT_GEN_PARAMS["repetition_penalty"],
                        step=0.5, label="Repetition penalty",
                        info="Higher reduces stutters/loops.",
                    )
                    speed = gr.Slider(
                        0.5, 2.0, value=DEFAULT_GEN_PARAMS["speed"],
                        step=0.05, label="Speed",
                        info="Speech rate (1.0 = natural).",
                    )

        btn.click(
            tts,
            inputs=[text, refs, temperature, top_p, top_k, repetition_penalty, speed],
            outputs=audio,
        )
    demo.launch(theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
