"""Gradio entrypoint: upload, Raw/Furnished toggle, Generate button,
before/after image comparison, on-demand read-aloud, English/Chinese
language switch. Orchestrator instantiated once."""
from __future__ import annotations

import tempfile

import gradio as gr
import numpy as np
from PIL import Image

from api_clients.audio_client import VOICE_BY_LANGUAGE, speak_text, transcribe
from orchestrator import Orchestrator
from translations import UI

orchestrator = Orchestrator()


def _save_temp_image(image_rgb: np.ndarray) -> str:
    path = tempfile.mktemp(suffix=".jpg")
    Image.fromarray(image_rgb).save(path)
    return path


def _extract_purpose(mm_value: dict | None) -> str:
    """MultimodalTextbox value is {"text": str, "files": [path, ...]}. If the mic
    was used, transcribe the recording; otherwise use whatever was typed."""
    if not mm_value:
        return ""
    files = mm_value.get("files") or []
    if files:
        return transcribe(files[0]) or mm_value.get("text", "")
    return mm_value.get("text", "")


def generate(image: np.ndarray, mode: str, purpose_input: dict, language: str):
    t = UI.get(language, UI["en"])
    if image is None:
        return t["upload_prompt"], None

    room_purpose = _extract_purpose(purpose_input)

    try:
        if mode == "raw":
            result = orchestrator.run_scenario_a(image, room_purpose or t["default_purpose"], language)
            text = result["text"]
        else:
            image_path = _save_temp_image(image)
            result = orchestrator.run_scenario_b(image, image_path, language)
            text = result["text"] + "\n\n" + result.get("checklist", "")

        return text, result["image"]
    except Exception as exc:
        return f"{t['error_prefix']}{exc}", None


def read_aloud(advice_text: str, language: str) -> str | None:
    if not advice_text:
        return None
    voice = VOICE_BY_LANGUAGE.get(language, VOICE_BY_LANGUAGE["en"])
    return speak_text(advice_text, voice=voice)


def switch_language(language: str):
    t = UI.get(language, UI["en"])
    return (
        gr.update(value=t["title"]),
        gr.update(label=t["mode_label"], choices=t["mode_choices"]),
        gr.update(label=t["purpose_label"], placeholder=t["purpose_placeholder"]),
        gr.update(value=t["generate_btn"]),
        gr.update(value=t["before"]),
        gr.update(value=t["after"]),
        gr.update(value=t["read_aloud_btn"]),
    )


with gr.Blocks(title="AI Interior Feng Shui Consultant / AI 室内风水顾问") as demo:
    with gr.Row():
        with gr.Column(scale=5):
            title_md = gr.Markdown(UI["en"]["title"])
        with gr.Column(scale=1, min_width=140):
            language = gr.Radio(
                [("English", "en"), ("中文", "zh")], value="en", label="Language / 语言", container=False,
            )

    with gr.Row():
        mode = gr.Radio(
            UI["en"]["mode_choices"], value="raw", label=UI["en"]["mode_label"], scale=1,
        )
        # A single chat-style input box: type the room's intended use, or click
        # the mic icon inside the box to speak it instead.
        room_purpose = gr.MultimodalTextbox(
            label=UI["en"]["purpose_label"],
            placeholder=UI["en"]["purpose_placeholder"],
            sources=["microphone"],
            scale=3,
        )

    generate_btn = gr.Button(UI["en"]["generate_btn"], variant="primary")

    # Before/after side by side (horizontal) so the two photos are directly comparable,
    # rather than stacked vertically and requiring scrolling back and forth.
    with gr.Row():
        with gr.Column():
            before_md = gr.Markdown(UI["en"]["before"])
            image_in = gr.Image(label="Room Photo", type="numpy")
        with gr.Column():
            after_md = gr.Markdown(UI["en"]["after"])
            output_image = gr.Image(label="Result")

    with gr.Row():
        with gr.Column(scale=4):
            output_text = gr.Markdown(label="Advice")
        with gr.Column(scale=1, min_width=120):
            read_aloud_btn = gr.Button(UI["en"]["read_aloud_btn"])
            output_audio = gr.Audio(label=None, autoplay=True, show_label=False)

    def _toggle_purpose_input(m: str):
        return gr.update(visible=(m == "raw"))

    mode.change(_toggle_purpose_input, inputs=mode, outputs=room_purpose)
    language.change(
        switch_language,
        inputs=language,
        outputs=[title_md, mode, room_purpose, generate_btn, before_md, after_md, read_aloud_btn],
    )

    generate_btn.click(generate, inputs=[image_in, mode, room_purpose, language], outputs=[output_text, output_image])
    read_aloud_btn.click(read_aloud, inputs=[output_text, language], outputs=output_audio)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
