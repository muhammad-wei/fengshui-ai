"""Gradio entrypoint: upload, Raw/Furnished toggle, Generate button,
text+image+audio output, mic input. Orchestrator instantiated once."""
from __future__ import annotations

import tempfile

import gradio as gr
import numpy as np
from PIL import Image

from api_clients.audio_client import transcribe
from orchestrator import Orchestrator

orchestrator = Orchestrator()


def _save_temp_image(image_rgb: np.ndarray) -> str:
    path = tempfile.mktemp(suffix=".jpg")
    Image.fromarray(image_rgb).save(path)
    return path


def generate(image: np.ndarray, mode: str, room_purpose: str):
    if image is None:
        return "Please upload a photo first.", None, None

    try:
        if mode == "Raw Space":
            result = orchestrator.run_scenario_a(image, room_purpose or "bedroom")
            text = result["text"]
        else:
            image_path = _save_temp_image(image)
            result = orchestrator.run_scenario_b(image, image_path)
            text = result["text"] + "\n\n" + result.get("checklist", "")

        return text, result["image"], result["audio"]
    except Exception as exc:
        return f"Something went wrong generating your report: {exc}", None, None


def on_mic_input(audio_path: str, current_purpose: str) -> str:
    if not audio_path:
        return current_purpose
    text = transcribe(audio_path)
    return text or current_purpose


with gr.Blocks(title="AI Interior Feng Shui Consultant") as demo:
    gr.Markdown("# AI Interior Feng Shui Consultant")

    mode = gr.Radio(["Raw Space", "Furnished Space"], value="Raw Space", label="Scenario")
    image_in = gr.Image(label="Room Photo", type="numpy")
    room_purpose = gr.Textbox(label="Intended use (Raw Space only)", placeholder="e.g. master bedroom", visible=True)
    mic_in = gr.Audio(label="Or speak the intended use", type="filepath")

    def _toggle_purpose_inputs(m: str):
        visible = m == "Raw Space"
        return gr.update(visible=visible), gr.update(visible=visible)

    mode.change(_toggle_purpose_inputs, inputs=mode, outputs=[room_purpose, mic_in])
    mic_in.change(on_mic_input, inputs=[mic_in, room_purpose], outputs=room_purpose)

    generate_btn = gr.Button("Generate", variant="primary")

    output_text = gr.Markdown(label="Advice")
    output_image = gr.Image(label="Result")
    output_audio = gr.Audio(label="Narrated Advice", autoplay=False)

    generate_btn.click(generate, inputs=[image_in, mode, room_purpose], outputs=[output_text, output_image, output_audio])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
