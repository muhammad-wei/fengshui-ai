# AI Interior Feng Shui Consultant

[中文版](README.zh-CN.md)

A Gradio app that takes a room photo and, depending on scenario, outputs a Feng-Shui-guided
layout (raw/unfurnished space) or an issue checklist with an annotated/edited image (furnished
space). Built for the NVIDIA DGX Spark hackathon — see `Claude.md` for the original spec.

## Architecture

- **Perception**: YOLO-World (open-vocabulary YOLO, `ultralytics`) detects doors, windows,
  mirrors, beams, and furniture via free-text class prompts — stock COCO classes don't include
  door/window/mirror, and fine-tuning a custom detector wasn't feasible in the sprint window.
- **Geometry** (`perception/geometry.py`, `perception/facts.py`): pure, deterministic Python —
  IoU, angle-between-centers, brightness, color-balance — implementing the 10 rules in
  `rules/rule_base.json`. No LLM ever computes geometry; it only narrates the `rule_verdicts`
  Python already decided.
- **Reasoning**: DeepSeek (`api_clients/deepseek_client.py`) narrates the facts into draft
  advice; Step-3.7-Flash, via StepFun's cloud "Step Plan" API (`api_clients/step_client.py`,
  OpenAI-compatible), formats that draft into strict schema JSON (`schema.py`). Step never sees
  raw geometry either.
- **Generation**: Wan2.7-Image (Alibaba DashScope, `api_clients/wan_client.py`) — text-to-image
  for Scenario A, image-to-image editing for Scenario B, falling back to a color/texture-change
  prompt if literal object relocation fails (diffusion models are unreliable at strict spatial
  moves).
- **Audio**: SenseVoice (DashScope) for mic input, `edge-tts` (local, free) for spoken advice.
- **Orchestration** (`orchestrator.py`): a linear script, not an agent framework — the UI's
  Raw/Furnished toggle hard-codes which path runs.

## Setup

```bash
uv sync
cp .env.example .env   # fill in DEEPSEEK_API_KEY, DASHSCOPE_API_KEY, and STEP_API_KEY
```

Step-3.7-Flash runs via StepFun's cloud "Step Plan" API (`https://api.stepfun.com/step_plan/v1`,
OpenAI-compatible) — no local download or GPU serving needed. An earlier version of this project
attempted local `llama.cpp` serving on this single DGX Spark, but even the smallest workable
quant (~92GB) was too slow to download in the sprint window; the cloud API sidesteps that
entirely.

### Run

```bash
./one-click-start.sh   # launches the Gradio app (all reasoning/generation is via cloud APIs)
```

## Smoke testing

Test each stage independently before relying on the full pipeline:

```bash
uv run python scripts/smoke_test.py --mode perception fixtures/raw_room.jpg
uv run python scripts/smoke_test.py --mode step
uv run python scripts/smoke_test.py --mode deepseek
uv run python scripts/smoke_test.py --mode wan
uv run python scripts/smoke_test.py --mode audio
uv run python scripts/smoke_test.py --mode e2e-a fixtures/raw_room.jpg "master bedroom"
uv run python scripts/smoke_test.py --mode e2e-b fixtures/furnished_bedroom_0.jpg
```

## Known limitations

- **Latency**: two sequential LLM calls (DeepSeek narration, then Step formatting) plus a Wan
  image generation call means end-to-end timing may exceed the spec's original 10s target.
  `orchestrator.py` logs per-stage timing so slow stages are visible rather than discovered live
  during a demo.
- **Mirror-facing detection** (`mirror_facing_bed_or_door` rule) uses 2D bbox IoU as a proxy for
  "facing" — no depth/pose estimation. It will miss same-2D-non-overlapping-but-actually-facing
  cases and can occasionally false-positive.
- **Beam detection** via the horizontal-dark-band heuristic is noisy on real photos (shadows,
  light fixtures, and crown molding can all look like dark horizontal bands) — treat it as
  advisory, lower-confidence output.
- **Photography assumption**: the door/window-angle and wealth-corner heuristics assume roughly
  frontal, room-spanning photography; angled/fisheye phone shots will degrade accuracy.
- **`SEND_IMAGE_TO_LLM`** defaults to `false` — the Python-computed `rule_verdicts` are already
  the authoritative grounding, and sending the raw photo to the VLM adds vision-token prefill
  cost that directly fights the latency budget.
