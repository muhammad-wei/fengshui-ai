# AI Interior Feng Shui Consultant – 2-Day Extreme MVP Project Specification (Spec)

> For: NVIDIA DGX Spark Hackathon · Final Sprint Edition  
> Core Principle: **Demoable > Perfect** | **Lightweight > Full-Featured** | **Deterministic > Mysticism**  
> Version: v2.0 | Effective Date: July 20, 2026 (Immediate Execution)

---

## 1. Project Vision (Redefined)

Build an **indoor-environment-focused AI consultant** that delivers an end-to-end demo **within two days**. Targets two core user scenarios:

- **Scenario A (Raw/Rough Space)**: Upload floor plan / raw construction site photos → Output **furniture layout plan** and **functional zone recommendations** (Feng Shui guided).
- **Scenario B (Furnished/Decorated Space)**: Upload existing interior photos → Output **soft furnishing adjustment checklist** (e.g., move plants, add screens, change color tones) to resolve Feng Shui issues.

**Core Deliverable**: A Gradio web application. Upload a photo + select a scenario mode → receive a diagnostic report with both text and visuals within **10 seconds**.

---

## 2. Architecture Slim-Down (4 Agents → 2 Cores + 1 Lightweight Scheduler)

With only two days and precious DGX Spark VRAM, **eliminate the standalone Evaluation Agent and the heavy RAG vector database**. Merge into a **"Perception–Reasoning–Generation" three-stage pipeline**, with the Orchestrator controlling retries via code logic (max 1 retry), completely avoiding multimodal VLM-based scoring.

| Module | Responsibility | Model | Estimated VRAM |
| :--- | :--- | :--- | :--- |
| **Perception** | Identify interior **structural elements** (doors, windows, load-bearing walls/beams, existing major furniture) and **orientation** (based on window positions). | **YOLOv8n** (detection) + **MobileNetV3** (scene classification) | < 1.5 GB |
| **Orchestrator + Reasoning** | Takes over all reasoning, rule matching, and report generation. Embeds a built-in **"Interior Feng Shui Hard Rule Base"** (see Section 4), injected directly into the System Prompt as JSON. | **Qwen2.5-7B-Instruct (INT4)** local deployment | < 4 GB |
| **Generation** | **Scenario-specific**:<br>- Rough space: Generates a **2D floor plan SVG diagram** (no diffusion models).<br>- Furnished space: Only changes color tones/filters (using **OpenCV color transfer**) and overlays annotated text labels indicating "Suggested relocation here". | **OpenCV + Matplotlib** / Lightweight Stable Diffusion **only as optional fallback** | < 2 GB (or 0) |

> **VRAM Control**: Orchestrator forces **serial loading** – unload Perception after completion, then load the LLM, then load Generation. Peak VRAM usage controlled within **6–8GB**, ensuring DGX Spark runs smoothly without OOM crashes.

---

## 3. Precise Workflows for the Two Core Scenarios

### Scenario A: Raw Space → Output "Feng Shui Layout Plan"
1. **User Input**: Upload raw space photo (must include window positions) + enter "intended as master bedroom / study".
2. **Perception**: Identify windows (light sources), door openings, exposed beams (if visible), and columns.
3. **Reasoning (Core Logic)**:
   - Infer **orientation** based on window positions (window = "facing direction").
   - Consult rule base: Bed must have "solid back against empty front" – headboard against a solid wall, not facing window or door.
   - Desk should have "Azure Dragon on the left" (left side higher/brighter), avoid sitting with back to the door.
4. **Generation Output**:
   - Return a **2D top-view annotated diagram (Matplotlib-generated)**: mark "Recommended Bed Area", "Recommended Desk Area", "Wardrobe Shielding Zone" with different colors.
   - Include text explanation: "Bed is recommended against the east wall, because the window faces south, avoiding direct morning light on the sleeping area."

### Scenario B: Furnished Space → Output "Feng Shui Adjustment Recommendations"
1. **User Input**: Upload existing living room / bedroom photo.
2. **Perception**: Detect existing sofas, beds, TVs, mirrors, plants, sharp wall corners.
3. **Reasoning (Core Logic)**:
   - Check for "Piercing Sha" (door–window–door direct alignment) – via pixel-coordinate straight-line matching between detected door and window bounding boxes.
   - Check for "Mirror facing bed/door" (detect reflective areas of mirrors).
   - Check for "Beam pressing down" (detect horizontal dark line objects above the ceiling area).
4. **Generation Output**:
   - Mark problem areas on the original image with **red arrows / heatmaps**.
   - Output a "Three Do's & Three Don'ts" adjustment checklist (e.g., "Do: Add a frosted glass screen at the entrance; Don't: Position the sofa with its back to the main door").

---

## 4. "Interior Feng Shui Hard Rule Base" (Quantified, No Mysticism)

No more RAG from ancient texts. **Hard-code the Top 10 Golden Rules** directly in the Prompt, supplemented with Python logic validation:

| Rule Name | Quantified Criterion | Applicable Scenario |
| :--- | :--- | :--- |
| **Solid Back, Empty Front** | Headboard wall detected as **windowless & doorless** continuous wall surface. | Raw layout |
| **Open Bright Hall** | Obstacle-free area in front of sofa/bed > 40% of horizontal field of view. | Furnished adjustment |
| **Door–Window Direct Alignment** | Angle between door center and window center line < 15°. | Furnished adjustment |
| **Mirror Not Facing Bed/Door** | IoU between mirror BBox and bed/door BBox > 0.1 (considering reflection). | Furnished adjustment |
| **Beam Pressing Down** | Horizontal dark line objects detected above sofa/bed area. | Furnished adjustment |
| **Left High, Right Low** | Relative to seated orientation, average height of objects on the left > right. | Raw / Adjustment |
| **Color Balance** | Excessive warm tones (red/orange) → suggest cool tones (blue/green). | Furnished adjustment |
| **Plants to Neutralize Sha** | Recommend broad-leaf plants near sharp corners (wall corners). | Furnished adjustment |
| **Dynamic–Static Separation** | Dining table not adjacent to bathroom door (door distance detection). | Raw layout |
| **Wealth Area Should Be Bright** | The diagonal corner from the entrance ("Ming Wealth Corner") should have highest brightness/luminance. | Raw / Adjustment |

---

## 5. Two-Day Extreme Development Plan (Half-Day Precision)

| Time Slot | Task | Deliverable |
| :--- | :--- | :--- |
| **Day 1 – Morning** | Build Gradio skeleton + deploy YOLOv8n + fine-tune indoor detection (COCO pretrained; extract only person/chair/sofa/bed/door/window/mirror classes). | Interface that can upload and draw bounding boxes around doors, windows, sofas. |
| **Day 1 – Afternoon** | Deploy Qwen2.5-7B (INT4). Write System Prompt with the 10 rules from Section 4 + 2 Few-shot examples. Implement Scenario A (raw space) layout logic (SVG generation). | Scenario A demo working, returning text + SVG diagram. |
| **Day 2 – Morning** | Implement Scenario B (furnished) adjustment logic. Use OpenCV to draw arrows + rectangles on problem areas. Integrate lightweight color transfer (fallback to PIL color temperature adjustment if SDXL is not ready). | Scenario B demo working, returning annotated original image. |
| **Day 2 – Afternoon** | Integrate UI (mode toggle switch). Record a 3-minute demo video (highlighting both scenario transitions). Write README and architecture diagram (one-page PPT). | **Final Submission Package**. |

---

## 6. Technology Stack Minimalist Checklist (One-Line Install)

```text
# Base Environment (Python 3.10)
torch >= 2.0 (with CUDA)
ultralytics (YOLOv8)
transformers (Qwen2.5)
gradio (frontend)
opencv-python, matplotlib, pillow
# Strictly EXCLUDE: LangGraph, ChromaDB, LLaVA, DepthAnything (all cut)
```

---

## 7. Demo Script (The "5-Minute Highlight" Judges Want to See)

1. **Opening (1 min)**: "We focus exclusively on interiors, targeting the two most critical renovation stages."
2. **Scenario A Demo (2 min)**: Upload a raw bedroom photo (window facing south). System generates a **top-view layout diagram** in 3 seconds, showing bed against east wall, desk against west wall, with prompt: "Window faces south; bed on the east wall avoids early morning glare, consistent with Solid Back/Empty Front."
3. **Scenario B Demo (2 min)**: Toggle mode, upload a cluttered furnished living room photo. System red-boxes "Mirror directly facing bedroom door" and "Beam above sofa", with text suggestions: "Relocate mirror to the entrance-side screen; add a dropped ceiling to soften the beam presence."
4. **Closing**: Emphasize that everything runs locally on DGX Spark, keeping user data private and secure.

---

## 8. Risk Mitigation (Two-Day Special Edition)

| Risk | Sprint-Focused Solution |
| :--- | :--- |
| Qwen2.5 outputs erratic JSON formats. | Use `json_repair` library as fallback + force `temperature=0.1`. |
| YOLO misidentifies curtains as doors. | Add **area filtering** at reasoning layer: ignore objects exceeding a size threshold to avoid false Piercing Sha detection. |
| High-resolution images slow down inference. | Gradio frontend forces `resize=640x640` before passing to backend. |
| Raw space photos have no furniture → YOLO detects nothing. | In "Raw Space" mode, **automatically skip object detection** and reason purely based on door/window positions. |
| Generated SVG looks ugly and unprofessional. | Use `matplotlib` with `seaborn` color palettes to ensure clean, professional visuals. |

---

## 9. Final Deliverables Checklist (Complete in 2 Days)

- [x] **Runable GitHub Repository** (with `requirements.txt` and a `one-click-start.sh` script)
- [x] **Gradio Live Demo** (local :7860 port)
- [x] **3-Minute Demo Video** (voice-over explaining both scenarios)
- [x] **One-Page Architecture Diagram** (PPT screenshot, highlighting "local + lightweight")
