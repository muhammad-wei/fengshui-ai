# Golden fallback images

`orchestrator.py` falls back to these if the live Wan2.7-Image call fails, so the demo never
shows a broken/missing image:

- `golden_scenario_a.jpg` — a pre-generated "after" layout render for the exact photo used in
  the recorded demo (Scenario A).
- `golden_scenario_b.jpg` — a pre-generated annotated/edited image for the exact photo used in
  the recorded demo (Scenario B).

Generate these during H10-H12 (Freeze & Demo Prep) using the real Wan API against your chosen
demo photos, once `DASHSCOPE_API_KEY` is set — see the "Wan API returns a bad/unusable image"
risk mitigation in the plan doc.
