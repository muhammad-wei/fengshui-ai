# Fixtures

Real test photos for `scripts/smoke_test.py` and `tests/`, all verified end to end against the
live cloud APIs (not just unit-tested in isolation):

**Scenario A (raw/unfurnished space)**
- `raw_room.jpg` — an unfinished concrete space with exposed pipes; no door/window visible in
  frame, so perception correctly returns 0 detections and the LLM falls back to general advice.
  Useful as a "sparse facts" edge case.
- `villa_raw_space.jpeg` — an unfinished villa room with floor-to-ceiling windows and a glass
  door clearly visible. **Known edge case**: the window is a grid of many individual panes, each
  scoring just under the default 0.15 confidence threshold (measured ~0.10–0.15 per pane at
  `conf=0.03`), so it also detects 0 objects at the default threshold — worth knowing about
  before using this as a "clean detection" demo photo.

**Scenario B (furnished space)**
- `furnished_bedroom_0.jpg` — a real furnished bedroom (bed, window with curtains, nightstands,
  doorway to a study nook). The best-performing fixture: real `bed` detection at 0.82 confidence,
  and the actionable-instruction fix to `orchestrator.py` was verified against this exact photo
  (Wan visibly added a bookshelf + floor lamp to the low side of the room).
- `furnished_living_room.jpg` — sofa, coffee table, framed wall art.
- `entrance_room.jpg` — entryway with a door, bicycle, cabinets.
- `villa.jpeg` — a grand, fully furnished villa foyer (chandeliers, staircase, mirror, fireplace)
  — detects `potted plant`, `desk`, `chair`.
- `bookshelf.jpg` — a close-up decor shelf, not a full room; useful as a robustness/edge-case
  test (little to no relevant geometry to reason about) rather than a realistic demo photo.
- `inside_garden.jpg` — an indoor plant corridor; also an edge case rather than a typical
  bedroom/living-room shot.

All eight photos have real, non-degenerate YOLO-World bounding-box looseness and detection-miss
behavior baked in — that's intentional. They're kept as-is (not cherry-picked for clean results)
so smoke tests reflect real-world accuracy, not a curated best case.
