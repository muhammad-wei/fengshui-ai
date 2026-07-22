# Fixtures

Test photos for `scripts/smoke_test.py` and `tests/`:

- `raw_bedroom.jpg` — an empty/unfurnished room with at least one visible window and door, for
  Scenario A (`perception` should detect `window`/`door`, `open_area_ratio`/`solid_back_empty_front`
  should have something to evaluate).
- `furnished_room.jpg` — a furnished living room or bedroom, ideally with a mirror, a sofa/bed,
  and ceiling visible above them, for Scenario B (`mirror_facing_bed_or_door`, `beam_pressing_down`).

Not committed yet — add real photos here before running the perception/e2e smoke tests.
