# Qwen performance investigation — 2026-09-27

The strongest evidence is that the deployed NF4/small planner proposes unsafe
paths, and the vehicle follows them closely. Separately, inference exceeds the
runner's real-time planning budget. The evidence does not isolate why the model
produces these paths: precision, image resolution, camera/domain differences,
and closed-loop state feedback remain unresolved contributors.

## Fresh offline checks

Replayed the three saved unassisted episodes in
`outputs/local-qwen-evaluation-20260926-001` against the installed Town05
OpenDRIVE map using `scripts/diagnose_dense_episode.py`'s `diagnose` function.
No simulator ticks or new model inference were performed.

| Episode | Recorded outcome | Plans | Mean / max geometric tracking error | Warm inference mean |
| --- | --- | ---: | ---: | ---: |
| 002 | Road departure at 5.1 s | 7 | 0.018 / 0.051 m | 0.668 s |
| 003 | Road departure at 5.0 s | 7 | 0.018 / 0.045 m | 0.675 s |
| 004 | Guardrail collision at 9.0 s | 15 | 0.014 / 0.061 m | 0.660 s |

These are repeated runs of one setup, not independent scenario coverage.
All used one candidate, NF4 SFT/direct/small, safety off, and no fallback.
The final ego positions in episodes 002 and 003 independently map to no driving
waypoint, confirming that their departure status was not solely caused by the
runner also checking the lead vehicle's road ID.

Tracking error is nearest distance to the active predicted polyline over the
next up to 0.5 simulation seconds. Mean time-aligned position errors were
0.115, 0.103, and 0.077 m respectively, with maxima below 0.247 m.
This supports investigating model proposals before steering-controller tuning;
it does not establish perfect speed or heading tracking.

The offline guard rejects all 29 full five-second proposals. Each episode's
first proposal reverses at a 2.9-second horizon; independently, its expanded
footprint crosses the road boundary at approximately 3.69–3.82 seconds.
These checks assume a centered 4.8 × 2.0 m footprint and a 0.25 m margin because
the unassisted recordings lack exact footprint metadata. Other-vehicle collision
replay is unavailable. Long-horizon rejection does not prove that braking would
have prevented the recorded outcome.

Reports: `outputs/performance-investigation-20260927/episode-002.json`,
`episode-003.json`, and `episode-004.json`.

## Runtime limitation

The runner replans every 0.5 simulated seconds, but warm model inference alone
takes 0.66–0.68 wall seconds. Local request overhead averages another
0.18–0.22 seconds, including input archiving and scene preparation; these stages
were not individually profiled. Episode 004 averages 0.837 seconds per complete
local request, before other simulation/dashboard work. Episode 002's first
inference takes 5.07 seconds.

The synchronous runner pauses simulation during inference, so this latency
reduces wall-clock throughput rather than aging the scene while the car moves.
It does not explain the recorded departures through stale moving-world inputs.
Episode 003's 5,885-second wall duration includes dashboard/session elapsed
time and must not be interpreted as model inference latency.

## Inputs and configuration

- All 29 archived payloads successfully reloaded with verified image hashes,
  16 history poses, and four frames per view. The focused adapter, controller,
  and input-archive suite passed: **18 tests**. Inspection of the pinned vendor
  scene contract found no command-order mismatch: straight is navigation index
  zero and driving command `[0, 1, 0, 0]`.
- The model receives three front-facing views. The six-camera dashboard/BEV
  display is separate: local BEV is disabled, and remote BEV explicitly reports
  `feeds_planner=False`. The displayed perception outputs do not correct plans.
- The small profile explicitly resizes history to 320×192 and current frames
  to 640×384. The pinned upstream default pixel budgets are 174,080 and 921,600
  respectively, substantially larger. These are preprocessing differences,
  not evidence that resolution alone causes the failures.
- NF4 quantizes 248 language-model linear modules while vision and planning
  remain BF16. The earlier same-scene comparison in
  [quantized live results](quantized-live-results.md) measured a 0.80 m mean
  trajectory difference versus BF16; difference is not an accuracy score.
- The custom CARLA camera mounts and rendered imagery have not been validated
  as matching the training distribution. The dense runner supplies a fixed
  straight command; it does not provide an explicit overtaking objective.
- All three handovers start at about 0.60 m/s with abrupt acceleration/braking
  in the initial history. This is another condition to isolate, not a proven
  input corruption.

Earlier BF16/high and NF4/small closed-loop comparisons also changed rendering
and scene observations. They cannot identify which change improved behavior.
The current evaluation intentionally executes candidate zero without map-based
ranking; adding guards or lane-following fallback would change the evaluated
system rather than demonstrate improved Qwen planning.

## Next discriminating experiment

Replay identical archived inputs through NF4/small and BF16/small with the same
noise seed, then compare raw trajectory feasibility and road alignment. To
isolate resolution, capture a shared high-resolution input set and run both
small and high preprocessing on it; upscaling the existing 640×384 archives
cannot recover missing detail. Hold all other settings fixed, then validate
promising profiles in repeated unassisted closed-loop scenarios.

For throughput, separately profile preprocessing/archive I/O, vision, language,
and the planning expert before changing kernels. The installed runtime uses
SDPA and lacks the optional native acceleration extensions described in the
README, but no current per-stage profile establishes their contribution.

No new GPU comparison was run: `nvidia-smi` reported GPU access blocked by the
operating system in this session. This investigation changed documentation only;
model weights and runtime behavior are unchanged.

Follow-up: the requested [BF16 replay](bf16-replay-results.md) completed on all
29 inputs after GPU access was enabled outside the sandbox. BF16 removed the
three initial reversal defects, but both precisions still failed all 29
independent road-boundary checks.
