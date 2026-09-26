# Qwen evaluation

The dense dashboard now defaults to safety off, one local sample, and no fallback.
Candidate zero is executed without CARLA map ranking or guard overrides. The UI
labels this **Qwen evaluation**. Assisted runs remain available explicitly and
must not be reported as Qwen performance.

Qwen produces a 50-point trajectory. Pure pursuit and PID translate it into
steering/throttle/brake, with a 4 m/s speed cap. CARLA autopilot supplies the first
1.5 seconds of history. Inference pauses simulation; these runs do not measure
real-time driving capability. CARLA ground truth is still used for scoring and
episode termination, not path selection in evaluation mode.

The local deployment uses NF4 language-model weights and the small image profile;
it is not an unquantized model baseline. The custom CARLA camera rig is another
domain difference. The adapter audit against the pinned vendor scene/data contract
found consistent 16-state 10 Hz history, four camera times per view, current-ego
X-forward/Y-left coordinates, left-positive heading radians, and navigation encoding.
This does not establish that CARLA images match the training distribution.

Every local inference saves `planner-inputs/input-FRAME.json` and content-addressed
lossless PNGs **before** model execution. These contain the exact RGB images and
numeric scene payload before Qwen resizing/preprocessing, plus model loading
settings. `qwen_drive_carla.input_archive.load_payload(path)` verifies pixel hashes
and restores a payload for `QwenPlanner.plan_payload`. Surround-camera dashboard
previews are separate from these planner inputs. Remote inference is not covered
by this exact local-input archive.

Use identical archived payloads for precision/resolution comparisons before
claiming a model improvement. Then compare unassisted closed-loop runs using the
same scenario and report collision, road departure, progress, and plan failures.
Malformed trajectories remain failures; numerical tracker validation is not a
CARLA safety filter. Earlier guarded or fallback-assisted results are not an
unassisted baseline. Model weights have not been changed.

Launch:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen/bin/python scripts/run_dense_debug.py \
  --local-planner --candidate-count 1 --safety-mode off --fallback none \
  --start-paused --output outputs/qwen-evaluation
```

## First recorded unassisted check — 2026-09-26

`outputs/local-qwen-evaluation-20260926-001/episode-002`: left the driving
carriageway at 5.1 simulation seconds; seven plans, two lane-invasion events,
zero collisions, zero fallback plans and zero safety interventions. Cleanup
reported no errors. All seven archived inputs passed pixel-hash replay checks.
This is one NF4/small run, not an aggregate score or a full-precision baseline.
