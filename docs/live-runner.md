# Live runner

The runner now connects synchronized CARLA observations to Qwen predictions and
a trajectory tracking controller. The control-loop unit tests use a simulated
client. Real Windows CARLA recording, synchronized camera history, concurrent
shadow inference and a moving handover have now been exercised; see
[integration results](windows-integration-results.md) and
[quantized live results](quantized-live-results.md). Broader driving quality and
controller tuning remain open.

## Setup and modes

Run `bash scripts/setup.sh` to refresh the CLI and dependencies. The script fetches
four official CARLA routing helper files and their license at source revision
`294096eb1c38eabf246e4f3a9cdab704e33a7f4c` (0.9.16). Helpers stay under ignored
`vendor/carla-agents`; `networkx` is installed with the CARLA extra. A full CARLA
source checkout or server download is not required for these helpers.

Start the Windows CARLA 0.9.16 server as described in the README. Substitute the
Windows host IP below (or 127.0.0.1 when mirrored networking supports it).

```bash
.venv/bin/drive-run --host WINDOWS_HOST_IP --list-spawns
```

Select origin/destination indices from that map. Examples below use 0 and 10;
they are not guaranteed to form a supported route on your map. GlobalRoutePlanner
builds a 2-metre route. Lane changes are rejected because the model adapter only
expresses straight/left/right junction commands. A monotonic route cursor supplies
the upcoming command within 20 metres and prevents jumping across distant route
intersections. The prototype assumes planar roads.

| Mode | Vehicle control | Qwen resident on GPU? |
| --- | --- | --- |
| `record` | Traffic Manager follows the supplied route | No |
| `shadow` (default) | Traffic Manager follows the supplied route; Qwen proposals logged | Yes |
| `closed-loop` | Qwen trajectories tracked with pure pursuit and speed PID | Yes |

```bash
# Record a route with dynamic navigation commands.
.venv/bin/drive-run --host WINDOWS_HOST_IP --mode record \
  --spawn-index 0 --destination-index 10 --seconds 60 \
  --output outputs/routed-recording

# Observe the model while autopilot drives.
.venv-qwen/bin/drive-run --host WINDOWS_HOST_IP --mode shadow \
  --precision nf4 \
  --spawn-index 0 --destination-index 10 --seconds 60 \
  --output outputs/shadow-001

# Give the experimental controller the vehicle.
.venv-qwen/bin/drive-run --host WINDOWS_HOST_IP --mode closed-loop \
  --precision nf4 --warmup-driver autopilot \
  --spawn-index 0 --destination-index 10 --seconds 25 --max-speed 3 \
  --output outputs/closed-loop-001
```

Use a fresh output directory for each run and a dedicated CARLA server without
other clients controlling ticks. `--mode record` works in the lightweight `.venv`;
the model modes need `.venv-qwen`. Server/client versions must match exactly.

On the 16 GB GPU, use `--precision nf4` for concurrent experiments. It quantizes
248 language-model Linear modules to 4-bit NF4 with double quantization and BF16
compute. Vision, embeddings/output head and the planning expert remain BF16.
The loader checks module coverage and protected-module dtypes. `bf16` remains
the default for reproducible baseline runs. Quantization changes predictions and
is not guaranteed to preserve driving quality. Town01 with NF4 has been tested;
larger maps/settings may still exhaust VRAM. There is no remote inference or CPU
offloading in this runner.

## Timing and controller

- Simulation and control run at 10 Hz, with physics substeps. Only the runner
  advances the world; inference pauses simulation time.
- The first 16 observations establish 1.5 seconds of motion history. Closed-loop
  mode holds the brake by default. `--warmup-driver autopilot` instead supplies
  moving history, then disables autopilot before the first model command.
  Handover frame, per-step driver, total distance and post-handover model distance
  are logged separately. Shadow/record mode uses autopilot throughout.
- Inference runs every 5 ticks by default (`--replan-ticks 1..10`). It uses the
  previously tested small image sizes, SFT direct planning and one trajectory.
- Predicted XY waypoints are transformed into world coordinates using the pose
  at prediction time. At every control tick, they are transformed into the new
  ego frame. Speed is derived from the plan's 0.1-second waypoint spacing.
- Pure pursuit chooses steering, using wheelbase and steering angle limits from
  CARLA vehicle physics. PID tracks speed with a default 8 m/s cap. Gains are
  provisional and speed-dependent steering curves are not modeled.
- No plan, expired plans and stationary predictions command braking. Non-finite,
  malformed or discontinuous predictions terminate a closed-loop episode.
  Shadow mode saves them with a `validation_error`, invalidates the proposed
  control and continues autopilot; `rejected_plans` counts them. NaN/Inf values
  are written as JSON null for diagnosis. Inference/sensor failures still abort.
- Collision events, route deviation beyond 8 metres (`--max-deviation`), reaching
  the destination or elapsed duration end a run. Inference/sensor errors stop
  autopilot, request braking and release actors. Cleanup attempts all actors and
  world settings, even if an individual destroy call fails.

This controller has not been demonstrated to obey traffic rules or avoid
obstacles. It has no separate obstacle detector or traffic-light override. Qwen
must infer those decisions from its observations. The stop conditions above are
experiment controls, not a validated autonomous-driving safety system.

## Outputs

| File | Contents |
| --- | --- |
| `metadata.json` | Settings, camera mounts, source of commands, setup/cleanup errors |
| `route.json` | Planned world XY path and junction commands |
| `frames.jsonl`, camera folders | Timestamped ego states and RGB PNGs; compatible with offline benchmark |
| `plan-<frame>.json` | Raw ego-frame waypoints, origin, timestamp, command, inference timing/memory, validation error |
| `steps.jsonl` | Route progress, speed, proposed control, whether applied, driver for next tick, sensor events |
| `summary.json` | Termination reason, progress, rejected plans, event counts, handover/model distance, timing/errors |

`complete` in metadata means the runner exited normally; `summary.status`
distinguishes destination completion, duration limit, collision and route
departure. Event counts are raw sensor events, not CARLA Leaderboard metrics.
PNG recording occurs each tick and can use significant storage on long runs.

## Local checks without CARLA

```bash
.venv/bin/python -m pytest -q
.venv-qwen/bin/python scripts/smoke_live_adapter.py
```

The second command reconstructs a CARLA-convention history from the upstream
recorded demo, sends its camera images through the live adapter, runs the actual
model, and produces a control proposal. It writes a report and trajectory under
`outputs/live-adapter-smoke`. It does not use a running simulator or prove
closed-loop driving quality.

On 2026-09-19 the real-model smoke test passed: a finite `(50, 3)` trajectory,
14.19 seconds for the cold inference and 11.90 GiB peak reserved memory. The
controller proposed braking for that scene. This is a pipeline validation on a
recorded observation, not evidence of successful vehicle motion in CARLA.
