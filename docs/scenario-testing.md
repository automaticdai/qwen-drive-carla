# Junction scenario suite

`scenarios/town01-junctions.json` defines four CARLA 0.9.16 / Town01 cases:

| Case | Route | Conditions |
| --- | --- | --- |
| left-clear | Spawn 241 → 225, approximately 65.6 m | Clear midday, no background traffic |
| right-clear | Spawn 11 → 153, approximately 62.4 m | Clear midday, no background traffic |
| left-traffic | Same left turn | Eight requested nearby autopilot vehicles |
| left-traffic-wet | Same left turn and traffic seed | WetCloudyNoon weather preset |

The junction begins approximately 18–19 m into each route. These are short
junction tests, not a comprehensive urban-driving benchmark. Background traffic
follows Traffic Manager routes; cross-traffic encounters are not scripted or
guaranteed. Vehicle centres within 15 m provide encounter evidence, not a
collision-risk or following-distance metric. Weather is a CARLA visual preset;
this implementation does not change tire friction.

Run from the repository root with a dedicated, already running CARLA server:

```bash
.venv-qwen/bin/python scripts/run_scenarios.py \
  --host WINDOWS_HOST_IP \
  --output outputs/junction-suite-NEW \
  --reference --follow-camera --replays --seconds 45
```

`--reference` runs an autopilot episode before each Qwen episode, with the same
endpoints, seed, speed cap and weather. The suite loads Town01 if needed and resets traffic lights between episodes.
It reuses the map because a repeated map load caused the local Windows server
to become unavailable during the first test attempt. The suite refuses to run in a synchronous world or one with existing
vehicles/sensors, and stops if cleanup reports errors. All spawned traffic is
owned and destroyed by the episode; weather and world settings are restored.
The spectator viewpoint remains at the final ego position. Use a fresh output
directory. The suite runs sequentially to fit the 16 GB GPU; Qwen uses NF4 and a
4 m/s target speed cap. Every model episode starts with 1.5 seconds of autopilot
history before handover. Configurations and seeds are repeatable; bit-for-bit
simulation or model determinism has not been established.

Watch the visible simulator with `--follow-camera`. `--replays` uses an installed
`ffmpeg` to produce `front-replay.mp4` in each episode directory, at 10 frames per
second. Replay time is simulation time and excludes inference pauses. No video
is produced if ffmpeg is unavailable. The lightweight `--mode record` reference
uses autopilot; it must not be interpreted as model driving.

`results.json` aggregates termination status, route progress, model-controlled
distance, collision and lane events, rejected predictions, traffic proximity,
commands supplied to the model, cleanup errors and subprocess exit codes.
`clean_completion` requires destination completion with zero collisions, lane
invasion events and rejected plans, plus available signal scoring with zero red-light
or ambiguous signal crossings. Older unscored logs cannot earn a clean completion
when summarized with the current code. Merely seeing a turn command does not prove
a turn was executed. The route geometry, frame states and replay establish
where the car actually travelled. Lane-event details now retain marking types;
collision details retain the other actor ID/type. These are sensor counts,
not CARLA Leaderboard infractions or verified traffic-rule compliance.

Per-episode logs preserve setup/inference failures. Ordinary model failures are
retained and the suite proceeds to the next case; interruption stops the suite.
For an individual experiment, the normal runner also accepts these settings:

```bash
.venv-qwen/bin/drive-run --host WINDOWS_HOST_IP --mode closed-loop \
  --precision nf4 --warmup-driver autopilot --spawn-index 241 \
  --destination-index 225 --traffic 8 --weather WetCloudyNoon \
  --follow-camera --max-speed 4 --seconds 45 --output outputs/wet-left-NEW
```

Signal scoring is experimental: the front-bumper midpoint must cross a lane-width
segment at a CARLA `get_stop_waypoints()` location in the lane's forward direction.
Red at both bounding ticks produces a `red_light` event and terminates the episode
as `red_light_violation`; a signal change during the crossing is marked ambiguous.
All traffic-light actor states and stop geometry are retained. Outside a vehicle's
light trigger, its `traffic_light_state` is null, not CARLA's default green.
This does not prevent violations, score yellow-light behavior, cover every possible
off-lane crossing, or establish full traffic-rule compliance. There is no red-light
braking override. The metric needs live validation against map geometry.

Pedestrian crossings, scripted cut-ins, emergency braking and longer multi-junction routes are not implemented by this
suite. It establishes turning and nearby-traffic failure cases before adding
those more demanding interactions.

Use `--resume` with the same arguments and output directory to skip cases already
listed in `results.json`. An incomplete episode directory is never overwritten;
inspect and move it aside before retrying. Build a local replay gallery with:

```bash
.venv/bin/python scripts/scenario_report.py outputs/junction-suite-NEW
```

The first eight live runs and runtime findings are documented in
[junction results](junction-results.md). A native exit failure disqualifies a
clean completion even when the driving summary reports reaching the destination.
Final cleanup is recorded separately. For a route comparison plot:

```bash
MPLCONFIGDIR=/tmp/qwen-mpl .venv-qwen/bin/python scripts/plot_scenarios.py outputs/junction-suite-NEW
```
