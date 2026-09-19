# Junction tests — 2026-09-19

Executed `scenarios/town01-junctions.json` on the RTX 5080 16 GB, with native
Windows CARLA 0.9.16 / Town01 and Qwen inference in WSL2. NF4, SDPA, 10 Hz
simulation, replanning every five ticks, 4 m/s target speed cap, seed 42 and a
45-second simulation limit. Qwen receives 1.5 seconds of autopilot warmup.
All episodes were visible with a following spectator camera.

## Qwen outcomes

| Scenario | Outcome | Model distance | Collisions | Lane events | Runtime |
| --- | --- | ---: | ---: | ---: | --- |
| Left, clear | Time limit; 15.3% route progress | 6.1 m | 0 | 0 | Normal exit |
| Right, clear | Reached destination | 55.1 m | 0 | 6 | Normal exit |
| Left, background traffic | Reached destination | 53.7 m | 0 | 5 | Native shutdown abort |
| Left, background traffic, wet | Reached destination | 57.0 m | 0 | 0 | Normal exit |

Endpoint completion uses the existing 3 m tolerance, corresponding to about
97% of the sampled route length. The wet case satisfies this suite's limited
clean-completion criterion: destination reached, zero collision/lane events,
zero rejected plans, normal process exit and verified cleanup. This does not
establish traffic-rule compliance or general driving reliability. The wet run
also cut inside the planned bend, reaching about 3 m nearest-route-sample
deviation despite zero lane events; the route overlay makes this visible.

All four autopilot reference runs reached the destination with zero collision
or lane events. The dry-traffic reference also aborted during native shutdown;
the other references exited normally. All Qwen runs reported zero rejected
trajectories. The right-turn Qwen lane events comprised five `Broken` marking
events and one `NONE` event. Counts are raw sensor callbacks, not scored traffic
infractions.

The left-clear vehicle was observed at a red light in two live diagnostic
snapshots. Its full episode predates signal-state logging, so we cannot attribute
the entire timeout to red-light waiting or to a planning/controller failure.
Subsequent runs log signal state every frame. The wet Qwen episode had 106
frames (10.6 simulation seconds) stopped below 0.1 m/s in a red-light trigger.

## Traffic exposure and comparison limits

Eight background vehicles were successfully spawned in the traffic references
and wet Qwen episode. The dry Qwen episode's final metadata was lost to its
native abort; its requested conditions and recorded proximity are retained.
Nearest recorded vehicle-centre distances were 22.9 m in dry Qwen and 20.6 m in
wet Qwen, with no frames within 15 m. Each traffic reference briefly approached
to 11.8 m. These are background-traffic tests, not close cross-traffic, cut-in or
pedestrian-interaction tests.

This was an initial engineering test series, not a controlled weather ablation:
there is one run per condition, model/simulator determinism is not established,
and runtime fixes were introduced during the suite. In particular, the first
left reference used a map reload, after which the Windows simulator became
unavailable during the next reload. The suite was resumed with a map-reuse and
traffic-light-reset procedure. Do not interpret the better wet outcome as proof
that wet weather improves the model.

## Runtime findings and fixes

The background-traffic runs exposed a CARLA native exception during shutdown:
`trying to operate on a destroyed actor`. Session cleanup now unregisters
background vehicles from Traffic Manager before destruction. Both subsequent
wet episodes exercised that fix and exited normally. Earlier failed runs and
exit codes remain in the results; they have not been replaced with successes.
Episode metadata is now checkpointed before driving so a native abort does not
lose all scene provenance. The summary separates route completion, process exit
and verified cleanup.

After the suite, a fresh server snapshot confirmed asynchronous mode, zero
vehicles and zero sensors. CARLA was left open. The test suite passes 34 unit
checks, including traffic cleanup ordering, weather restoration after a cleanup
failure, marking details, and avoiding a clean-completion label for lane events.

## Artifacts and reproduction

Artifacts are under `outputs/junction-suite-001/`:

- `report.html`: local replay gallery with references and Qwen episodes.
- `route-comparison.png`: planned routes and actual vehicle paths.
- `results.json`: final metrics, runtime status and traffic/signal exposure.
- `results-initial.json`: original aggregate retained before adding richer metrics.
- Each episode: front replay, camera frames, ego states, route and predictions.
- `cleanup-check.json`: independent final simulator-state check.

See [scenario testing](scenario-testing.md) for the command and scope. Each run
needs a new output directory, or `--resume` to skip already-recorded results.
The next useful expansion is a scripted encounter with a known crossing vehicle
or stopped lead car, plus traffic-light violation scoring and repeated seeds.
