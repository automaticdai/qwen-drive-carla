# Dense driving diagnosis and CARLA-assisted braking — 2026-09-26

The saved local NF4/small episode `outputs/local-preview-20260926c/episode-001`
left the driving carriageway after 10.9 simulation seconds, with seven raw lane
sensor events, no collision, and 19 Qwen plans. Offline examination points to
unsafe proposed paths as the main source of this departure. This is evidence
from one episode, not a general assessment of the model.

## Evidence

The recorded vehicle yaw moved from 147.7° at handover to 122.2° at 6.1 seconds,
while the nearest lane direction at that point was approximately 144°. The
vehicle crossed from lane −2 into lane −1 and continued toward the road edge.

Over the next 0.5 seconds after each prediction, the mean distance from the
actual actor position to the proposed path was **0.010 m**, with a maximum of
**0.051 m**. This measures geometric path tracking; it does not establish that
trajectory timing or predicted speed was accurate. These observations do not
support replacing the steering controller as the first fix for this episode.
The separate time-aligned position error averaged 0.069 m and peaked at 0.264 m
over the same short execution windows.

The new guard rejects all 19 saved proposals. The first proposal, at handover
(1.6 seconds into the episode), contains reverse motion at its 2.9-second
horizon. Independently, its expanded footprint leaves the allowed road at
approximately 3.81 seconds into the proposed trajectory. The road checks use
the installed Town05 OpenDRIVE file through CARLA's offline map API, without
advancing or changing the simulator.

Old recordings do not include exact actor bounding boxes or other-vehicle
snapshots. This replay assumes a centered 4.8 × 2.0 m ego footprint plus the
guard's 0.25 m margin, and default steering geometry. Collision checking cannot
be reconstructed for this old episode. Live guarded runs use actual CARLA
bounding boxes and wheel geometry and save other-vehicle snapshots.

## Implemented behavior

`scripts/run_dense_debug.py` now defaults to `--safety-mode carla`, for both
local and remote inference. This is **Qwen with CARLA ground-truth assistance**.
The ordinary `drive-run` runner is unchanged.

- Check all five seconds of a new trajectory for reversals, gross sideways
  motion, excessive yaw change, road departures and predicted vehicle overlap.
- Use oriented vehicle bounding boxes with actor-local offsets and rotations;
  expand the ego box by 0.25 m. Check its perimeter against the union of the
  scenario's three driving lanes, allowing lane changes inside that union.
- Interpolate checks between predicted samples to reduce missed crossings.
  Other vehicles follow constant world-velocity predictions.
- Reject unsafe plans, invalidate the previous plan, and brake. Replan at the
  existing 0.5-second simulation cadence so a later valid proposal can resume.
- On every model control tick, check a bicycle-model stopping path using actual
  speed and proposed steering, a 0.3-second response allowance, assumed 3 m/s²
  acceleration while throttle is applied, and 3 m/s² braking. This can intervene
  between predictions or when a stationary proposal contradicts actual motion.
- Reset the speed-controller integral during overrides. Prefer straight braking;
  if that stopping path is obstructed, evaluate braking with proposed steering.
- Log the assistance source, rejection reason, predicted hazard time/actor,
  proposed/applied controls, configuration and per-tick vehicle snapshots. The
  dashboard labels the assisted driver and displays guard decisions.

`--safety-mode off` reproduces unassisted behavior for comparisons. Do not score
assisted results as evidence of improved Qwen perception or planning.

## Validation and limits

Unit tests cover box transforms, adjacent lanes, crossing traffic, road edges,
inter-sample hazards, invalid/reversing/sideways paths, stopping distance and
recovery. A synthetic closed-loop braking test exercises repeated control
updates. Runner integration tests exercise rejected-plan braking, logging and
resumption in both assistance modes without loading a GPU model.
The full regression suite passed **96 tests** with localhost sockets enabled;
the dashboard JavaScript also passed a syntax check.

The full five-second checks are deliberately conservative: all saved plans
were rejected. The car may stop frequently until planning improves. This work
does not implement an overtaking decision process or candidate generation.
Constant-velocity traffic prediction, planar geometry and assumed braking are
approximations; pedestrians, traffic-signal compliance and arbitrary static
obstacles are outside this guard's scope. The stopping model is not a guarantee
that braking can avoid an already unavoidable incident.

CARLA RPC at the saved Windows host address timed out during this work, including
outside the network sandbox. **No new live closed-loop success is claimed.**
Offline rejection of an old plan does not establish the counterfactual outcome
after a brake intervention.

## Reproduce the diagnosis

```bash
MPLCONFIGDIR=/tmp/qwen-drive-matplotlib .venv-qwen/bin/python \
  scripts/diagnose_dense_episode.py outputs/local-preview-20260926c/episode-001 \
  --map /mnt/c/Users/autom/AppData/Local/CARLA/0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr \
  --output outputs/driving-diagnosis-NEW --plot
```

The command writes `report.json` with per-plan decisions and tracking metrics,
and `diagnosis.png` with road/path overlays and vehicle-versus-road yaw.
`--plot` requires matplotlib; the CARLA server and Qwen model are not needed.

Subsequent live work found and corrected an autopilot-to-brake command-delivery
bug and added multi-candidate selection plus optional lane following. See
[the follow-up implementation and live evidence](candidate-selection.md).
