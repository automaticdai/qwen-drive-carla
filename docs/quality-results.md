# Quality-profile validation — 2026-09-19

The selected cloud profile is now **BF16 SFT, direct planning, high-resolution
images**. The official BF16 RL reasoning profile was also deployed and tested,
but its rollout stalled, so it remains an explicit experimental option rather
than the default. Both use the compatible Qwen-Drive-1.0 4B backbone. This work
does not integrate a larger general-purpose Qwen model as a driving agent.

Local CARLA 0.9.16 was launched with `-quality-level=Epic`, a visible
1920×1080 window, and three 1280×768 RGB cameras. The real Windows child process
command line confirmed the Epic and resolution flags. Town01 was selected
through the Python API; the packaged launcher did not honor the attempted map
argument, which was removed from the launcher script. Standard rendering was
improved; hardware ray tracing was not enabled or verified.

## Driving comparison

All cloud episodes below used the same short Town01 route (spawn 2 → 133),
3 m/s target speed cap, 1.5 seconds of initial autopilot history and 25-second
simulation limit. Destination completion uses the existing 3 m tolerance.

| Profile | Outcome | Model-controlled distance | Collisions | Lane events | Mean request time |
| --- | --- | ---: | ---: | ---: | ---: |
| Earlier NF4 SFT, small inputs, Low rendering | Completed | 33.18 m | 0 | 2 | 3.24 s |
| BF16 RL reasoning, high inputs, Epic | Time limit; 21% progress | 4.99 m | 0 | 0 | 7.76 s |
| **BF16 SFT direct, high inputs, Epic** | **Completed** | **32.02 m** | **0** | **0** | **6.57 s** |

The selected SFT run took 19.9 simulation seconds and 356.49 wall-clock seconds,
including recording and transport. It generated 37 trajectories with no rejected
plans. Maximum distance to the nearest sampled route point was 1.19 m; this is
not a pure lateral-deviation metric. Warm requests averaged 6.48 seconds, mean
server inference was 3.75 seconds, and request bodies averaged 7.23 MB.

RL generated valid trajectories but repeatedly requested little immediate
motion, even when its generated sentence suggested accelerating. It ran for the
full 25 simulated seconds (515.95 wall seconds), with 47 plans and no rejected
trajectories. Generated reasoning should not be treated as reliable evidence of
the model's actual controls.

This is one short-route attempt per configuration. The earlier NF4 comparison
also used different rendering and image resolution. These results do not
isolate the effect of quantization, resolution or expert training and do not
establish general driving improvement. Turns, close interactions and multiple
seeds still need testing with the selected quality profile.

## Memory and rendering evidence

- Selected BF16 SFT model: **14.80 GiB peak torch-reserved GPU memory** on the
  cloud L4 during the episode. This excludes other processes and CUDA overhead.
- BF16 RL reasoning: **12.38 GiB peak torch-reserved GPU memory** in its episode.
- Local RTX 5080: **7,322 MiB (~7.15 GiB)** maximum total reported GPU memory
  across 176 one-second samples during Epic recording / the initial RL test.
  This is a Town01 measurement, not a bound for larger maps.
- The current model view size increased from 640×384 to 1280×768. History views
  increased from 320×192 to 640×384. Field of view and mounts are unchanged.
- Client-side history resizing was checked against the upstream torchvision
  PIL BICUBIC resize on a real captured frame: pixels matched exactly.

The Low/Epic visual comparison uses different recorded positions and is an
illustration, not a controlled pixel comparison. Higher resolution and Epic
rendering do not alter the configured physical dynamics.

## State and artifacts

The cloud tmux service `qwen-drive-agent` and SSH tunnel remain running with
BF16 SFT/high inputs. Local CARLA remains open at Epic quality. A fresh snapshot
confirmed asynchronous mode, zero vehicles and zero sensors after the test.
The cloud health endpoint confirmed the selected profile. All **47 tests** pass
locally and on the cloud VM.

- `outputs/quality-sft-live-001/`: selected-profile replay, trajectory/state logs,
  metrics, metadata and independent cleanup/health check.
- `outputs/quality-cloud-live-001/`: RL experiment, including its timeout result.
- `outputs/quality-cloud-smoke-001/`: real high-resolution RL inference smoke test.
- `outputs/epic-record-001/`: high-resolution autopilot recording.
- `outputs/quality-comparison.html`: Low/Epic visual comparison.
- `outputs/epic-gpu.csv`: local GPU samples.
- `outputs/cloud-deployment.json`: current service profile; the earlier manifest
  is preserved as `cloud-deployment-baseline.json`.

See [quality profile operations](quality-profile.md) for start/stop and driving
commands. `quality` selects BF16 SFT/high inputs; `reasoning` selects BF16
RL/high inputs; `baseline` selects NF4 SFT/small inputs.
