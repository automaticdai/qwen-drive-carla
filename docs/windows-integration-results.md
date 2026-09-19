# Windows CARLA integration — 2026-09-19

Windows CARLA now runs on this machine, and the WSL client successfully recorded
a route and passed its real camera frames to Qwen-Drive. This validates recording
and offline inference. It does not establish closed-loop driving quality.

## Installation

- Official CARLA 0.9.16 Windows package: [release](https://github.com/carla-simulator/carla/releases/tag/0.9.16).
- Installed at `C:\Users\autom\AppData\Local\CARLA\0.9.16` (19.4 GB uncompressed).
- Archive retained at `downloads/CARLA_0.9.16.zip`; provenance and a local SHA-256
  digest are in `downloads/CARLA_0.9.16.download.json`. Temporary chunks removed.
- WSL connected to the Windows host at `172.30.64.1:2000`; no firewall changes
  were needed. This IP can change when WSL restarts.
- Startup logs are in `C:\Users\autom\AppData\Local\CARLA\logs`.
- CARLA was stopped after recording to release VRAM for Qwen. It is installed,
  but is not left running in the background.

The repo includes a resumable archive downloader, a staging-directory Windows
installer, and an off-screen launch script with optional stdout/stderr logs.
The checksum records the downloaded file; it is not an upstream signature.

## Actual capture

| Item | Result |
| --- | --- |
| Map and route | Town01, spawn 2 to spawn 133, about 38 m |
| Driver | CARLA Traffic Manager, desired speed 5 m/s |
| Outcome | `completed_route` (within the runner's 3 m endpoint tolerance) |
| Simulated duration | 8.6 s |
| Wall duration | 13.46 s |
| Capture | 86 ticks, three RGB images per tick |
| History validation | All 71 available 1.5 s windows passed |
| Sensor events | 0 collisions, 0 lane invasions |
| Maximum distance to sampled route point | 0.98 m; this includes waypoint spacing, not just lateral error |
| Cleanup | 0 remaining vehicles/sensors; asynchronous world settings restored |

Data and logs: `outputs/windows-carla/record-002/`.
Server cleanup check: `outputs/windows-carla/cleanup-check.json`.

The first attempt uncovered a startup bug: a newly spawned actor's cached
`get_location()` returned `(0, 0, 0)` before the first simulation tick. The route
was consequently built from the wrong origin and the episode stopped as
off-route. Route creation now uses the known spawn transform; the repeat capture
completed successfully. The failed attempt remains in `record-001` for diagnosis.

## Qwen on CARLA images

The server was stopped before loading Qwen. Inference used recording index 30,
the SFT planner, BF16, SDPA, one trajectory sample, and the small image profile.

| Item | Result |
| --- | --- |
| Model load | 7.06 s |
| First inference | 2.52 s |
| Second inference | 0.76 s |
| Peak reserved GPU memory | 11.90 GiB |
| Output | Finite `(1, 50, 3)` trajectory |
| Predicted forward travel over 5 s | 49.95 m |
| Recorded autopilot forward travel over 5 s | 22.40 m |
| Mean displacement from recorded reference | 13.32 m |
| Final displacement from recorded reference | 27.57 m |

The predicted path is broadly straight, but the model predicts substantially
faster travel than the speed-limited autopilot reference. These are observations
from one scene, not aggregate benchmark results. They do not determine whether
the difference comes from rendering/domain transfer, preprocessing, or model
speed preference. The model is not given the Traffic Manager's 5 m/s speed cap;
the experimental controller separately caps tracked speed.

Prediction, plot, timings and comparison are in
`outputs/windows-carla/qwen-record-002/`.

Observed total GPU usage with CARLA and other Windows applications was about
9,971 MiB on Town10HD_Opt and 5,674 MiB after switching to Town01. Concurrent
Qwen/CARLA execution has not been tested; these measurements indicate insufficient
comfortable headroom for the full BF16 model plus rendering on this 16 GB GPU.
Sequential recording/inference works. Memory reduction or a separate inference
GPU remains the next step before a concurrent shadow or closed-loop trial.

## Reproduce on this installation

In Windows PowerShell:

```powershell
& "$env:LOCALAPPDATA\CARLA\0.9.16\CarlaUE4.exe" -RenderOffScreen -quality-level=Low -nosound
```

From the repo in WSL, after the server starts (replace the IP if needed):

```bash
.venv/bin/python -c 'import carla; c=carla.Client("172.30.64.1",2000); c.set_timeout(60); c.load_world("Town01")'
.venv/bin/drive-run --host 172.30.64.1 --mode record \
  --spawn-index 2 --destination-index 133 --seconds 15 --max-speed 5 \
  --output outputs/my-town01-recording
```

Stop the CARLA process you launched, then:

```bash
.venv-qwen/bin/drive-benchmark --recording outputs/my-town01-recording \
  --index 30 --output outputs/my-town01-prediction --runs 2
```

Each output directory must be new. Earlier environment/model setup instructions
are in the README. Automated tests still pass (26 tests).
