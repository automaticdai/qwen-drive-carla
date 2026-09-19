# Qwen BEV perception

This integration runs the released Qwen BEV perception head on six synchronized
CARLA RGB cameras. It produces actual model predictions: 3D boxes, semantic
occupancy and a road-map raster. It is a separate batch perception path; it does
not supply BEV features to the planning expert or change vehicle control.

The camera ring uses 896×512 images, 90° horizontal FOV, and yaw angles
0, 60, 120, 180, −120, −60 degrees at 1.7 m height. Exports carry the actual
CARLA mount matrices and pinhole intrinsics, converted from CARLA's left-handed
coordinates to ego X-forward/Y-left/Z-up. Camera optical axes are right/down/
forward. The virtual lidar frame equals ego; no lidar or ground truth is fabricated.
The `nuscenes` dataset-type field selects the head's six-camera/grid convention;
`source: carla` and `out_of_distribution: true` identify the data honestly.

The released model was trained with specific nuScenes/nuPlan camera rigs. Our
CARLA mounts/FOV/scenery differ, so correct calibration does not establish
perception accuracy. The detection classes also do not include traffic-light
color. Adding this head does not repair the observed red-light violation.

## Capture locally

With an idle Windows CARLA server on Town01:

```bash
.venv/bin/python scripts/capture_bev.py --host 172.30.64.1 \
  --tm-port 8010 --spawn-index 241 --traffic 8 --frames 1 \
  --output outputs/bev-capture-NEW
```

This holds the ego car braked, settles the cameras for ten ticks, then captures
one synchronized frame per requested sample. All owned actors are cleaned up;
weather and world settings are restored. `capture.json` retains cleanup and rig
metadata. Existing three-camera driving recordings cannot supply missing rear
views and are not accepted as surround captures.

## Run on the L4

In the cloud release directory, install a separate runtime once:

```bash
bash scripts/setup_bev_cloud.sh
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$PWD/.venv-bev/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
```

The isolated `.venv-bev` uses torch 2.8.0/cu129, matching the installed compiler,
and leaves `.venv-qwen` unchanged. The official kernels compile on first use.
`download_model.py --include-perception` adds the pinned perception weights.

Transfer the capture directory through the existing SSH connection using
`gcloud compute scp --recurse`; no public inference port is required. Run BEV
sequentially with planning to fit the L4. If the planning service is running,
stop its named tmux session with Ctrl-C and wait for it to exit before inference:

```bash
tmux send-keys -t qwen-drive-agent C-c
# Verify that qwen-drive-agent has exited before the next command.
.venv-bev/bin/python scripts/infer_bev.py \
  --frames outputs/bev-capture-NEW --output outputs/bev-predictions-NEW --limit 1
.venv-bev/bin/python scripts/render_bev.py \
  --frame outputs/bev-capture-NEW/FRAME_TOKEN \
  --prediction outputs/bev-predictions-NEW/FRAME_TOKEN.npz \
  --output outputs/bev-predictions-NEW/bev.png
bash scripts/start_remote_service.sh quality
```

Restore the planning service even if BEV inference fails. Copy the predictions
and PNG back through SCP to inspect locally. Each prediction has a JSON record
of dimensions, runtime, GPU memory, source and `feeds_planner: false`. Occupancy
is shown as a topmost-nonempty-voxel projection; it is not a ground-truth map.

To validate the head independently of CARLA calibration, use
`--frames vendor/qwen-drive/data/demo/perception` with a fresh output directory.

## Why the current driving demo is slow

The wet-junction run took 652.08 wall seconds for 34.1 simulation seconds (19.1×
slower than real time). Its 65 plans averaged 6.63 seconds end to end: 3.65 seconds
reported model inference and 2.98 seconds of other request overhead. The latter
includes image encoding, transport and response handling; network-only latency
was not separately measured. Mean JSON request size was 7.07 MB. The remaining
221 seconds include simulator ticks, high-resolution rendering, image recording,
and other client work, not a separately measured GPU-rendering cost.

Simulation pauses for each cloud request, with a new plan every 0.5 simulation
seconds. The 1440p spectator and the 1280×768 model cameras are independent.
The controller additionally caps speed at 4 m/s (14.4 km/h); that cap is distinct
from the simulation's wall-time slowdown. BF16/high resolution was selected for
fidelity, not demonstrated faster or safer driving. The custom CARLA rig, route
commands, simulator visuals and trajectory tracker have not been validated as
equivalent to the published evaluation setup. These are possible contributors
to poor driving, not proven explanations for each violation.

Useful next experiments are matched small-vs-high camera runs, request-stage
profiling, calibrated controller tracking tests, and repeated red-light scoring.
Unpausing simulation during multi-second inference without stale-plan handling
would not be a valid speed fix. Neither more BEV computation nor a higher speed
cap should be presented as an established driving-quality improvement.

Upstream: [perception documentation](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/docs/perception.md).

## First validation — 2026-09-19

Both official CUDA kernels compiled on the L4. One official eight-camera sample
and one real six-camera Town01 capture completed with finite outputs:

| Input | Timed inference/output stage | Peak reserved GPU memory |
| --- | ---: | ---: |
| Official sample `4d0d1ccbb1035a90` | 5.20 s | 20.18 GiB |
| CARLA capture `00018591` | 3.82 s | 17.35 GiB |

These are single cold inference measurements after model loading, including
prediction serialization but excluding model load and SSH transfer. Each emitted
300 scored box candidates, a 200×200×16 occupancy grid and a 200×400 map raster.
Candidates are not 300 verified objects. The CARLA visualization includes apparent
ego-body detections; no ground-truth perception accuracy was measured.

Local evidence: `outputs/bev-carla-001/bev.png`, predictions and JSON metrics beside
it, and `outputs/bev-official-001/`. Capture cleanup was empty-error and the live
world returned to async with zero vehicles/sensors. The BF16 SFT/direct/high
planning service was restored and its health endpoint verified. All 66 tests passed.
