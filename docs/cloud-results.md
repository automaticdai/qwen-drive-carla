# Google Cloud inference validation — 2026-09-19

Qwen is deployed on the existing Google Cloud VM
`instance-20260830-001452` in `us-central1-a`, project
`idyllic-silo-475917-h4`. The VM runs Ubuntu 24.04 with an NVIDIA L4.
The isolated release directory is
`/home/yfrl/qwen-drive-20260919T050311Z-2343929`; tmux session
`qwen-drive-agent` runs the NF4 inference service on loopback port 8765.
A local WSL SSH forward connects to that port. Existing VM workloads and system
Python environments were not changed, and no VM or public firewall rule was
created. A deployment manifest is saved as `outputs/cloud-deployment.json`.

## Validation

- 45 tests pass both locally and on the cloud Ubuntu VM.
- Real recorded CARLA scene sent through the cloud tunnel produced a finite
  `(50, 3)` trajectory. The cold request took 9.07 seconds total, including
  6.58 seconds of GPU inference, with a 4.57 MB request.
- A real closed-loop Windows CARLA / Town01 episode used the lightweight local
  `.venv`, confirmed to have no PyTorch installation. All model inference was
  performed on the cloud L4.

The driving test used spawn 2 → destination 133, a 3 m/s target speed cap,
10 Hz synchronous simulation, replanning every five ticks, and 1.5 seconds of
initial autopilot history. It reached the destination after 18.9 simulation
seconds / 141.59 wall-clock seconds. Qwen controlled 33.18 m of the 36.12 m
travelled. There were no collisions, two lane-invasion events and no rejected
plans. This validates the distributed integration, not reliable driving.

| Transport metric | Measured value |
| --- | ---: |
| Planning requests | 35 |
| Mean total client planning time | 3.238 s |
| Mean server GPU inference time | 1.374 s |
| Mean request size | 4.800 MB |
| Total request body bytes | 168.011 MB |

Times include lossless PNG/base64 transport; client planning time includes
encoding and network overhead. Simulation pauses during each request. The
18.9-second video replay removes those pauses and must not be interpreted as
real-time performance over the network. The legacy `summary.inference_seconds`
field contains total client planning time for a remote planner; individual
plan metrics distinguish `seconds` from `inference_seconds`.

After the test, a fresh local CARLA snapshot confirmed asynchronous mode,
zero vehicles and zero sensors. The cloud `/health` endpoint still reported
ready through the tunnel. The service and tunnel were left running for reuse;
CARLA remains open but idle. VM billing continues while the VM is running.

Artifacts:

- `outputs/cloud-smoke-001/report.json` and `trajectory.npy`.
- `outputs/cloud-live-001/front-replay.mp4`.
- `outputs/cloud-live-001/summary.json`, `transport-summary.json`,
  `cleanup-check.json`, per-plan metrics, frames and camera PNGs.

See [remote inference operations](remote-inference.md) for the exact tunnel,
service start/stop, local driving and remote scenario-suite commands.
