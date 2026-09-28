# BF16 versus NF4 archived-input replay — 2026-09-27

BF16 removes the three initial reversal defects, but does not resolve unsafe
road alignment on the tested inputs. Both precisions fail all 29 full-horizon
road-boundary checks. This is an offline comparison, not a BF16 driving rollout.

Replayed every archived input from episodes 002–004 in
`outputs/local-qwen-evaluation-20260926-001`, using the same local model weights,
SFT expert, direct planning, SDPA, small image profile, one sample, and noise seed
42. Fresh NF4 replay reproduced the saved XY trajectories exactly on all inputs.
BF16 was run first, then NF4 in a separate process on the RTX 5080.

| Measurement | NF4 | BF16 |
| --- | ---: | ---: |
| Inputs completed | 29 | 29 |
| Numerical trajectory-validation failures | 0 | 0 |
| Motion-feasibility failures | 3 | 0 |
| Independent road-boundary failures | 29 | 29 |
| Mean warm inference, excluding first call | 1.005 s | 4.529 s |
| Median warm inference | 0.979 s | 4.521 s |
| First inference | 17.506 s | 26.364 s |
| Peak PyTorch reserved GPU memory | 6.48 GiB | 11.37 GiB |

BF16 changed XY predictions by **0.314 m on average** across all points and
inputs, and endpoints by **0.841 m on average**. Those distances measure changes,
not accuracy. At each initial handover, NF4 predicts a reversal near the
2.9-second horizon; BF16 removes it while retaining a path that cuts across the
curving road. The three first BF16 proposals reach the expanded-footprint road
boundary at approximately 3.4 seconds.

The comparison supports quantization as a contributor to the observed reversal
defects. It also demonstrates that changing precision alone does not eliminate
road-boundary failures on these archived inputs. It does not determine whether
BF16 would avoid the actual crashes after its own closed-loop history develops.

## Limits and runtime conditions

- The 29 states come from three repetitions of one scenario driven by NF4.
  Later states already reflect NF4's earlier decisions; they are not an unbiased
  BF16 rollout. Initial handovers provide the cleanest comparison.
- Road/feasibility checks use the offline Town05 map, a centered 4.8 × 2.0 m
  footprint, 0.25 m margin, and default guard steering geometry. They inspect the
  full five-second horizon. No other-vehicle collision replay is available.
- Image hashes were verified while loading each input. Resolution remains small;
  these results do not test the high-resolution profile or camera calibration.
- The GPU already reported 4,952 MiB in use and 34% utilization before model
  loading. During BF16, a sample reported 14,794 MiB total usage and 99%
  utilization. Other workloads were not stopped or controlled. These timings
  describe this session, not an isolated precision-throughput benchmark; the
  reason for the BF16 slowdown was not profiled. Memory pressure and contention
  remain possible explanations. Both runs used expandable CUDA allocator
  segments, and neither failed with an out-of-memory error.
- GPU execution required running outside the filesystem/network sandbox. The
  earlier sandbox NVML failure was not a host GPU failure.

## Artifacts and reproduction

- BF16 predictions, individual checks, and comparison plot:
  `outputs/bf16-replay-20260927-001/`.
- Fresh NF4 predictions and checks: `outputs/nf4-replay-20260927-001/`.
- Aggregate measurements:
  `outputs/bf16-replay-20260927-001/comparison.json`.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
.venv-qwen/bin/python scripts/replay_precision.py \
  --precision bf16 \
  --episodes outputs/local-qwen-evaluation-20260926-001/episode-002 \
             outputs/local-qwen-evaluation-20260926-001/episode-003 \
             outputs/local-qwen-evaluation-20260926-001/episode-004 \
  --map /mnt/c/Users/autom/AppData/Local/CARLA/0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr \
  --output outputs/bf16-replay-NEW
```

Run again with `--precision nf4` and a fresh output directory for the baseline.
The script refuses an existing output directory, persists results after every
input, and records failures in its report. No runtime defaults or weights were
changed. Script compilation and CLI checks passed, along with 35 existing
adapter, controller, input-archive, and safety tests.

## Follow-up live BF16 video

`outputs/bf16-live-video-20260927-001/episode-001` ran the real Town05 dense
scenario with Epic rendering, BF16 SFT/direct/small, one sample, safety off,
and no fallback. Metadata confirms zero quantized linear modules.

The episode terminated at **5.4 simulation seconds** with
`left_driving_carriageway`, two lane-invasion events, zero collisions, eight
plans, and no safety interventions or fallback plans. The final ego position
independently maps to no driving waypoint. This is one failed live run, not an
aggregate comparison. Warm inference averaged 7.14 seconds with CARLA running;
episode wall time was 90.94 seconds, excluding model loading.

The recorded `drive.mp4` contains all 54 simulation frames at 10 fps, H.264,
1280×736, with front view, side insets, and telemetry. Inference pauses are
omitted and the initial autopilot warmup is labeled. Full video decoding passed.
The runner's cleanup reported no errors; an independent RPC check confirmed
asynchronous mode and zero vehicles/sensors afterward. Recording-related
regression tests passed: 13 tests with local socket access enabled.
