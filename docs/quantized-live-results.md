# Quantized live results — 2026-09-19

Qwen and Windows CARLA now run concurrently on the RTX 5080 16 GB. A short
straight route reached the destination after a moving handover, but lane-marking
events and standstill/rejected-prediction behavior remain unresolved.

## Low-memory model profile

`--precision nf4` quantizes 248 language-model Linear modules using bitsandbytes
0.50.2, 4-bit NF4, double quantization and BF16 compute. The vision module,
embeddings/output head and planning expert remain BF16. Loader checks enforce
the expected quantization scope and the protected modules' dtype. The original
BF16 profile remains the default and available for comparisons.

On the same recorded CARLA scene used by the earlier BF16 benchmark:

| Measurement | BF16 | NF4 |
| --- | --- | --- |
| Peak reserved model VRAM | 11.90 GiB | 6.80 GiB |
| Warm inference | 0.76 s | 0.79 s |
| Model load | 7.06 s | 4.96 s |

NF4 changed the predicted XY trajectory by 0.80 m on average and 1.93 m at its
endpoint relative to BF16. This single comparison is not an accuracy evaluation.
Details: `outputs/nf4-recorded-smoke/`.

Implementation follows the installed Transformers bitsandbytes integration,
including its skip-module list for 4-bit conversion; see the
[official quantization documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

## Concurrent shadow run

`outputs/nf4-shadow-002/` completed the Town01 spawn-2 to spawn-133 route while
Traffic Manager controlled the vehicle and Qwen proposed trajectories:

- 86 ticks / 8.6 simulated seconds; 25.80 wall seconds.
- 14 predictions, with 0.734 s mean warm inference.
- 2 predictions rejected by the unchanged 45 m/s/discontinuity check.
- 0 collision events; 0 lane-invasion events.
- Observed total GPU usage peaked at 13,206 MiB (12.90 GiB), including CARLA and
  other Windows GPU usage. Samples: `outputs/nf4-shadow-002-gpu.csv`.

The first shadow attempt stopped on a rejected prediction. Shadow mode now saves
and counts invalid proposals while autopilot continues. Closed-loop mode still
stops on a rejected prediction. One replayed rejection contained a waypoint step
equivalent to 51.2 m/s, despite an otherwise ordinary-looking endpoint; rejection
limits were not relaxed to make the route pass. Raw plans and errors are retained.

## Closed-loop experiments

All trials used the same empty, straight Town01 route and a **3 m/s target-speed
cap**, which is not a guarantee that physical vehicle speed never overshoots.

| Trial | Outcome | Model-controlled distance | Lane-marking events |
| --- | --- | --- | --- |
| Brake warmup, 15 s | Time limit; essentially stationary | Approximately 0 m | 0 |
| Moving handover, 15 s | Time limit; 79% route progress | 27.79 m | 1 |
| Moving handover, up to 25 s | Destination tolerance reached at 16.4 s | 32.00 m | 2 |

The standstill model repeatedly predicted waiting before future movement. With
frequent replanning, the vehicle never launched. This was not fixed by forcing
throttle. Instead, a separate explicit `--warmup-driver autopilot` experiment
provides the first 16 observations (1.5 s of history), then disables autopilot
before the first model command. The handover and distance driven afterward are
logged separately; this does not demonstrate starting from rest.

The final moving-handover run (`outputs/nf4-handover-002/`) had:

- 164 ticks / 16.4 simulated seconds; 50.13 wall seconds.
- 34.94 m total travel, including 2.94 m before model control.
- 30 predictions, none rejected; no collision events.
- Two raw lane-invasion sensor events, so this is **not a clean lane-keeping run**.
- Maximum distance to the nearest sampled route point of 1.36 m. This includes
  sampling distance and is not a pure lateral-error measurement.
- `completed_route` means within the runner's 3 m endpoint tolerance, at 94.7%
  sampled-route progress; it does not imply benchmark success.

The lanes crossed were not classified in these logs. Controller tuning, camera
configuration, quantization effects and simulator domain transfer need further
evaluation; the present evidence does not isolate the cause of the crossings.
No traffic, junction, pedestrian, weather or multi-route evaluation is claimed.

## Reproduce

Refresh dependencies with `bash scripts/setup.sh`. Start the installed Windows
server and load Town01 as described in `windows-integration-results.md`, then:

```bash
.venv-qwen/bin/drive-run --host 172.30.64.1 --mode shadow --precision nf4 \
  --spawn-index 2 --destination-index 133 --seconds 15 --max-speed 5 \
  --output outputs/my-nf4-shadow

.venv-qwen/bin/drive-run --host 172.30.64.1 --mode closed-loop --precision nf4 \
  --warmup-driver autopilot --spawn-index 2 --destination-index 133 \
  --seconds 25 --max-speed 3 --output outputs/my-nf4-handover

.venv-qwen/bin/python scripts/plot_episode.py outputs/my-nf4-handover
```

Use the current Windows-host IP if it differs, and fresh output directories.
CARLA is stopped after this session's experiments to release GPU memory.
All 29 automated tests pass, including rejected-plan handling and handover
distance/control accounting.
