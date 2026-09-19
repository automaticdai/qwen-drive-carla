# Dense three-lane overtaking with live BEV

The debug runner uses Town05 road 37, lanes −3/−2/−1, starting in lane −2 at
OpenDRIVE s=200 m. It verifies the same-direction lane count and spawns exactly
24 background cars. An incomplete spawn fails setup rather than silently
substituting a sparse scene. Cars are staggered across all three lanes at roughly
11–13 m centre spacing. The lead vehicle starts 14 m ahead at 0.8 m/s; other
centre-lane traffic targets 1.2 m/s, adjacent traffic 2 m/s. Traffic Manager keeps
their lanes and uses a 3 m following distance. Actual speeds depend on traffic.

The ego gets 1.5 s of autopilot history, then the BF16 SFT Qwen planner takes
control with a 4 m/s cap. The navigation command remains straight. No forced
lane change or overtaking controller is used. The released planning interface
does not take a dedicated "overtake" command; this scene tests whether its
predicted trajectories choose and complete a pass. It is not guaranteed to do so.

Passing means the ego has changed lane under model control and remained at
least 8 m ahead of the tracked lead car for ten consecutive 10 Hz samples on
the same carriageway, without a collision. This is a limited pass metric, not
proof of legal overtaking or a return to the original lane. The run stops on a
collision, red-light crossing, leaving the carriageway, a completed pass, user
stop, or the configured time limit. Broken-line sensor events are displayed,
not automatically treated as failures, because lane changes are allowed.

## Start

On the existing L4 release, stop the current named planner session and wait for
it to exit. Then launch the shared model profile:

```bash
tmux send-keys -t qwen-drive-agent C-c
# Wait for qwen-drive-agent to exit.
bash scripts/start_remote_service.sh debug
```

This uses the separate `.venv-bev` CUDA 12.9 environment established by
`setup_bev_cloud.sh`. A single BF16 VLM is shared between the planning expert
and perception head. One request lock serializes GPU work. `/plan` remains
available; `/debug` adds six calibrated current camera views and returns the
trajectory plus predicted boxes, road-map PNG and occupancy-projection PNG.
Both correspond to the same source frame, validated by request ID/frame token.
No public inference port is opened; the existing SSH tunnel is reused.

From local WSL, with a dedicated idle Windows CARLA server:

```bash
.venv/bin/python scripts/run_dense_debug.py --host 172.30.64.1 \
  --planner-url http://127.0.0.1:8765 --tm-port 8010 \
  --seconds 60 --start-paused --image-transport jpeg --output outputs/dense-debug-NEW
```

Open **http://localhost:8877** on Windows. Run, Pause, Step, Stop and Reset are
available. A reset rebuilds the scene and pauses after one preview tick so
the cameras show the actual new traffic. Step advances 0.1 simulation seconds;
a step at a planning boundary also waits for inference. Stop during inference
takes effect before another physics tick. Stop cleans up actors and restores
async simulation; Reset does that then creates a new paused episode. Close the
runner with Ctrl-C when finished. The dashboard remains available after each
episode and retains the final views for inspection.

## Live display and limits

The dashboard shows six camera feeds, predicted object boxes and the proposed
trajectory in BEV, semantic road map, a topmost-nonempty-voxel occupancy
projection, vehicle control, infractions, lead gap and overtaking evidence.
The prediction confidence threshold is adjustable. Source frame, simulation
age and wall age are displayed. Cameras continue updating at simulation ticks;
predictions update every five ticks. Simulation freezes while the cloud computes.
Thus this is **live debugging at inference cadence**, not real-time 10/30 FPS
perception. Moving the simulation during a multi-second inference request would
need a different controller and stale-plan strategy.

The first shared request used about 19.3 GiB of reserved L4 memory and took
13.1 seconds including transfer. Subsequent observed requests took about
10.4 seconds. These timings are not guaranteed. BEV and planning use the same
VLM but independent heads: perception outputs do not feed the planner. The
CARLA surround rig remains out of distribution. Telemetry lane IDs, signal
states and lead gaps are explicitly CARLA truth, not Qwen detections.

For side-by-side display on a 2560×1440 desktop, the visible CARLA window was
temporarily resized to the left half and the dashboard placed on the right.
The launcher still defaults to 2560×1440. Planning cameras remain 1280×768 and
the six BEV cameras 896×512. Nine RGB sensors are active during this debug mode.

Each episode saves scenario/model metadata, per-tick controls/events/positions,
and per-plan BEV and trajectory responses. Current code also saves the six
dashboard camera previews at each prediction frame; these are JPEG previews,
not the original PNG model inputs. Controls do not queue behind arbitrary GPU
requests; a second inference request receives busy status.

## First observed result

`outputs/dense-debug-001/episode-001` terminated after **7.9 simulation seconds**
with a collision against its tracked lead car (actor 473). Qwen did not pass:
13 plans, four lane-marking events, one collision, and zero recorded red-light
crossings. Wall time was 208.6 seconds, including time initially paused for
setup checks. Cleanup completed without errors. This is a failed overtaking
attempt, retained for diagnosis—not a successful overtaking demonstration.

The first episode predates camera-preview persistence; its BEV responses and
per-tick evidence are retained. The live UI was browser-tested with actual
camera/BEV data and no JavaScript errors. The combined regression suite passed
70 tests before the first run.

## Vehicle instrument panel and faster image transport

The instrument cluster reads the numeric inputs from the exact Qwen request:
ego speed, forward/left velocity, forward/left acceleration, navigation command
and 16-sample speed history. Source frame and simulation age identify which
decision the displayed inputs belong to. The speedometer is in km/h; velocity
is m/s and acceleration m/s². Inputs are sampled at planning cadence rather
than presented as fresh between requests.

Target speed, throttle, brake and normalized steering are separately labeled
as trajectory-tracker commands, not direct neural-network outputs. Steering
wheel rotation is illustrative; the interface does not invent RPM, fuel or a
physical steering-wheel angle that Qwen does not receive. During autopilot
warmup the tracker controls remain unavailable.

At the user's request, debug runs now default to **JPEG quality 95, 4:4:4**.
This is lossy and can change predictions. `--image-transport lossless` selects
lossless WebP; `--image-transport legacy` reproduces the original PNG transport.
The new transport encodes up to four images in parallel, reuses recent encoded
images locally, uploads only image hashes already absent from the server cache,
and gzip-compresses JSON at level 1. Camera calibration and numeric inputs are
not quantized. Model resolution and inference settings remain the same.

Caches are bounded. Hashes verify uploaded image integrity. If another request
evicts a referenced image, the server rejects the request before inference and
the client resends the images once; model failures and timeouts are never retried.
Request ID and source frame still guard against stale responses. A server without
cache support receives the original PNG format for compatibility.

Metrics expose encoding, serialization, HTTP time, server ingestion/decode time,
planning, BEV, wire size and reused-image counts. Server `server_decode_seconds`
includes reading the HTTP body, decompression and image decoding; it is not a
network-free decoder benchmark. HTTP time includes inference, so it must not be
reported as pure network latency. Raw instrument input snapshots are saved with
each prediction response.

`scripts/benchmark_debug_transport.py` compares PNG/JPEG on three matched scenes,
alternating request order, with no model controls applied. The captures include
overlapping histories to exercise the actual cache reuse pattern. This is a
transport/inference timing comparison, not evidence of unchanged driving quality.

Measured on 2026-09-19 in `outputs/transport-benchmark-001`: three matched
scene pairs, alternating request order. Excluding the first pair to avoid cold
model startup effects, the two warm pairs averaged:

| Measurement | Original PNG | Cached JPEG 95 / 4:4:4 |
| --- | ---: | ---: |
| Wire request size | 11.94 MB | 1.94 MB |
| Image encoding | 1.72 s | 0.024 s |
| Request time outside planning + BEV | 4.68 s | 1.86 s |
| Total request time | 10.67 s | 7.85 s |

That is 84% fewer transmitted bytes, 60% less non-inference request time and
26% lower total request latency in this small benchmark. JPEG requests reused
six image references each once the rolling history was warm. The remaining
request time is dominated by GPU planning and BEV. Full regression validation
passed 77 tests after these changes.

### Windowed 1080p layout

The Windows launcher defaults to a 1920×1080 client view when `-Visible` is used.
With CARLA and the dashboard running, run `scripts/tile_dense_debug_windows.ps1`
in Windows PowerShell to place CARLA on the left and the dashboard on the right.
On a 2560×1440 desktop, window borders leave approximately 624 pixels for the
responsive dashboard. Camera inputs sent to Qwen retain their configured sizes.
