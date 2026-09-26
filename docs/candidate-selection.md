# Guarded selection among Qwen trajectories

The dense debug runner can draw several trajectories from a single Qwen scene
encoding and choose a passing proposal. It defaults to three candidates in local
mode; `--candidate-count 1` provides the one-sample baseline. Counts are bounded
at six. The ordinary `drive-run` interface retains its single-trajectory behavior.

Each candidate must pass the existing CARLA ground-truth trajectory guard and
an initial stopping-envelope check. Rejection is a hard gate: a candidate with
more progress cannot compensate for a failed safety check. Among passing
candidates, selection rewards forward distance along the scenario road, capped
at the configured speed limit times the five-second horizon, and penalizes
acceleration magnitude, heading variation and lateral displacement. These are
heuristic scores, not calibrated probabilities. Stable ties select the earlier
candidate. Candidate coordinates are never averaged or edited.

The selected trajectory goes to the existing tracker; the guard checks the
actual proposed control again on every tick. If all candidates fail, the old
plan is cleared and braking continues until a later proposal passes. This
candidate-selection mechanism does not add a lane-change decision process. An
optional map-based fallback is documented below. If all samples share the same failure, extra samples cannot fix it.

Assisted selection uses CARLA map progress and ground-truth obstacle states.
The dashboard shows candidate count, passing count and selected candidate
(displayed starting at 1). Prediction files retain all candidates, their gate
results and score components, and the selected index (stored starting at 0).
`trajectory` is the selected path (including an explicitly enabled fallback); otherwise it retains
the first rejected proposal for diagnosis. `selected_index: null` means none
was accepted. Episode summaries count rejected batches, individual passing
candidates and selections of alternatives to candidate zero.

With `--safety-mode off`, the runner always uses candidate zero and performs no
CARLA-based ranking. Use `--candidate-count 1` as well to avoid spending compute
on unused proposals in an unassisted baseline.

## Local and remote usage

```bash
.venv-qwen/bin/python scripts/run_dense_debug.py --local-planner \
  --precision nf4 --image-profile small --candidate-count 3 \
  --safety-mode carla --seconds 30 --start-paused \
  --output outputs/multi-candidate-NEW
```

The cloud debug service also accepts `--candidate-count`, defaulting to three:

```bash
.venv-bev/bin/python -m qwen_drive_carla.debug_inference \
  --image-profile high --candidate-count 3 --port 8765
```

Remote counts are configured on the server and advertised by its health
response; the local runner flag does not change them. A new client can consume
an older single-candidate service, reporting the actual count as one. The new
server retains the original first `trajectory` field and adds `candidates` to
`/debug` responses; `/plan` is unchanged. No cloud deployment is performed just
by changing the local code.

## Controlled local comparison

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv-qwen/bin/python \
  scripts/evaluate_dense_candidates.py --counts 1 3 --repeats 2 --seconds 12 \
  --output outputs/candidate-comparison-NEW
```

This loads the local model once, runs fresh guarded dense scenarios, alternates
count order on successive repeats, and writes `comparison.json` plus complete
episode artifacts. Both arms use the same scenario settings and existing seed
42; this is repeated execution, not a multi-seed robustness study. The script
stops after an error or user stop and closes its dashboard on completion. It
reports model-controlled distance, moving ticks, candidate acceptance, hazard
reasons, infractions and inference request time. It does not treat stopping
without a collision as successful driving.

## Braking handover correction

The first live comparison exposed an independent control-delivery bug: the
runner logged full brake, but CARLA retained the last Traffic Manager throttle.
An isolated test observed throttle 0.85 and brake 0 after repeated requests for
brake 1. Changing the requested brake slightly forced transmission, confirming
that identical manual commands were being skipped by CARLA's actor-side cache.
The [CARLA 0.9.16 client source](https://github.com/carla-simulator/carla/blob/0.9.16/LibCarla/source/carla/client/Vehicle.cpp#L51)
contains the corresponding command deduplication, while autopilot registration
uses a separate path.

`CarlaSession.apply` now sends `ApplyVehicleControl` through an acknowledged
batch with `do_tick=False`, bypassing that actor cache and waiting for the server
before the next physics tick. It raises on a rejected command. This correction
also applies to the ordinary runner because both use `CarlaSession`. Snapshot
controls are logged separately from the newly requested control: they describe
the command observed at the current frame, which normally matches the preceding
tick's request.

The live regression `scripts/check_brake_handover.py` sends the exact initial
spawn brake after autopilot handover. With acknowledged commands it observed
full brake and zero throttle on all 35 checked ticks, and stopped from
6.666 m/s. Evidence is in `outputs/brake-handover-20260926-acknowledged/`.
The earlier comparison with the delivery bug is retained under
`outputs/candidate-comparison-20260926-001/`; it must not be used as a valid
one-versus-three planning comparison.

## Optional lane-following fallback

Add `--fallback lane-follow` to a local or remote dense runner to allow a
**CARLA lane following + guard** controller when Qwen has no passing candidate
or its selected candidate's immediate target speed is below 0.15 m/s. This is
opt-in and requires `--safety-mode carla`. Its progress is conventional map-based
assistance, not improved neural planning.

The fallback proposes current-lane paths at 1.5, 1.0 and 0.5 m/s, with a gradual
speed ramp. Every path passes the same full-horizon road/vehicle checks and
stopping-envelope check before selection. Faster paths that would overlap a
lead vehicle are rejected, allowing a slower proposal or braking. The fallback
refuses red/yellow signal states, junctions, ambiguous lane continuations,
speeds above 4 m/s, offsets above 0.3 m from the lane center, or heading error
above 10°. It stays on the current lane of the configured three-lane road;
it does not attempt an overtake or recovery from a departure.

The dashboard labels the active path source and driver. Prediction artifacts
retain the original Qwen candidates and selection plus separate
`fallback_candidates`, `fallback_selection` and `fallback_trigger` fields.
`trajectory_source` identifies the path actually sent to the tracker. Summaries
count `fallback_plans` and `qwen_rejected_batches` separately. A valid model plan
can resume at the next decision; the fallback does not change the model weights
or disguise the model's rejected proposals.

```bash
.venv-qwen/bin/python scripts/run_dense_debug.py --local-planner \
  --candidate-count 3 --safety-mode carla --fallback lane-follow \
  --seconds 30 --start-paused --output outputs/assisted-following-NEW
```

The comparison script accepts the same `--fallback lane-follow` option. The
fallback inherits the guard's limitations, including constant-velocity vehicle
prediction and no pedestrian/static-obstacle perception. It is restricted to
this dedicated dense-vehicle scenario.

## Measured local results — 2026-09-26

After the handover correction, six live episodes used local RTX 5080 inference,
NF4/small planning inputs, Epic rendering and the same dense Town05 scenario.
All six ended normally at the configured time limit with **zero collisions,
zero lane-marking events and zero observed/requested control mismatches**. This
is a limited scenario test, not route completion or general driving validation.

| Configuration | Repeats | Distance by 12 sim seconds | Distance by 20 sim seconds |
| --- | ---: | ---: | ---: |
| One Qwen candidate + guard | 2 | 0.020–0.021 m | Not run |
| Three Qwen candidates + guard | 2 | 0.047–0.098 m | Not run |
| Three Qwen candidates + guarded map fallback | 2 | **7.59–7.79 m** | **12.30–12.43 m** |

Distance starts at the 1.6-second handover observation. The first two conditions
ran for 12 seconds; the fallback condition ran for 20 seconds, so the 12-second
column provides a shared observation window. Repeats used the existing seed
42; this is not a multi-seed study. One/three-sample comparisons reversed order
on their second repeat. The fallback runs were performed afterward.

Three-sample selection chose alternatives to candidate zero on four and eight
of 21 decisions, but still stalled. More samples alone did not yield useful
progress. The optional fallback supplied 31 and 33 of 37 trajectory decisions,
respectively. It stayed in lane and finished about 7.5 and 7.4 m behind the lead
vehicle by road-coordinate centre distance. **No overtake was achieved.** The
progress increase is predominantly conventional map-based assistance.

Peak torch-reserved memory was 6.80 GiB without fallback and 6.82 GiB with it;
these figures exclude CARLA and other GPU processes. Shared cloud inference was
not redeployed or live-tested in this work. Full local regression validation
passed **127 tests**, with dashboard JavaScript syntax checked separately.

Artifacts:

- `outputs/candidate-comparison-20260926-002/`: corrected one/three-sample runs.
- `outputs/candidate-fallback-20260926-001/`: two optional-fallback runs.
- `outputs/driving-improvement-validation-20260926/results.json`: combined
  metrics, matching 12-second windows and control-delivery checks.
- `outputs/driving-improvement-validation-20260926/progress.png`: progress and
  lead-gap curves.
- `outputs/driving-improvement-validation-20260926/cleanup.json`: independent
  check confirming async Town05, zero vehicles and zero sensors after testing.

CARLA remains open and idle. Evaluation dashboards close when the comparison
finishes; use the interactive runner command above for another driving session.
