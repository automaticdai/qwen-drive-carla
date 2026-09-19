"""Route-conditioned recording, shadow inference and closed-loop CARLA driving."""
import argparse
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from .controller import Control, TrajectoryTracker
from .route import build_route
from .session import CarlaSession, save_record


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run_episode(session, planner, route, output, *, mode="shadow", steps=600,
                replan_ticks=5, max_speed=8.0, max_deviation=8.0, tracker=None, warmup_driver="brake"):
    """Only this loop advances time. Inference wall time never advances simulation."""
    if mode not in ("record", "shadow", "closed-loop"):
        raise ValueError("Unknown run mode")
    if warmup_driver not in ("brake", "autopilot"):
        raise ValueError("warmup_driver must be brake or autopilot")
    if steps < 16 or not 1 <= replan_ticks <= 10:
        raise ValueError("Need 16+ steps and a replan interval of 1–10 ticks")
    if mode != "record" and planner is None:
        raise ValueError("This mode requires a planner")
    tracker = tracker or TrajectoryTracker(max_speed=max_speed, max_age=replan_ticks * 0.1 + 0.1)
    history = deque(maxlen=16)
    summary = {"status": "running", "mode": mode, "ticks": 0, "plans": 0, "rejected_plans": 0,
               "collision_events": 0, "lane_invasion_events": 0, "red_light_events": 0,
               "signal_ambiguous_events": 0, "progress": 0.0,
               "inference_seconds": [], "max_deviation_m": 0.0}
    summary.update(warmup_driver=warmup_driver if mode == "closed-loop" else "autopilot",
                   handover_frame=None, distance_m=0.0, model_distance_m=0.0)
    started = time.perf_counter()
    last_plan_tick = None
    previous_xy = None
    model_control_active = False
    try:
        with (output / "frames.jsonl").open("w") as frames, (output / "steps.jsonl").open("w") as log:
            for tick in range(steps):
                record = session.tick()
                if previous_xy is not None:
                    distance = float(np.linalg.norm(np.asarray(record["pose"][:2]) - previous_xy))
                    summary["distance_m"] += distance
                    if model_control_active:
                        summary["model_distance_m"] += distance
                previous_xy = np.asarray(record["pose"][:2])
                state = route.update(record["pose"][:2])
                record["command"] = state["command"]
                history.append(record)
                save_record(record, output, frames)
                events = session.drain_events()
                for event in events:
                    summary[event["kind"] + "_events"] += 1
                summary.update(ticks=tick + 1, progress=state["progress"],
                               max_deviation_m=max(summary["max_deviation_m"], state["deviation_m"]))
                speed = float(np.linalg.norm(record["velocity"]))
                control = Control(reason="autopilot" if mode == "record" else "warmup")
                metrics = None
                apply_model_control = mode == "closed-loop" and (len(history) == 16 or warmup_driver == "brake")
                if summary["collision_events"]:
                    summary["status"] = "collision"
                elif summary["red_light_events"]:
                    summary["status"] = "red_light_violation"
                elif state["deviation_m"] > max_deviation:
                    summary["status"] = "off_route"
                elif state["reached"]:
                    summary["status"] = "completed_route"
                else:
                    if planner is not None and len(history) == 16:
                        if mode == "closed-loop" and summary["handover_frame"] is None:
                            session.autopilot(False)
                            summary["handover_frame"] = record["frame"]
                        if last_plan_tick is None or tick - last_plan_tick >= replan_ticks:
                            trajectory, metrics = planner.plan(list(history))
                            validation_error = None
                            try:
                                tracker.set_plan(trajectory, record["pose"], record["timestamp"])
                            except ValueError as exc:
                                validation_error = str(exc)
                                summary["rejected_plans"] += 1
                            raw = np.asarray(trajectory, dtype=float)
                            # JSON null preserves where a model produced NaN/Inf.
                            logged_trajectory = np.where(np.isfinite(raw), raw, None).tolist()
                            write_json(output / f"plan-{record['frame']:08d}.json",
                                       dict(frame=record["frame"], timestamp=record["timestamp"],
                                            origin=record["pose"], command=record["command"],
                                            trajectory=logged_trajectory, metrics=metrics,
                                            validation_error=validation_error))
                            summary["plans"] += 1
                            summary["inference_seconds"].append(metrics["seconds"])
                            last_plan_tick = tick
                            if validation_error is not None and mode == "closed-loop":
                                raise ValueError(validation_error)
                        control = tracker.step(record["pose"], speed, record["timestamp"])
                    if apply_model_control:
                        session.apply(control)
                        model_control_active = len(history) == 16
                log.write(json.dumps(dict(frame=record["frame"], timestamp=record["timestamp"],
                                          speed_mps=speed, route=state, control=asdict(control),
                                          control_applied=apply_model_control and summary["status"] == "running",
                                          driver=("model" if model_control_active else
                                                  "brake" if mode == "closed-loop" and warmup_driver == "brake"
                                                  else "autopilot"),
                                          events=events, inference=metrics), allow_nan=False) + "\n")
                log.flush()
                if summary["status"] != "running":
                    break
            else:
                summary["status"] = "time_limit"
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                       error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        # Also stop autopilot on inference errors, sensor timeouts and interruption.
        try:
            session.autopilot(False)
            session.apply(Control(reason="episode_end"))
        except Exception as exc:
            summary["stop_error"] = str(exc)
        summary["wall_seconds"] = time.perf_counter() - started
        summary["simulation_seconds"] = summary["ticks"] * 0.1
        write_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--destination-index", type=int, default=10)
    parser.add_argument("--list-spawns", action="store_true", help="List spawn coordinates without changing the world")
    parser.add_argument("--mode", choices=("record", "shadow", "closed-loop"), default="shadow")
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--replan-ticks", type=int, default=5, help="10 Hz ticks between plans (1–10)")
    parser.add_argument("--max-speed", type=float, default=8, help="m/s")
    parser.add_argument("--max-deviation", type=float, default=8, help="Stop when more than this many metres from route")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--traffic", type=int, default=0, help="Background vehicles within 120 m of start")
    parser.add_argument("--weather", choices=("ClearNoon", "WetCloudyNoon", "HardRainNoon", "ClearSunset"), default="ClearNoon")
    parser.add_argument("--follow-camera", action="store_true", help="Follow ego in the visible CARLA window")
    parser.add_argument("--camera-profile", choices=("small", "high"), default="small")
    parser.add_argument("--model", type=Path, default=Path("models/Qwen-Drive-1.0-4B"))
    parser.add_argument("--precision", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument("--planner-url", help="Remote Qwen service through a local SSH tunnel, e.g. http://127.0.0.1:8765")
    parser.add_argument("--planner-timeout", type=float, default=120, help="Remote request timeout in wall-clock seconds")
    parser.add_argument("--warmup-driver", choices=("brake", "autopilot"), default="brake",
                        help="Closed-loop only: remain stopped or use autopilot for the initial 16 observations")
    parser.add_argument("--agents", type=Path, default=Path("vendor/carla-agents"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 0 <= args.traffic <= 50:
        parser.error("--traffic must be between 0 and 50")
    if (not all(math.isfinite(v) and v > 0 for v in
                (args.seconds, args.max_speed, args.max_deviation, args.timeout, args.planner_timeout)) or
            args.seconds < 1.6 or not 1 <= args.replan_ticks <= 10):
        parser.error("Use finite positive limits, 1.6+ seconds and 1–10 replan ticks")
    if args.list_spawns:
        import carla
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        print(json.dumps([dict(index=i, x=t.location.x, y=t.location.y, z=t.location.z, yaw=t.rotation.yaw)
                          for i, t in enumerate(client.get_world().get_map().get_spawn_points())], indent=2))
        return
    if args.output is None:
        parser.error("--output is required for a run")
    args.output.mkdir(parents=True, exist_ok=False)
    session = CarlaSession(args.host, args.port, args.tm_port, args.spawn_index, args.seed, args.timeout,
                           camera_profile=args.camera_profile)
    metadata = {"complete": False, "command_source": "global_route", "mode": args.mode,
                "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        with session:
            session.configure_scene(weather=args.weather, traffic=args.traffic, follow_camera=args.follow_camera)
            world_map = session.world.get_map()
            spawns = world_map.get_spawn_points()
            if not 0 <= args.destination_index < len(spawns):
                raise ValueError(f"destination-index must be in [0, {len(spawns) - 1}]")
            # Actor getters use the client's last snapshot. Before the first tick
            # a newly spawned vehicle can report (0, 0, 0); use its known spawn.
            route, locations = build_route(world_map, spawns[args.spawn_index].location,
                                            spawns[args.destination_index].location, args.agents)
            write_json(args.output / "route.json", dict(map=world_map.name, points=route.points.tolist(),
                                                       commands=route.commands))
            planner = None
            if args.mode != "record":
                if args.planner_url:
                    from .remote import RemotePlanner
                    planner = RemotePlanner(args.planner_url, timeout=args.planner_timeout)
                else:
                    from .planner import QwenPlanner
                    planner = QwenPlanner(args.model, precision=args.precision)
                metadata["model_profile"] = planner.loading_info
            if args.mode != "closed-loop" or args.warmup_driver == "autopilot":
                session.autopilot(True)
                session.tm.set_path(session.vehicle, locations)
                session.tm.set_desired_speed(session.vehicle, args.max_speed * 3.6)
            physics = session.vehicle.get_physics_control()
            front, rear = physics.wheels[:2], physics.wheels[2:]
            front_center = np.mean([[w.position.x, w.position.y, w.position.z] for w in front], axis=0)
            rear_center = np.mean([[w.position.x, w.position.y, w.position.z] for w in rear], axis=0)
            wheelbase = float(np.linalg.norm(front_center - rear_center)) / 100
            tracker = TrajectoryTracker(max_speed=args.max_speed, wheelbase=wheelbase,
                                         max_steer_degrees=max(w.max_steer_angle for w in front),
                                         max_age=args.replan_ticks * 0.1 + 0.1)
            # Preserve scene provenance even if native CARLA code aborts before finally.
            write_json(args.output / "metadata.json", {**session.metadata, **metadata})
            summary = run_episode(session, planner, route, args.output, mode=args.mode,
                                  steps=round(args.seconds * 10), replan_ticks=args.replan_ticks,
                                  max_speed=args.max_speed, max_deviation=args.max_deviation, tracker=tracker,
                                  warmup_driver=args.warmup_driver)
            metadata["complete"] = True
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(args.output / "metadata.json", {**session.metadata, **metadata})


if __name__ == "__main__":
    main()
