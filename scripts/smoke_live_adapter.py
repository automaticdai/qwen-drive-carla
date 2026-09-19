"""Exercise the live history -> Qwen -> controller path using the upstream demo.

This uses recorded dataset observations, not a running CARLA server.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/live-adapter-smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "source": "upstream demo; not CARLA"}
    try:
        from qwen_drive.benchmarks import read_scene_file
        from qwen_drive.images import ImageArchive
        from qwen_drive_carla.adapter import VIEWS
        from qwen_drive_carla.controller import TrajectoryTracker
        from qwen_drive_carla.planner import QwenPlanner
        demo = Path("vendor/qwen-drive/data/demo")
        scene = next(read_scene_file(demo / "planning_scenes.jsonl",
                                    image_archive=ImageArchive.open(demo / "frames.parquet"))).scene
        command = {0: "straight", 1: "left", 2: "right"}[scene.nav_command]
        images = {name: [frame.load() for frame in scene.views[tag]] for name, tag in VIEWS.items()}
        records = []
        for i in range(16):
            x, y, heading = scene.history[i]
            vx, vy = scene.history_velocity[i]
            ax, ay = scene.history_acceleration[i]
            records.append(dict(frame=i, timestamp=i / 10, pose=[x, -y, -np.degrees(heading)],
                                velocity=[vx, -vy], acceleration=[ax, -ay], command=command,
                                images={name: frames[i // 5] for name, frames in images.items()}))
        planner = QwenPlanner("models/Qwen-Drive-1.0-4B")
        trajectory, metrics = planner.plan(records)
        tracker = TrajectoryTracker()
        tracker.set_plan(trajectory, records[-1]["pose"], records[-1]["timestamp"])
        control = tracker.step(records[-1]["pose"], float(np.linalg.norm(records[-1]["velocity"])),
                               records[-1]["timestamp"])
        np.save(args.output / "trajectory.npy", trajectory)
        report.update(status="ok", shape=list(trajectory.shape), metrics=metrics,
                      proposed_control=vars(control), command=command)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
