"""Recording contract and planar CARLA-world to Qwen-ego conversion.

World inputs: metres, CARLA yaw in degrees, world-frame velocity/acceleration.
Outputs: current ego frame, X forward, Y left, heading in radians, left positive.
"""
import json
import math
from pathlib import Path

import numpy as np

VIEWS = {"front": "<FRONT VIEW>", "front_left": "<FRONT LEFT VIEW>",
         "front_right": "<FRONT RIGHT VIEW>"}
COMMANDS = {"straight": (0, [0, 1, 0, 0]), "left": (1, [1, 0, 0, 0]),
            "right": (2, [0, 0, 1, 0])}


def ego_vectors(vectors, current_yaw):
    yaw = math.radians(current_yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray(vectors, dtype=np.float64) @ np.array([[c, s], [s, -c]])


def ego_poses(poses, current):
    poses = np.asarray(poses, dtype=np.float64)
    xy = ego_vectors(poses[:, :2] - np.asarray(current[:2]), current[2])
    heading = np.deg2rad(current[2] - poses[:, 2])
    heading = (heading + np.pi) % (2 * np.pi) - np.pi
    return np.column_stack([xy, heading])


def load_records(directory):
    return [json.loads(line) for line in (Path(directory) / "frames.jsonl").read_text().splitlines() if line]


def scene_payload(records, index):
    if index < 15 or index >= len(records):
        raise ValueError("A query needs 16 recorded states (1.5 seconds of history)")
    history = records[index - 15:index + 1]
    timestamps = np.array([r["timestamp"] for r in history])
    if not np.allclose(np.diff(timestamps), 0.1, atol=1e-4, rtol=0):
        raise ValueError("Ego history must have uninterrupted 10 Hz simulation timestamps")
    frames = [r["frame"] for r in history]
    if any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise ValueError("Recording contains missing or repeated simulation frames")
    current = history[-1]
    poses = np.asarray([r["pose"] for r in history], dtype=float)
    velocities = np.asarray([r["velocity"] for r in history], dtype=float)
    accelerations = np.asarray([r["acceleration"] for r in history], dtype=float)
    if not all(np.isfinite(a).all() for a in (poses, velocities, accelerations, timestamps)):
        raise ValueError("Non-finite ego state")
    nav, command = COMMANDS[current["command"]]
    views = {tag: [history[i]["images"][name] for i in (0, 5, 10, 15)]
             for name, tag in VIEWS.items()}
    velocity = ego_vectors(velocities, current["pose"][2])
    acceleration = ego_vectors(accelerations, current["pose"][2])
    return dict(views=views, history=ego_poses(poses, current["pose"]),
                history_velocity=velocity, history_acceleration=acceleration,
                ego_velocity=velocity[-1], ego_acceleration=acceleration[-1],
                nav_command=nav, driving_command=command, token=str(current["frame"]))


def driving_scene(directory, index=15):
    from qwen_drive import CameraFrame, DrivingScene
    payload = scene_payload(load_records(directory), index)
    payload["views"] = {tag: [CameraFrame(Path(directory) / path) for path in paths]
                        for tag, paths in payload["views"].items()}
    return DrivingScene(**payload)


def input_snapshot(payload, timestamp):
    """JSON-safe numeric inputs from the exact payload sent to Qwen."""
    return dict(source_frame=payload['token'], timestamp=float(timestamp),
                ego_velocity=np.asarray(payload['ego_velocity']).tolist(),
                ego_acceleration=np.asarray(payload['ego_acceleration']).tolist(),
                history_velocity=np.asarray(payload['history_velocity']).tolist(),
                nav_command=payload['nav_command'], driving_command=payload['driving_command'],
                axes='X forward, Y left', source='Qwen request inputs')
