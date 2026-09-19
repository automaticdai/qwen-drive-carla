"""Planar trajectory tracking in simulation (metres, seconds and radians)."""
from dataclasses import dataclass
import math

import numpy as np

from .adapter import ego_vectors


@dataclass(frozen=True)
class Control:
    throttle: float = 0.0
    brake: float = 1.0
    steer: float = 0.0
    target_speed: float = 0.0
    reason: str = "no_plan"


def validate_trajectory(trajectory):
    points = np.asarray(trajectory, dtype=float)
    if points.shape != (50, 3) or not np.isfinite(points).all():
        raise ValueError("Expected a finite (50, 3) trajectory")
    # Reject discontinuities rather than sending extreme samples to the controller.
    steps = np.diff(np.vstack(([0.0, 0.0], points[:, :2])), axis=0)
    if np.any(np.linalg.norm(steps, axis=1) > 4.5):
        raise ValueError("Trajectory exceeds 45 m/s or contains a discontinuity")
    return points


class TrajectoryTracker:
    def __init__(self, max_speed=8.0, wheelbase=2.8, max_steer_degrees=35.0, max_age=0.6):
        if not all(math.isfinite(x) and x > 0 for x in
                   (max_speed, wheelbase, max_steer_degrees, max_age)):
            raise ValueError("Controller limits must be finite and positive")
        self.max_speed = max_speed
        self.wheelbase = wheelbase
        self.max_steer = math.radians(max_steer_degrees)
        self.max_age = min(max_age, 4.9)
        self.world_xy = None
        self.plan_time = None
        self.speeds = None
        self.integral = 0.0
        self.previous_error = None

    def set_plan(self, trajectory, origin, timestamp):
        # Invalidate the previous plan even if this replacement fails validation.
        self.world_xy = None
        points = validate_trajectory(trajectory)
        if len(origin) != 3 or not np.isfinite(origin).all() or not math.isfinite(timestamp):
            raise ValueError("Invalid plan origin or timestamp")
        # The reflection/rotation matrix is its own inverse.
        self.world_xy = ego_vectors(points[:, :2], origin[2]) + np.asarray(origin[:2])
        steps = np.diff(np.vstack(([0.0, 0.0], points[:, :2])), axis=0)
        self.speeds = np.linalg.norm(steps, axis=1) / 0.1
        self.plan_time = timestamp

    def stop(self, reason):
        self.integral = 0.0
        self.previous_error = None
        return Control(reason=reason)

    def step(self, pose, speed, timestamp, dt=0.1):
        if (len(pose) != 3 or not np.isfinite(pose).all() or
                not all(math.isfinite(x) for x in (speed, timestamp, dt)) or dt <= 0):
            return self.stop("invalid_state")
        if self.world_xy is None:
            return self.stop("no_plan")
        age = timestamp - self.plan_time
        if age < -1e-5 or age > self.max_age + 1e-5:
            return self.stop("stale_plan")
        index = min(int(max(0.0, age) / 0.1 + 1e-6), 49)
        local = ego_vectors(self.world_xy - np.asarray(pose[:2]), pose[2])
        candidates = np.flatnonzero((np.arange(50) >= index) & (local[:, 0] > 0.1))
        target_speed = min(float(np.mean(self.speeds[index:min(index + 3, 50)])), self.max_speed)
        if target_speed < 0.15 or len(candidates) == 0:
            return self.stop("stationary_or_behind")
        # Pure pursuit uses a spatial lookahead; plan age chooses the speed interval.
        lookahead = max(2.0, max(speed, 0.0) * 0.6)
        distances = np.linalg.norm(local[candidates], axis=1)
        ahead = candidates[distances >= lookahead]
        target = local[ahead[0] if len(ahead) else candidates[-1]]
        curvature = 2 * target[1] / max(float(target @ target), 0.01)
        # Qwen left-positive -> CARLA steering right-positive.
        steer = float(np.clip(-math.atan(self.wheelbase * curvature) / self.max_steer, -1, 1))
        error = target_speed - max(speed, 0.0)
        self.integral = float(np.clip(self.integral + error * dt, -2, 2))
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error
        effort = 0.4 * error + 0.05 * self.integral + 0.02 * derivative
        return Control(throttle=float(np.clip(effort, 0, 0.6)),
                       brake=float(np.clip(-effort, 0, 1)), steer=steer,
                       target_speed=target_speed, reason="tracking")
