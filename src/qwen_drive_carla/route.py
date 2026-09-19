"""Monotonic progress and near-term navigation intent on a CARLA route."""
import sys
from pathlib import Path

import numpy as np


class RouteProgress:
    def __init__(self, points, commands, lookahead=20.0):
        self.points = np.asarray(points, dtype=float)
        self.commands = list(commands)
        if (self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 2 or
                not np.isfinite(self.points).all() or len(self.commands) != len(self.points)):
            raise ValueError("Route needs at least two finite XY points and matching commands")
        if any(c not in ("left", "right", "straight", "follow") for c in self.commands):
            raise ValueError("Only lane following and junction turns are supported")
        self.distances = np.concatenate(([0], np.cumsum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))))
        if self.distances[-1] < 5:
            raise ValueError("Choose a route at least 5 metres long")
        self.index = 0
        self.lookahead = lookahead

    def update(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("Route position must be finite XY")
        # Bound the search by distance to avoid jumping to a later crossing of the route.
        end = max(self.index + 1, int(np.searchsorted(self.distances, self.distances[self.index] + 10)))
        candidates = self.points[self.index:end]
        self.index += int(np.argmin(np.linalg.norm(candidates - position, axis=1)))
        deviation = float(np.linalg.norm(self.points[self.index] - position))
        end = int(np.searchsorted(self.distances, self.distances[self.index] + self.lookahead, side="right"))
        command = "straight"
        for item in self.commands[self.index:end]:
            if item != "follow":
                command = item
                break
        remaining = float(self.distances[-1] - self.distances[self.index])
        reached = remaining <= 3 and np.linalg.norm(self.points[-1] - position) <= 3
        return dict(command=command, index=self.index, deviation_m=deviation,
                    progress=float(self.distances[self.index] / self.distances[-1]),
                    remaining_m=remaining, reached=bool(reached))


def build_route(world_map, origin, destination, agents_path):
    path = Path(agents_path).resolve()
    if not (path / "agents/navigation/global_route_planner.py").is_file():
        raise RuntimeError("CARLA route helpers missing; run scripts/download_carla_agents.py")
    sys.path.insert(0, str(path))
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    trace = GlobalRoutePlanner(world_map, sampling_resolution=2.0).trace_route(origin, destination)
    names = {"LEFT": "left", "RIGHT": "right", "STRAIGHT": "straight", "LANEFOLLOW": "follow"}
    if any(option.name not in names for _, option in trace):
        raise ValueError("This route requires lane changes; choose endpoints on a route without lane changes")
    locations = [waypoint.transform.location for waypoint, _ in trace]
    progress = RouteProgress([[loc.x, loc.y] for loc in locations],
                             [names[option.name] for _, option in trace])
    return progress, locations
