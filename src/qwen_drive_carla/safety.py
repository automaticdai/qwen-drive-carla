"""Conservative, simulator-ground-truth assistance for the dense debug scenario.

This is a planar vehicle/road guard, not learned perception or a safety guarantee.
Other vehicles use constant-velocity predictions. Braking uses a configurable
assumed deceleration; it must be validated against the simulator/weather.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from .adapter import ego_vectors
from .controller import Control, validate_trajectory


@dataclass(frozen=True)
class Footprint:
    half_length: float
    half_width: float
    offset_x: float = 0.0
    offset_y: float = 0.0
    yaw_degrees: float = 0.0

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all() or min(self.half_length, self.half_width) <= 0:
            raise ValueError('Invalid vehicle footprint')

    def corners(self, pose, margin=0.0):
        angle = math.radians(pose[2])
        rotation = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        center = np.asarray(pose[:2]) + np.array([self.offset_x, self.offset_y]) @ rotation
        angle += math.radians(self.yaw_degrees)
        rotation = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        x, y = self.half_length + margin, self.half_width + margin
        return np.array([[x, y], [-x, y], [-x, -y], [x, -y]]) @ rotation + center


@dataclass(frozen=True)
class Obstacle:
    actor_id: int
    pose: tuple
    velocity: tuple
    footprint: Footprint

    def corners(self, seconds):
        pose = [self.pose[0] + self.velocity[0] * seconds,
                self.pose[1] + self.velocity[1] * seconds, self.pose[2]]
        return self.footprint.corners(pose)


def overlaps(a, b):
    """Separating-axis test for oriented rectangles, including touching."""
    for box in (a, b):
        for edge in (box[1] - box[0], box[2] - box[1]):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def perimeter(corners, spacing=.5):
    return np.concatenate([np.linspace(a, b, max(2, math.ceil(np.linalg.norm(b-a)/spacing)+1))
                           for a, b in zip(corners, np.roll(corners, -1, axis=0))])


@dataclass(frozen=True)
class Decision:
    reason: str = 'clear'
    time_ahead: float | None = None
    actor_id: int | None = None

    @property
    def safe(self):
        return self.reason == 'clear'


class GroundTruthGuard:
    source = 'CARLA ground truth; assisted control'

    def __init__(self, footprint, road_contains, *, wheelbase=2.8, max_steer_degrees=35.,
                 margin=.25, deceleration=3., reaction_seconds=.3):
        if not all(math.isfinite(x) and x > 0 for x in
                   (wheelbase, max_steer_degrees, deceleration, reaction_seconds)) or not 0 <= margin < 2 or max_steer_degrees >= 90:
            raise ValueError('Invalid guard limits')
        self.footprint, self.road_contains = footprint, road_contains
        self.wheelbase = wheelbase
        self.max_steer = math.radians(max_steer_degrees)
        self.margin, self.deceleration, self.reaction_seconds = margin, deceleration, reaction_seconds

    def configuration(self):
        return dict(source=self.source, footprint=asdict(self.footprint), margin_m=self.margin,
                    deceleration_mps2=self.deceleration, reaction_seconds=self.reaction_seconds,
                    plan_horizon_seconds=5., obstacle_prediction='constant world velocity',
                    wheelbase_m=self.wheelbase, max_steer_degrees=math.degrees(self.max_steer))

    def check_path(self, poses, times, obstacles=()):
        # Interpolate at <= 0.1 m / 0.025 s / 2 degrees to catch hazards between waypoints.
        previous, previous_t = np.asarray(poses[0]), float(times[0])
        for pose, timestamp in zip(poses, times):
            pose = np.asarray(pose)
            delta = pose - previous
            delta[2] = (delta[2] + 180) % 360 - 180
            steps = max(1, math.ceil(np.linalg.norm(delta[:2])/.1),
                        math.ceil(abs(delta[2])/2), math.ceil((timestamp-previous_t)/.025))
            for fraction in np.linspace(0, 1, steps+1)[1:]:
                sample = previous + fraction * delta
                t = previous_t + fraction * (timestamp-previous_t)
                box = self.footprint.corners(sample, self.margin)
                if not all(self.road_contains(float(x), float(y)) for x, y in perimeter(box)):
                    return Decision('road_boundary', float(t))
                for obstacle in obstacles:
                    if overlaps(box, obstacle.corners(t)):
                        return Decision('vehicle_collision', float(t), obstacle.actor_id)
            previous, previous_t = pose, timestamp
        return Decision()

    def check_plan(self, trajectory, origin, obstacles=()):
        try:
            points = validate_trajectory(trajectory)
            if len(origin) != 3 or not np.isfinite(origin).all():
                raise ValueError('Invalid plan origin')
        except (ValueError, TypeError):
            return Decision('invalid_trajectory')
        xy = ego_vectors(points[:, :2], origin[2]) + np.asarray(origin[:2])
        poses = np.vstack((origin, np.column_stack((xy, origin[2] - np.degrees(points[:, 2])))))
        # Reject reversals and turning radii beyond the vehicle steering limit.
        delta = np.diff(poses[:, :2], axis=0)
        distance = np.linalg.norm(delta, axis=1)
        heading = np.radians(poses[:-1, 2])
        forward = delta[:, 0]*np.cos(heading) + delta[:, 1]*np.sin(heading)
        sideways = -delta[:, 0]*np.sin(heading) + delta[:, 1]*np.cos(heading)
        yaw_change = np.radians((np.diff(poses[:, 2])+180) % 360 - 180)
        bad = ((forward < -.05) | (np.abs(sideways) > .05 + .25*distance) |
               (np.abs(yaw_change) > distance * math.tan(self.max_steer)/self.wheelbase + .03))
        if bad.any():
            return Decision('infeasible_motion', float((np.flatnonzero(bad)[0]+1)*.1))
        return self.check_path(poses, np.arange(51)*.1, obstacles)

    def stopping_path(self, pose, speed, control, *, immediate=False):
        # Assume acceleration while the current command can still take effect, then full braking.
        speed = max(0., speed)
        reaction = 0. if immediate else self.reaction_seconds
        acceleration = 3. if control.throttle > 0 else 0.
        duration = reaction + (speed + acceleration*reaction)/self.deceleration
        state = np.asarray(pose, dtype=float).copy()
        poses, times = [state.copy()], [0.]
        curvature = math.tan(control.steer*self.max_steer)/self.wheelbase
        t = 0.
        while t < duration - 1e-9:
            dt = min(.025, duration-t)
            if t < reaction - 1e-9:
                dt = min(dt, reaction-t)
                next_speed = speed + acceleration*dt
            else:
                next_speed = max(0., speed-self.deceleration*dt)
            distance = .5*(speed+next_speed)*dt
            yaw = math.radians(state[2])
            mid_yaw = yaw + .5*distance*curvature
            state[:2] += distance*np.array([math.cos(mid_yaw), math.sin(mid_yaw)])
            state[2] += math.degrees(distance*curvature)
            speed = next_speed; t += dt
            poses.append(state.copy()); times.append(t)
        return np.asarray(poses), np.asarray(times)

    def filter_control(self, control, pose, speed, obstacles=(), *, plan_decision=Decision()):
        if not np.isfinite([*pose, speed, control.throttle, control.steer, control.brake]).all() or speed < 0:
            return Control(reason='safety_invalid_state'), Decision('invalid_state')
        decision = plan_decision
        if decision.safe:
            decision = self.check_path(*self.stopping_path(pose, speed, control), obstacles)
        if decision.safe:
            return control, decision
        # Prefer straight braking; retain proposed steering only if it gives a clear stopping path.
        brake = Control(reason='safety_'+decision.reason)
        straight = self.check_path(*self.stopping_path(pose, speed, brake, immediate=True), obstacles)
        if not straight.safe and control.steer:
            turning = Control(brake=1., steer=control.steer, reason=brake.reason)
            alternative = self.check_path(*self.stopping_path(pose, speed, turning, immediate=True), obstacles)
            if alternative.safe or (alternative.time_ahead or 0) > (straight.time_ahead or 0):
                brake = turning
        return brake, decision


class CarlaRoad:
    """Allowed lane union, with no nearest-road projection of off-road samples."""
    def __init__(self, world_map, road_id, lanes, z=0.):
        self.world_map, self.road_id, self.lanes, self.z = world_map, road_id, set(lanes), z

    def __call__(self, x, y):
        import carla
        wp = self.world_map.get_waypoint(carla.Location(x=x, y=y, z=self.z), project_to_road=False,
                                         lane_type=carla.LaneType.Driving)
        return wp is not None and wp.road_id == self.road_id and wp.lane_id in self.lanes

    def progress(self, x, y):
        """Town05 dense scenario travels in increasing road-s across all allowed lanes."""
        import carla
        wp = self.world_map.get_waypoint(carla.Location(x=x, y=y, z=self.z), project_to_road=False,
                                         lane_type=carla.LaneType.Driving)
        if wp is None or wp.road_id != self.road_id or wp.lane_id not in self.lanes:
            return float('nan')
        return float(wp.s)


def actor_footprint(actor):
    box = actor.bounding_box
    return Footprint(box.extent.x, box.extent.y, box.location.x, box.location.y, box.rotation.yaw)


def capture_obstacles(world, ego):
    """Snapshot all other vehicles on the ego's vertical level; simulation is synchronous."""
    obstacles = []
    ego_z = ego.get_location().z
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == ego.id:
            continue
        transform, velocity = actor.get_transform(), actor.get_velocity()
        if abs(transform.location.z-ego_z) > 3.:
            continue
        obstacles.append(Obstacle(actor.id, (transform.location.x, transform.location.y, transform.rotation.yaw),
                                  (velocity.x, velocity.y), actor_footprint(actor)))
    return obstacles
