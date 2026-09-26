"""Explicit map-based fallback for the non-junction dense debug scenario only."""
import math

import numpy as np

from .adapter import ego_poses


def following_candidates(pose, speed, lane_pose_at_distance, *, targets=(1.5, 1., .5)):
    """Generate slow lane-following proposals; callers must guard every proposal.

    The callback returns a CARLA-world lane-center (x, y, yaw-degrees) or None
    when the lane ends/branches/leaves the allowed non-junction road.
    """
    if len(pose) != 3 or not np.isfinite([*pose, speed]).all() or not 0 <= speed <= 4:
        return np.empty((0, 50, 3))
    start = lane_pose_at_distance(0.)
    if start is None:
        return np.empty((0, 50, 3))
    offset = np.asarray(pose[:2])-np.asarray(start[:2])
    yaw_error = (pose[2]-start[2]+180) % 360-180
    # Do not attempt recovery from a bad departure or substantial lane crossing.
    if np.linalg.norm(offset) > .3 or abs(yaw_error) > 10:
        return np.empty((0, 50, 3))
    candidates = []
    for target in targets:
        if not math.isfinite(target) or not 0 < target <= 1.5:
            raise ValueError('Fallback target must be between 0 and 1.5 m/s')
        velocity, distance, poses = speed, 0., []
        for _ in range(50):
            next_velocity = velocity + float(np.clip(target-velocity, -.3, .15))
            distance += .05*(velocity+next_velocity)
            velocity = next_velocity
            lane_pose = lane_pose_at_distance(distance)
            if lane_pose is None:
                break
            # Ease out small initial offsets over 5 m, avoiding a jump to lane center.
            fraction = min(distance/5., 1.)
            blend = 1.-fraction*fraction*(3.-2.*fraction)
            xy = np.asarray(lane_pose[:2])+offset*blend
            poses.append([*xy, lane_pose[2]+yaw_error*blend])
        if len(poses) == 50:
            candidates.append(ego_poses(poses, pose))
    return np.asarray(candidates) if candidates else np.empty((0, 50, 3))


def carla_following_candidates(world_map, pose, speed, signal, *, road_id, lanes):
    import carla
    if signal is not None and str(signal).split('.')[-1] in ('Red', 'Yellow'):
        return np.empty((0, 50, 3))
    waypoint = world_map.get_waypoint(carla.Location(x=pose[0], y=pose[1]), project_to_road=False,
                                       lane_type=carla.LaneType.Driving)
    if waypoint is None or waypoint.road_id != road_id or waypoint.lane_id not in lanes or waypoint.is_junction:
        return np.empty((0, 50, 3))

    def lane_pose(distance):
        choices = [waypoint] if distance == 0 else waypoint.next(float(distance))
        if len(choices) != 1:
            return None
        point = choices[0]
        if point.road_id != road_id or point.lane_id != waypoint.lane_id or point.is_junction:
            return None
        transform = point.transform
        return [transform.location.x, transform.location.y, transform.rotation.yaw]

    return following_candidates(pose, speed, lane_pose)
