from dataclasses import asdict

import numpy as np
import pytest

from qwen_drive_carla.controller import Control, TrajectoryTracker
from qwen_drive_carla.safety import (Decision, Footprint, GroundTruthGuard, Obstacle,
                                    actor_footprint, overlaps)


def straight(speed=2.):
    return np.column_stack((np.arange(1, 51)*.1*speed, np.zeros((50, 2))))


def guard(road=lambda x, y: abs(y) <= 5.25):
    return GroundTruthGuard(Footprint(2., .9), road)


def car(x, y=0, velocity=(0., 0.), yaw=0.):
    return Obstacle(42, (x, y, yaw), velocity, Footprint(2., .9))


def test_footprint_offset_and_rotation_match_carla_coordinates():
    box = Footprint(2, 1, 1, 0, 90).corners((10, 20, 90))
    np.testing.assert_allclose(box.mean(axis=0), [10, 21])
    np.testing.assert_allclose(np.ptp(box, axis=0), [4, 2])
    with pytest.raises(ValueError):
        Footprint(float('nan'), 1)


def test_vehicle_box_collision_respects_adjacent_lane_and_rotation():
    g = guard()
    assert g.check_plan(straight(), [0, 0, 0], [car(7, 3.5)]).safe
    result = g.check_plan(straight(), [0, 0, 0], [car(7)])
    assert result.reason == 'vehicle_collision' and result.actor_id == 42
    assert 1.2 < result.time_ahead < 1.5
    a = Footprint(2, 1).corners([0, 0, 45])
    assert overlaps(a, Footprint(2, 1).corners([1, 1, -45]))
    assert not overlaps(a, Footprint(2, 1).corners([7, 0, -45]))


def test_vehicle_corners_trigger_before_center_leaves_road():
    result = guard().check_plan(straight(), [0, 4.2, 0])
    assert result.reason == 'road_boundary' and result.time_ahead == 0


def test_lane_change_inside_allowed_lane_union_is_not_rejected():
    t = np.arange(1, 51)*.1
    # Smooth, physically feasible 3.5 m lane change over 5 seconds.
    y = 3.5*(1-np.cos(np.pi*t/5))/2
    heading = np.arctan((3.5*np.pi/10)*np.sin(np.pi*t/5)/3)
    path = np.column_stack((3*t, y, heading))
    assert guard().check_plan(path, [0, 0, 0]).safe


def test_moving_cross_traffic_and_moving_away_lead():
    g = guard()
    assert g.check_plan(straight(), [0, 0, 0], [car(7, velocity=(2, 0))]).safe
    result = g.check_plan(straight(), [0, 0, 0], [car(7, 6, velocity=(0, -2))])
    assert result.reason == 'vehicle_collision'


def test_swept_path_detects_boundary_between_samples():
    g = guard(lambda x, y: not 5.40 < x < 5.65)
    result = g.check_path([[0., 0., 0.], [10., 0., 0.]], [0., 1.])
    assert result.reason == 'road_boundary'
    assert 0 < result.time_ahead < 1


def test_qwen_left_positive_path_becomes_world_left_at_rotated_origin():
    seen = []
    g = guard(lambda x, y: seen.append((x, y)) or True)
    path = straight()
    path[:, 1] = .01*path[:, 0]**2
    path[:, 2] = np.arctan(.02*path[:, 0])
    assert g.check_plan(path, [10, 20, 90]).safe
    # Northbound + Qwen left is world +X, matching the adapter reflection.
    assert max(x for x, _ in seen) > 11
    assert max(y for _, y in seen) > 30


@pytest.mark.parametrize('path', [np.zeros((49, 3)), np.full((50, 3), np.nan), straight(100)])
def test_invalid_plans_fail_closed(path):
    assert guard().check_plan(path, [0, 0, 0]).reason == 'invalid_trajectory'


def test_reversing_and_impossible_turns_rejected():
    assert guard().check_plan(straight(-2), [0, 0, 0]).reason == 'infeasible_motion'
    path = straight()
    path[3:, 2] = 1.
    assert guard().check_plan(path, [0, 0, 0]).reason == 'infeasible_motion'
    sideways = straight()
    sideways[:, [0, 1]] = sideways[:, [1, 0]]
    assert guard().check_plan(sideways, [0, 0, 0]).reason == 'infeasible_motion'


def test_emergency_brakes_for_actual_speed_even_with_stationary_plan():
    g = guard()
    assert g.check_plan(np.zeros((50, 3)), [0, 0, 0], [car(8)]).safe
    control, result = g.filter_control(Control(throttle=.2, brake=0), [0, 0, 0], 5, [car(8)])
    assert result.reason == 'vehicle_collision'
    assert control.throttle == 0 and control.brake == 1
    # Comfortable gap does not cause constant stopping.
    proposed = Control(throttle=.2, brake=0)
    control, result = g.filter_control(proposed, [0, 0, 0], 2, [car(15)])
    assert result.safe and control == proposed


def test_stopping_distance_includes_reaction_and_acceleration():
    poses, times = guard().stopping_path([0, 0, 0], 4, Control(throttle=.2, brake=0))
    expected = 4*.3 + .5*3*.3**2 + (4+3*.3)**2/(2*3)
    assert poses[-1, 0] == pytest.approx(expected, abs=.001)
    assert times[-1] == pytest.approx(.3+4.9/3)


def test_rejected_replacement_clears_old_plan_and_pid_then_safe_plan_can_resume():
    tracker = TrajectoryTracker()
    tracker.set_plan(straight(), [0, 0, 0], 0)
    tracker.step([0, 0, 0], 0, 0)
    rejected = Decision('road_boundary', 1.)
    tracker.clear_plan()
    assert tracker.integral == 0 and tracker.previous_error is None
    proposed = tracker.step([0, 0, 0], 0, .1)
    control, result = guard().filter_control(proposed, [0, 0, 0], 0, plan_decision=rejected)
    assert result == rejected and control.reason == 'safety_road_boundary' and control.brake == 1
    tracker.set_plan(straight(), [0, 0, 0], .2)
    control, result = guard().filter_control(tracker.step([0, 0, 0], 0, .2), [0, 0, 0], 0)
    assert result.safe and control.throttle > 0


def test_non_finite_control_state_brakes():
    control, result = guard().filter_control(Control(), [0, 0, float('nan')], 2)
    assert result.reason == 'invalid_state' and control.brake == 1


def test_carla_actor_footprint_adapter_preserves_offset_rotation():
    from types import SimpleNamespace as NS
    actor = NS(bounding_box=NS(extent=NS(x=2, y=1), location=NS(x=.3, y=-.2), rotation=NS(yaw=5)))
    assert asdict(actor_footprint(actor)) == dict(half_length=2, half_width=1, offset_x=.3, offset_y=-.2, yaw_degrees=5)


def test_closed_loop_guard_stops_before_stationary_lead_in_bicycle_model():
    g = guard()
    obstacle = car(10)
    x, speed = 0., 4.
    interventions = 0
    for _ in range(60):
        proposed = Control(throttle=.3, brake=0)
        command, decision = g.filter_control(proposed, [x, 0, 0], speed, [obstacle])
        interventions += not decision.safe
        next_speed = max(0., speed + (-3. if command.brake else 3.)*.1)
        x += .5*(speed+next_speed)*.1
        speed = next_speed
        assert not overlaps(g.footprint.corners([x, 0, 0]), obstacle.corners(0))
    assert interventions > 0
    assert x < 10-4 and speed < .5
