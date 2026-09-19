import numpy as np
import pytest

from qwen_drive_carla.controller import TrajectoryTracker, validate_trajectory


def straight(speed=5):
    return np.column_stack((np.arange(1, 51) * .1 * speed, np.zeros((50, 2))))


def test_acceleration_braking_and_left_steering():
    tracker = TrajectoryTracker()
    points = straight()
    points[:, 1] = .02 * points[:, 0] ** 2
    tracker.set_plan(points, [0, 0, 0], 0)
    control = tracker.step([0, 0, 0], 0, 0)
    assert control.steer < 0
    assert control.throttle > 0 and control.brake == 0
    control = tracker.step([0, 0, 0], 12, .1)
    assert control.throttle == 0 and control.brake > 0


def test_plan_stays_in_world_frame_as_ego_moves_and_rotates():
    tracker = TrajectoryTracker()
    tracker.set_plan(straight(), [10, 20, 90], 10)
    np.testing.assert_allclose(tracker.world_xy[0], [10, 20.5])
    control = tracker.step([10, 21, 90], 5, 10.2)
    assert abs(control.steer) < 1e-10
    # Ego now faces east; original northbound path lies to its right.
    assert tracker.step([10, 21, 0], 5, 10.2).reason == "stationary_or_behind"


def test_stationary_and_stale_plans_brake():
    tracker = TrajectoryTracker(max_age=.6)
    assert tracker.step([0, 0, 0], 0, 0).brake == 1
    tracker.set_plan(np.zeros((50, 3)), [0, 0, 0], 0)
    assert tracker.step([0, 0, 0], 0, 0).brake == 1
    tracker.set_plan(straight(), [0, 0, 0], 0)
    assert tracker.step([0, 0, 0], 1, .7).reason == "stale_plan"
    assert tracker.step([0, 0, 0], 1, -.1).reason == "stale_plan"


@pytest.mark.parametrize("points", [np.zeros((49, 3)), np.full((50, 3), np.nan), straight(100)])
def test_invalid_plan_rejected_and_previous_plan_invalidated(points):
    tracker = TrajectoryTracker()
    tracker.set_plan(straight(), [0, 0, 0], 0)
    with pytest.raises(ValueError):
        tracker.set_plan(points, [0, 0, 0], 0)
    assert tracker.step([0, 0, 0], 0, 0).reason == "no_plan"


def test_speed_cap_and_right_turn():
    tracker = TrajectoryTracker(max_speed=3)
    points = straight(10)
    points[:, 1] = -.01 * points[:, 0] ** 2
    tracker.set_plan(points, [0, 0, 0], 0)
    control = tracker.step([0, 0, 0], 0, 0)
    assert control.target_speed == 3
    assert control.steer > 0
    assert np.isfinite(validate_trajectory(points)).all()


def test_pursuit_reduces_lateral_error_in_bicycle_simulation():
    tracker = TrajectoryTracker(max_age=4.9)
    tracker.set_plan(straight(), [0, 0, 0], 0)
    x, y, yaw = 0.0, 2.0, 0.0
    for step in range(40):
        control = tracker.step([x, y, np.degrees(yaw)], 5, step * .1)
        # CARLA right-positive yaw, constant speed for this lateral-only check.
        yaw += 5 / tracker.wheelbase * np.tan(control.steer * tracker.max_steer) * .1
        x += 5 * np.cos(yaw) * .1
        y += 5 * np.sin(yaw) * .1
    assert abs(y) < .2
