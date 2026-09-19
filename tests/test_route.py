import numpy as np
import pytest

from qwen_drive_carla.route import RouteProgress


def test_upcoming_turn_and_completion():
    points = [[x, 0] for x in range(0, 42, 2)]
    route = RouteProgress(points, ["follow"] * 5 + ["left"] * 5 + ["follow"] * 11)
    assert route.update([0, 0])["command"] == "left"
    for x in range(0, 41, 2):
        result = route.update([x, 0])
    assert result["command"] == "straight"
    assert result["reached"] and result["progress"] == 1
    # Route progress cannot move backward after localization jitter.
    assert route.update([38, 0])["index"] == 20


def test_route_crossing_does_not_jump_to_destination():
    points = [[x, 0] for x in range(20)] + [[x, 1] for x in reversed(range(20))]
    route = RouteProgress(points, ["follow"] * len(points))
    state = route.update([0, 1])
    assert state["index"] == 0 and not state["reached"]


def test_reject_unsupported_or_empty_routes():
    with pytest.raises(ValueError):
        RouteProgress([[0, 0], [10, 0]], ["follow", "lane_change"])
    with pytest.raises(ValueError):
        RouteProgress([], [])
    with pytest.raises(ValueError):
        RouteProgress([[0, 0], [np.nan, 0]], ["follow", "follow"])
