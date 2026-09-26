import math

import numpy as np

from qwen_drive_carla.candidates import select_candidate
from qwen_drive_carla.lane_following import following_candidates
from qwen_drive_carla.safety import Footprint, GroundTruthGuard, Obstacle


def lane(s):
    return [s, 0., 0.]


def guard():
    return GroundTruthGuard(Footprint(2., .9), lambda x, y: abs(y) < 1.75)


def test_stopped_vehicle_gets_a_guarded_moving_lane_following_proposal():
    candidates = following_candidates([0, 0, 0], 0, lane)
    assert candidates.shape == (3, 50, 3)
    selection = select_candidate(candidates, guard(), [0, 0, 0], 0)
    assert selection.index == 0 and selection.decision.safe
    assert selection.evaluations[0]['initial_control_reason'] == 'tracking'
    assert 6 < candidates[0, -1, 0] < 7


def test_stopped_lead_causes_slower_selection_or_no_valid_fallback():
    candidates = following_candidates([0, 0, 0], 0, lane)
    obstacle = lambda x: Obstacle(3, (x, 0, 0), (0, 0), Footprint(2, .9))
    selection = select_candidate(candidates, guard(), [0, 0, 0], 0, [obstacle(9)])
    assert selection.index == 1  # Fastest proposal would overlap the lead vehicle.
    assert select_candidate(candidates, guard(), [0, 0, 0], 0, [obstacle(8)]).index == 2
    assert select_candidate(candidates, guard(), [0, 0, 0], 0, [obstacle(5)]).index is None


def test_fallback_follows_curve_and_preserves_small_initial_offset():
    curve = lambda s: [100*math.sin(s/100), 100*(1-math.cos(s/100)), math.degrees(s/100)]
    paths = following_candidates([0, .1, 0], .5, curve)
    assert len(paths) == 3
    assert abs(paths[0, 0, 1]) < .001
    assert paths[0, -1, 1] < 0  # CARLA right turn is Qwen negative Y.
    assert guard().check_plan(paths[0], [0, .1, 0]).safe


def test_fallback_refuses_departure_misalignment_branch_and_excess_speed():
    assert len(following_candidates([0, 1, 0], 0, lane)) == 0
    assert len(following_candidates([0, 0, 30], 0, lane)) == 0
    assert len(following_candidates([0, 0, 0], 5, lane)) == 0
    assert len(following_candidates([0, 0, 0], 0, lambda s: None if s > 1 else lane(s))) == 0


def test_carla_fallback_refuses_red_yellow_and_junctions(monkeypatch):
    import sys
    from types import SimpleNamespace as NS
    from qwen_drive_carla.lane_following import carla_following_candidates
    monkeypatch.setitem(sys.modules, 'carla', NS(Location=lambda **k: NS(**k), LaneType=NS(Driving=1)))
    def forbidden(*args, **kwargs):
        raise AssertionError('Signal must block generation before map lookup')
    for signal in ('Red', 'Yellow', 'TrafficLightState.Red'):
        assert len(carla_following_candidates(NS(get_waypoint=forbidden), [0, 0, 0], 0, signal,
                                             road_id=37, lanes=(-3, -2, -1))) == 0
    junction = NS(road_id=37, lane_id=-2, is_junction=True)
    assert len(carla_following_candidates(NS(get_waypoint=lambda *a, **k: junction), [0, 0, 0], 0, None,
                                         road_id=37, lanes=(-3, -2, -1))) == 0
