import numpy as np
import pytest

from qwen_drive_carla.candidates import candidate_array, candidate_count, select_candidate
from qwen_drive_carla.safety import Footprint, GroundTruthGuard, Obstacle


def straight(speed=2):
    return np.column_stack((np.arange(1, 51)*.1*speed, np.zeros((50, 2))))


def guard():
    return GroundTruthGuard(Footprint(2, .9), lambda x, y: abs(y) < 5.25)


def test_valid_alternative_replaces_unsafe_first_without_averaging():
    candidates = np.stack([straight(-2), straight(2), straight(1)])
    result = select_candidate(candidates, guard(), [0, 0, 0], 1)
    assert result.index == 1 and result.decision.safe
    assert result.log()['valid_count'] == 2
    assert result.evaluations[0]['score'] is None
    assert result.evaluations[0]['decision']['reason'] == 'infeasible_motion'


def test_no_safe_candidate_retains_braking_decision():
    result = select_candidate([straight(-2), straight(-1)], guard(), [0, 0, 0], 1)
    assert result.index is None and not result.decision.safe
    assert result.log()['valid_count'] == 0


def test_single_candidate_and_stable_ties():
    assert select_candidate([straight()], guard(), [0, 0, 0], 1).index == 0
    assert select_candidate([straight(), straight()], guard(), [0, 0, 0], 1).index == 0


def test_collision_risk_cannot_be_outweighed_by_progress():
    lead = Obstacle(42, (9, 0, 0), (0, 0), Footprint(2, .9))
    result = select_candidate([straight(4), straight(.5)], guard(), [0, 0, 0], 0, [lead])
    assert result.index == 1
    assert result.evaluations[0]['decision']['reason'] == 'vehicle_collision'


def test_stationary_proposal_still_checks_actual_stopping_distance():
    lead = Obstacle(42, (8, 0, 0), (0, 0), Footprint(2, .9))
    result = select_candidate([straight(0)], guard(), [0, 0, 0], 5, [lead])
    assert result.index is None and result.decision.reason == 'vehicle_collision'


def test_nonfinite_candidate_does_not_hide_valid_local_alternative():
    result = select_candidate([np.full((50, 3), np.nan), straight()], guard(), [0, 0, 0], 0)
    assert result.index == 1 and result.log()['valid_count'] == 1
    with pytest.raises(ValueError, match='Non-finite'):
        candidate_array([np.full((50, 3), np.nan), straight()], finite=True)


def test_progress_uses_road_coordinate_and_refuses_invalid_map_values():
    calls = []
    def progress(x, y):
        calls.append((x, y))
        return y  # Rotated route: world +Y is forward.
    g = GroundTruthGuard(Footprint(2, .9), lambda x, y: True)
    result = select_candidate([straight(1), straight(2)], g, [0, 0, 90], 0, progress=progress)
    assert result.index == 1 and result.evaluations[1]['progress_m'] == pytest.approx(10)
    assert len(calls) == 4
    result = select_candidate([straight()], g, [0, 0, 0], 0, progress=lambda x, y: np.nan)
    assert result.index is None and result.decision.reason == 'invalid_progress'


def test_smooth_path_beats_oscillatory_speed_with_same_progress():
    a = straight()
    b = a.copy(); b[::2, 0] += .06
    result = select_candidate([b, a], guard(), [0, 0, 0], 1)
    assert result.log()['valid_count'] == 2 and result.index == 1


@pytest.mark.parametrize('value', [0, 7, -1, 2.5, True, '3'])
def test_candidate_count_is_bounded(value):
    with pytest.raises(ValueError):
        candidate_count(value)


@pytest.mark.parametrize('value', [[], np.zeros((7, 50, 3)), np.zeros((50, 3)), np.zeros((3, 49, 3))])
def test_candidate_array_shape_and_count(value):
    with pytest.raises(ValueError):
        candidate_array(value)
