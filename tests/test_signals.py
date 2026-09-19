import pytest

from qwen_drive_carla.signals import SignalMonitor


def monitor():
    return SignalMonitor([dict(light_id=12, pose=[0, 0, 0], z=0, width=3.5)])


@pytest.mark.parametrize('state,expected', [('Red', 'red_light'), ('Green', None), ('Yellow', None)])
def test_forward_crossing_uses_actual_light_state(state, expected):
    m = monitor()
    assert m.update([-1, 0, 0], {'12': state}) == []
    events = m.update([1, 0, 0], {'12': state})
    assert [e['kind'] for e in events] == ([expected] if expected else [])
    assert m.update([2, 0, 0], {'12': state}) == []


@pytest.mark.parametrize('start,end', [([1, 0, 0], [-1, 0, 0]),
                                      ([-1, 4, 0], [1, 4, 0]),
                                      ([-1, 0, 5], [1, 0, 5]),
                                      ([-1, 0, 0], [-.1, 0, 0])])
def test_reverse_adjacent_lane_overpass_and_stopping_do_not_count(start, end):
    m = monitor()
    m.update(start, {'12': 'Red'})
    assert m.update(end, {'12': 'Red'}) == []


def test_signal_transition_during_crossing_is_ambiguous():
    m = monitor()
    m.update([-1, 0, 0], {'12': 'Red'})
    assert m.update([1, 0, 0], {'12': 'Green'})[0]['kind'] == 'signal_ambiguous'


def test_rotated_stop_line():
    m = SignalMonitor([dict(light_id=12, pose=[10, 10, 90], z=0, width=3.5)])
    m.update([10, 9, 0], {'12': 'Red'})
    assert m.update([10, 11, 0], {'12': 'Red'})[0]['kind'] == 'red_light'
