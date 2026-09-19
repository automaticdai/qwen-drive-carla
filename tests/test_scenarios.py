import json
from pathlib import Path
import runpy
from types import SimpleNamespace

from qwen_drive_carla.session import CarlaSession

summarize = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/run_scenarios.py'))['summarize']


def test_completion_with_lane_crossing_is_not_clean(tmp_path):
    (tmp_path / 'summary.json').write_text(json.dumps(dict(status='completed_route', collision_events=0,
                                                         lane_invasion_events=1, rejected_plans=0)))
    (tmp_path / 'frames.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        dict(command='left', nearest_traffic_m=30), dict(command='straight', nearest_traffic_m=9)]))
    result = summarize(tmp_path)
    assert not result['clean_completion']
    assert result['nearest_traffic_center_m'] == 9
    assert result['frames_with_traffic_within_15m'] == 1
    assert result['commands_observed'] == ['left', 'straight']


def test_setup_failure_is_retained_without_frames(tmp_path):
    (tmp_path / 'metadata.json').write_text(json.dumps(dict(error='CUDA out of memory')))
    result = summarize(tmp_path)
    assert result['status'] == 'setup_error' and not result['clean_completion']
    assert result['error'] == 'CUDA out of memory'
    assert result['nearest_traffic_center_m'] is None


def test_unscored_signals_cannot_be_clean_completion(tmp_path):
    (tmp_path / 'summary.json').write_text(json.dumps(dict(status='completed_route', collision_events=0,
                                                         lane_invasion_events=0, rejected_plans=0)))
    (tmp_path / 'frames.jsonl').write_text(json.dumps(dict(command='straight')))
    assert not summarize(tmp_path)['clean_completion']


def test_weather_restored_even_when_actor_cleanup_fails():
    calls = []
    class Actor:
        def destroy(self):
            raise RuntimeError('actor disappeared')
    session = CarlaSession()
    session.world = SimpleNamespace(set_weather=lambda value: calls.append(('weather', value)),
                                    apply_settings=lambda value: calls.append(('settings', value)))
    session.original_weather = 'original sky'
    session.original = 'async settings'
    session.actors = [Actor()]
    import pytest
    with pytest.warns(UserWarning):
        session.close()
    assert calls == [('weather', 'original sky'), ('settings', 'async settings')]


def test_lane_event_keeps_marking_types_for_diagnosis():
    session = CarlaSession()
    session.record_event('lane_invasion', SimpleNamespace(frame=2, timestamp=.2,
                         crossed_lane_markings=[SimpleNamespace(type='Solid'), SimpleNamespace(type='Broken')]))
    assert session.drain_events()[0]['markings'] == ['Solid', 'Broken']


def test_background_autopilot_is_unregistered_before_destroy():
    calls = []
    actor = SimpleNamespace(set_autopilot=lambda enabled, port: calls.append(('autopilot', enabled, port)),
                            destroy=lambda: calls.append('destroy'))
    session = CarlaSession()
    session.background_actors = [actor]
    session.actors = [actor]
    session.close()
    assert calls == [('autopilot', False, 8000), 'destroy']
    assert not session.background_actors
