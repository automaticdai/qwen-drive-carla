"""Exercise the actual dense runner without CARLA/GPU, including control application."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest


@pytest.mark.parametrize('mode', ['carla', 'off'])
@pytest.mark.parametrize('alternative', [False, True])
@pytest.mark.parametrize('fallback', [False, True])
def test_dense_runner_rejects_brakes_logs_and_recovers(tmp_path, monkeypatch, mode, alternative, fallback):
    spec = importlib.util.spec_from_file_location('dense_runner', Path(__file__).parents[1]/'scripts/run_dense_debug.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    location = NS(x=0., y=0., z=0.)
    waypoint = NS(road_id=37, lane_id=-2, s=200, transform=NS(location=location))
    world_map = NS(name='Town05', get_waypoint=lambda *a, **k: waypoint)
    actors = NS(filter=lambda pattern: [])
    world = NS(get_settings=lambda: NS(synchronous_mode=False), get_map=lambda: world_map, get_actors=lambda: actors)
    fake_carla = NS(Client=lambda *a: NS(set_timeout=lambda x: None, get_world=lambda: world),
                    Location=lambda **kwargs: NS(**kwargs), LaneType=NS(Driving=1))
    monkeypatch.setitem(sys.modules, 'carla', fake_carla)
    wheel = lambda x: NS(position=NS(x=x, y=0, z=0), max_steer_angle=35)
    vehicle = NS(id=1, get_location=lambda: location,
                 bounding_box=NS(extent=NS(x=2., y=.9), location=location, rotation=NS(yaw=0)),
                 get_physics_control=lambda: NS(wheels=[wheel(140), wheel(140), wheel(-140), wheel(-140)]))
    applied = []

    class Session:
        def __init__(self, *args, **kwargs):
            self.world = world; self.vehicle = vehicle; self.frame = 0
            self.metadata = {'cameras': {}, 'cleanup_errors': []}
            self.tm = NS(set_desired_speed=lambda *a: None, set_path=lambda *a: None)
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def configure_scene(self, **kwargs): pass
        def autopilot(self, enabled): pass
        def drain_events(self): return []
        def apply(self, control): applied.append(control)
        def tick(self):
            self.frame += 1
            return dict(frame=self.frame, pose=[0., 0., 0.], timestamp=self.frame*.1,
                        velocity=[0., 0.], images={}, traffic_light_state=None)

    class Planner:
        loading_info = {'execution': 'local'}
        calls = 0
        def debug(self, history, cameras):
            self.calls += 1
            speed = -2 if self.calls == 1 else 2
            path = np.column_stack((np.arange(1, 51)*.1*speed, np.zeros((50, 2))))
            response = dict(trajectory=path.tolist(), bev=None, metrics={}, qwen_inputs={})
            if alternative:
                other = np.column_stack((np.arange(1, 51)*.2, np.zeros((50, 2))))
                response['candidates'] = [path.tolist(), other.tolist()]
            return path, response

    state = {}
    dashboard = NS(stopped=False, allow_tick=lambda: True, condition=threading.Condition(),
                   publish=lambda **kwargs: state.update(kwargs))
    monkeypatch.setattr(runner, 'CarlaSession', Session)
    monkeypatch.setattr(runner, 'spawn_transform', lambda m: None)
    monkeypatch.setattr(runner, 'populate', lambda s: NS(id=2, get_location=lambda: location))
    monkeypatch.setattr(runner, 'scenario_waypoint', lambda *a: waypoint)
    monkeypatch.setattr(runner, 'SURROUND', [])
    from qwen_drive_carla.lane_following import following_candidates
    monkeypatch.setattr(runner, 'carla_following_candidates',
                        lambda world_map, pose, speed, signal, **kwargs:
                        following_candidates(pose, speed, lambda distance: [distance, 0., 0.]))
    args = NS(host='unused', local_planner=True, local_planner_instance=Planner(), tm_port=8010,
              image_profile='small', image_transport='jpeg', seconds=2.2, safety_mode=mode,
              fallback='lane-follow' if fallback else 'none')
    output = tmp_path/'episode'
    runner.episode(args, dashboard, output)
    summary = json.loads((output/'summary.json').read_text())
    rows = [json.loads(line) for line in (output/'steps.jsonl').read_text().splitlines()]
    assert summary['status'] == 'time_limit' and summary['plans'] == 2
    assert summary['rejected_plans'] == (1 if mode == 'carla' and not alternative and not fallback else 0)
    if mode == 'carla' and alternative:
        assert summary['alternative_selections'] == 1
        assert rows[15]['control']['throttle'] > 0
        response = json.loads((output/'prediction-00000016.json').read_text())
        assert response['selection']['selected_index'] == 1
        assert response['trajectory'] == response['candidates'][1]
        assert response['selection']['candidates'][0]['decision']['reason'] == 'infeasible_motion'
    elif mode == 'carla' and fallback:
        assert summary['fallback_plans'] == 1
        assert summary['qwen_rejected_batches'] == 1
        assert rows[15]['control']['throttle'] > 0
        assert rows[15]['driver'] == 'CARLA lane following + guard'
        response = json.loads((output/'prediction-00000016.json').read_text())
        assert response['trajectory_source'] == 'CARLA lane following'
        assert response['selection']['selected_index'] is None
    elif mode == 'carla':
        assert summary['safety_intervention_ticks'] == 5
        assert all(c.brake == 1 and c.throttle == 0 for c in applied[:5])
        assert rows[15]['control']['reason'] == 'safety_infeasible_motion'
        assert rows[15]['driver'] == 'Qwen + CARLA guard'
        assert 'ground truth' in rows[15]['safety']['source']
        assert json.loads((output/'prediction-00000016.json').read_text())['safety']['plan_decision']['reason'] == 'infeasible_motion'
    else:
        assert summary['fallback_plans'] == 0
        assert rows[15]['control']['throttle'] == 0
    assert rows[20]['control']['throttle'] > 0
    assert applied[-1].reason == 'episode_end' and applied[-1].brake == 1
    assert state['safety']['decision']['reason'] == 'clear'
