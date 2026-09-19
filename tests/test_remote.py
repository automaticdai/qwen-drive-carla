from contextlib import contextmanager
import threading

import numpy as np
from PIL import Image
import pytest

from qwen_drive_carla.adapter import scene_payload
from qwen_drive_carla.remote import RemotePlanner, decode_scene, encode_scene, make_server


def records():
    image = Image.new('RGB', (640, 384), (42, 130, 201))
    return [dict(frame=i+1, timestamp=i*.1, pose=[i, 2, 90], velocity=[0, 3], acceleration=[0, .1],
                 command='left', images={k: image for k in ('front', 'front_left', 'front_right')}) for i in range(16)]


class Planner:
    loading_info = {'precision': 'test'}
    def plan_payload(self, payload):
        self.payload = payload
        return np.column_stack((np.arange(1, 51)*.2, np.zeros((50, 2)))), {'seconds': .02}


@contextmanager
def service(planner):
    server = make_server(planner, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield RemotePlanner(f'http://127.0.0.1:{server.server_port}', timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_lossless_scene_roundtrip():
    original = scene_payload(records(), 15)
    restored = decode_scene(encode_scene(original))
    np.testing.assert_array_equal(original['history'], restored['history'])
    for tag in original['views']:
        for a, b in zip(original['views'][tag], restored['views'][tag]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert restored['nav_command'] == 1 and restored['token'] == '16'


def test_real_http_request_and_metrics():
    planner = Planner()
    with service(planner) as client:
        trajectory, metrics = client.plan(records())
    assert trajectory.shape == (50, 3)
    assert metrics['inference_seconds'] == .02 and metrics['request_bytes'] > 0
    assert planner.payload['token'] == '16'


def test_bad_request_does_not_lock_out_next_inference():
    with service(Planner()) as client:
        with pytest.raises(RuntimeError, match='413'):
            client.request('/plan', b'')
        with pytest.raises(RuntimeError, match='400'):
            client.request('/plan', b'{bad json')
        assert client.plan(records())[0].shape == (50, 3)


def test_reject_mismatched_response_before_control():
    with service(Planner()) as client:
        client.request = lambda *args: dict(protocol=1, request_id='stale', token='16')
        with pytest.raises(RuntimeError, match='requested frame'):
            client.plan(records())


def test_server_error_propagates():
    class Broken(Planner):
        def plan_payload(self, payload):
            raise RuntimeError('GPU failed')
    with service(Broken()) as client:
        with pytest.raises(RuntimeError, match='500'):
            client.plan(records())


def test_invalid_scene_rejected():
    data = encode_scene(scene_payload(records(), 15))
    data['history'][0][0] = float('nan')
    with pytest.raises(ValueError, match='array'):
        decode_scene(data)
    data = encode_scene(scene_payload(records(), 15))
    data['driving_command'] = [0, 1, 0, 0]
    with pytest.raises(ValueError, match='command'):
        decode_scene(data)


@pytest.mark.parametrize('url', ['http://example.com:8765', 'https://localhost', 'http://localhost:8765/plan'])
def test_require_tunnel_url(url):
    with pytest.raises(ValueError, match='SSH tunnel'):
        RemotePlanner(url)


def test_busy_server_does_not_queue_second_request():
    entered, release = threading.Event(), threading.Event()
    class Slow(Planner):
        def plan_payload(self, payload):
            entered.set()
            assert release.wait(3)
            return super().plan_payload(payload)
    with service(Slow()) as client:
        outcomes = []
        worker = threading.Thread(target=lambda: outcomes.append(client.plan(records())))
        worker.start()
        try:
            assert entered.wait(2)
            with pytest.raises(RuntimeError, match='503'):
                client.plan(records())
        finally:
            release.set()
            worker.join(timeout=5)
        assert len(outcomes) == 1


def test_timeout_is_not_automatically_retried():
    import time
    class Slow(Planner):
        calls = 0
        def plan_payload(self, payload):
            self.calls += 1
            time.sleep(.2)
            return super().plan_payload(payload)
    planner = Slow()
    with service(planner) as client:
        client.timeout = .05
        with pytest.raises(RuntimeError, match='timed out'):
            client.plan(records())
    assert planner.calls == 1


def test_high_resolution_scene_and_history_resize():
    payload = scene_payload(records(), 15)
    image = Image.new('RGB', (1280, 768), (22, 41, 77))
    payload['views'] = {key: [image] * 4 for key in payload['views']}
    restored = decode_scene(encode_scene(payload, [(640, 384)] * 3 + [(1280, 768)]))
    for images in restored['views'].values():
        assert [i.size for i in images] == [(640, 384)] * 3 + [(1280, 768)]
        assert images[-1].getpixel((1000, 700)) == (22, 41, 77)


def test_rl_planner_requires_reasoning_before_model_load():
    from qwen_drive_carla.planner import QwenPlanner
    with pytest.raises(ValueError, match='requires reasoning'):
        QwenPlanner('unused', planner='rl', planning_mode='direct')
