import math
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from qwen_drive_carla.adapter import ego_poses, ego_vectors, scene_payload
from qwen_drive_carla.record import image_for_frame


def records():
    return [dict(frame=100 + i, timestamp=i / 10, pose=[i, 0, 0], velocity=[10, 0],
                 acceleration=[0, 0], command="straight",
                 images={name: f"{name}/{i}.png" for name in ("front", "front_left", "front_right")})
            for i in range(16)]


def test_straight_history_and_camera_times():
    data = scene_payload(records(), 15)
    np.testing.assert_allclose(data["history"][:, 0], np.arange(-15, 1))
    np.testing.assert_allclose(data["history"][-1], [0, 0, 0])
    assert data["views"]["<FRONT VIEW>"] == [f"front/{i}.png" for i in (0, 5, 10, 15)]
    assert data["driving_command"] == [0, 1, 0, 0]


def test_rotated_ego_and_left_sign():
    # Facing CARLA +Y: east is ego-left; north is ego-forward.
    np.testing.assert_allclose(ego_vectors([[0, 2], [3, 0]], 90), [[2, 0], [0, 3]], atol=1e-12)
    result = ego_poses([[10, 22, 90], [13, 20, 0]], [10, 20, 90])
    np.testing.assert_allclose(result, [[2, 0, 0], [0, 3, math.pi / 2]], atol=1e-12)


def test_heading_wrap():
    result = ego_poses([[0, 0, 179]], [0, 0, -179])
    assert result[0, 2] == pytest.approx(math.radians(2))


@pytest.mark.parametrize("change", ["timestamp", "frame", "nan"])
def test_reject_corrupt_history(change):
    data = records()
    if change == "timestamp":
        data[5]["timestamp"] += .02
    elif change == "frame":
        data[5]["frame"] += 1
    else:
        data[5]["velocity"][0] = float("nan")
    with pytest.raises(ValueError):
        scene_payload(data, 15)


def test_incomplete_history():
    with pytest.raises(ValueError):
        scene_payload(records(), 14)


def test_sensor_queue_skips_old_but_rejects_future():
    messages = queue.Queue()
    for frame in (1, 2, 4):
        messages.put(SimpleNamespace(frame=frame))
    assert image_for_frame(messages, 2, .1).frame == 2
    with pytest.raises(RuntimeError, match="skipped"):
        image_for_frame(messages, 3, .1)


def test_sensor_timeout():
    with pytest.raises(TimeoutError):
        image_for_frame(queue.Queue(), 1, .001)
