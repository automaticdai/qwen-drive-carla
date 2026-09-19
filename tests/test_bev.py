import json
import math

import numpy as np
from PIL import Image
import pytest

from qwen_drive_carla.bev import SURROUND, camera_calibration, export_frame


@pytest.mark.parametrize('yaw', [0, 60, 120, 180, -120, -60])
def test_calibration_projects_optical_axis_to_center_and_preserves_handedness(yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    mount = np.eye(4)
    mount[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    mount[:3, 3] = [1, .5, 1.7]
    k, r, t = camera_calibration(mount, 896, 512, 90)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1)
    # 10 m along the CARLA camera's forward axis must be optical depth +10.
    carla_point = mount @ [10, 0, 0, 1]
    ego_point = carla_point[:3] * [1, -1, 1]
    optical = r.T @ (ego_point - t)
    np.testing.assert_allclose(optical, [0, 0, 10], atol=1e-12)
    pixel = k @ optical
    np.testing.assert_allclose(pixel[:2] / pixel[2], [448, 256])


def test_front_camera_left_and_above_project_left_and_up():
    k, r, t = camera_calibration(np.eye(4), 896, 512, 90)
    pixel = k @ (r.T @ (np.array([10, 2, 1]) - t))
    assert pixel[0] / pixel[2] < 448 and pixel[1] / pixel[2] < 256


def test_export_keeps_six_views_and_no_invented_ground_truth(tmp_path):
    record = dict(frame=11, timestamp=1.1, pose=[1, 2, 3],
                  images={name: Image.new('RGB', (896, 512)) for name, _, _ in SURROUND})
    cameras = {name: dict(fov=90, mount_matrix=np.eye(4).tolist()) for name, _, _ in SURROUND}
    directory = tmp_path / 'frame'
    export_frame(record, cameras, directory)
    data = json.loads((directory / 'frame.json').read_text())
    assert data['source'] == 'carla' and data['out_of_distribution']
    assert data['cam_order'] == [name for name, _, _ in SURROUND]
    assert len(list((directory / 'images').glob('*.png'))) == 6
    assert not (directory / 'gt.npz').exists()
    with np.load(directory / 'calib.npz') as calibration:
        assert calibration['cam_intrinsic'].shape == (6, 3, 3)
        np.testing.assert_equal(calibration['lidar2ego'], np.eye(4))
