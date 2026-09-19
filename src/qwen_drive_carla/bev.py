"""Calibrated surround-camera export for the official Qwen BEV perception head.

CARLA input is left-handed; exported ego is X forward, Y left, Z up.
The virtual lidar frame equals ego. No lidar or ground truth is fabricated.
"""
import json
import math
from pathlib import Path

import numpy as np

SURROUND = (
    ('CAM_FRONT', 'FRONT VIEW', 0),
    ('CAM_FRONT_RIGHT', 'FRONT RIGHT VIEW', 60),
    ('CAM_BACK_RIGHT', 'BACK RIGHT VIEW', 120),
    ('CAM_BACK', 'BACK VIEW', 180),
    ('CAM_BACK_LEFT', 'BACK LEFT VIEW', -120),
    ('CAM_FRONT_LEFT', 'FRONT LEFT VIEW', -60),
)


def camera_calibration(mount_matrix, width, height, fov):
    """Camera optical (right/down/forward) -> ego (forward/left/up)."""
    mount = np.asarray(mount_matrix, dtype=float)
    if mount.shape != (4, 4) or not np.isfinite(mount).all():
        raise ValueError('Invalid camera mount matrix')
    if width <= 0 or height <= 0 or not 0 < fov < 180:
        raise ValueError('Invalid camera intrinsics')
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    intrinsic = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    handedness = np.diag([1, -1, 1])
    optical_to_carla = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    return intrinsic, handedness @ mount[:3, :3] @ optical_to_carla, handedness @ mount[:3, 3]


def export_frame(record, cameras, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'images').mkdir()
    content, intrinsics, rotations, translations = [], [], [], []
    for name, tag, _ in SURROUND:
        spec = cameras[name]
        image = record['images'][name]
        if image.size != (896, 512):
            raise ValueError('BEV capture requires six 896x512 images')
        image.save(directory / 'images' / (name + '.png'))
        content.extend([dict(text='<' + tag + '>'), dict(image=name)])
        k, r, t = camera_calibration(spec['mount_matrix'], 896, 512, spec['fov'])
        intrinsics.append(k); rotations.append(r); translations.append(t)
    content.append(dict(text='Analyze the scene.'))
    frame = dict(dataset_type='nuscenes', source='carla', out_of_distribution=True,
                 calibration='measured CARLA sensor mounts; virtual lidar equals ego',
                 cam_order=[name for name, _, _ in SURROUND], content=content,
                 frame=record['frame'], timestamp=record['timestamp'], pose=record['pose'],
                 rig=cameras, ground_truth_available=False)
    (directory / 'frame.json').write_text(json.dumps(frame, indent=2) + '\n')
    np.savez(directory / 'calib.npz', cam_intrinsic=intrinsics,
             sensor2lidar_rotation=rotations, sensor2lidar_translation=translations,
             lidar2ego=np.eye(4))


def load_frame(directory):
    # Upstream PerceptionFrame requires gt.npz even though inference never uses it.
    # Override only loading; retain its actual projection/processor contract.
    from PIL import Image
    from qwen_drive_perception.dataset import PerceptionFrame

    class CarlaPerceptionFrame(PerceptionFrame):
        def __init__(self, path):
            self.path = Path(path)
            self.frame = json.loads((self.path / 'frame.json').read_text())
            if self.frame.get('source') != 'carla' or self.frame['cam_order'] != [n for n, _, _ in SURROUND]:
                raise ValueError('Expected a calibrated CARLA six-camera export')
            self.token = self.path.name
            self.dataset_type = self.frame['dataset_type']
            self.cam_order, self.content = self.frame['cam_order'], self.frame['content']
            with np.load(self.path / 'calib.npz', allow_pickle=False) as calibration:
                for name, shape in [('cam_intrinsic', (6, 3, 3)), ('sensor2lidar_rotation', (6, 3, 3)),
                                    ('sensor2lidar_translation', (6, 3)), ('lidar2ego', (4, 4))]:
                    value = calibration[name]
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError('Invalid calibration: ' + name)
                    setattr(self, name, value)
            self.gt, self.lidar = {}, None

        def image(self, cam):
            if cam not in self.cam_order:
                raise ValueError('Unknown camera')
            with Image.open(self.path / 'images' / (cam + '.png')) as image:
                return image.convert('RGB')

    return CarlaPerceptionFrame(directory)
