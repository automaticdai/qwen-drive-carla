import numpy as np
import pytest
from PIL import Image

from qwen_drive_carla.adapter import scene_payload
from qwen_drive_carla.input_archive import save_payload, load_payload


def test_exact_scene_roundtrip_and_corruption(tmp_path):
    records = [dict(frame=i, timestamp=i / 10, pose=[i, 2, 90], velocity=[0, 10],
                    acceleration=[0, 0], command='left', images={name: Image.new('RGB', (12, 8), (i, 42, 99))
                    for name in ('front', 'front_left', 'front_right')}) for i in range(16)]
    original = scene_payload(records, 15)
    path = save_payload(tmp_path, original, {'precision': 'nf4'})
    restored = load_payload(path)
    for key in original:
        if key != 'views':
            np.testing.assert_equal(restored[key], original[key])
    for tag, frames in original['views'].items():
        for before, after in zip(frames, restored['views'][tag]):
            assert before.mode == after.mode and before.size == after.size
            assert before.tobytes() == after.tobytes()
    assert len(list(tmp_path.glob('*.png'))) == 4  # identical pixels deduplicated
    save_payload(tmp_path, original, {})
    assert len(list(tmp_path.glob('*.png'))) == 4
    for image in tmp_path.glob('*.png'):
        Image.new('RGB', (12, 8)).save(image)
    with pytest.raises(ValueError, match='checksum'):
        load_payload(path)
