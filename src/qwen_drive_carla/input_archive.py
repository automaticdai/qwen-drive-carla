"""Lossless, replayable local planner payloads, captured before preprocessing."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def save_payload(directory, payload, loading_info):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    document = {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value
                for key, value in payload.items() if key != 'views'}
    document['views'] = {}
    for tag, frames in payload['views'].items():
        document['views'][tag] = []
        for image in frames:
            digest = hashlib.sha256(str((image.mode, image.size)).encode() + image.tobytes()).hexdigest()
            path = directory / f'{digest}.png'
            if not path.exists():
                image.save(path, format='PNG')
            document['views'][tag].append(dict(path=path.name, sha256=digest))
    path = directory / f"input-{payload['token']}.json"
    path.write_text(json.dumps(dict(payload=document, model=loading_info,
                                   stage='local RGB inputs before Qwen resizing/preprocessing'), indent=2))
    return path


def load_payload(path):
    path = Path(path)
    payload = json.loads(path.read_text())['payload']
    views = {}
    for tag, frames in payload['views'].items():
        views[tag] = []
        for frame in frames:
            with Image.open(path.parent / frame['path']) as source:
                image = source.copy()
            digest = hashlib.sha256(str((image.mode, image.size)).encode() + image.tobytes()).hexdigest()
            if digest != frame['sha256']:
                raise ValueError('Planner input image checksum mismatch')
            views[tag].append(image)
    payload['views'] = views
    for key in ('history', 'history_velocity', 'history_acceleration', 'ego_velocity', 'ego_acceleration'):
        payload[key] = np.asarray(payload[key], dtype=float)
    return payload
