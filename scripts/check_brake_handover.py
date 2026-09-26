"""Live regression: an initial full brake must be resent after Traffic Manager handover."""
import argparse
import json
from pathlib import Path

import numpy as np

from qwen_drive_carla.controller import Control
from qwen_drive_carla.dense_traffic import spawn_transform
from qwen_drive_carla.session import CarlaSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='172.30.64.1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import carla
    client = carla.Client(args.host, 2000); client.set_timeout(10)
    world = client.get_world()
    if world.get_settings().synchronous_mode or world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
        raise RuntimeError('Handover check requires an idle dedicated simulator')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    session = CarlaSession(args.host, tm_port=8010, spawn_transform=spawn_transform(world.get_map()))
    try:
        with session:
            session.autopilot(True)
            for _ in range(16):
                record = session.tick()
            initial_speed = float(np.linalg.norm(record['velocity']))
            before = record['observed_control']
            session.autopilot(False)
            for _ in range(35):
                session.apply(Control())  # Exactly the same full brake as at spawn.
                record = session.tick()
                rows.append(dict(frame=record['frame'], speed_mps=float(np.linalg.norm(record['velocity'])),
                                 observed_control=record['observed_control']))
        result = dict(initial_speed_mps=initial_speed, before_handover=before, rows=rows,
                      metadata=session.metadata)
        result['passed'] = (initial_speed > 1 and rows[-1]['speed_mps'] < .1 and
                            all(row['observed_control']['brake'] > .99 and row['observed_control']['throttle'] < .01
                                for row in rows))
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'metadata')}))
        if not result['passed']:
            raise RuntimeError('Braking handover regression failed; see result.json')
    finally:
        (args.output/'metadata.json').write_text(json.dumps(session.metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
