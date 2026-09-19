"""Capture synchronized, calibrated surround frames for cloud BEV perception."""
import argparse
import json
from pathlib import Path

from qwen_drive_carla.bev import export_frame
from qwen_drive_carla.session import CarlaSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8010)
    parser.add_argument('--spawn-index', type=int, default=241)
    parser.add_argument('--traffic', type=int, default=8)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.frames <= 20 or not 0 <= args.traffic <= 50:
        parser.error('Use 1–20 frames and 0–50 traffic vehicles')
    import carla
    client = carla.Client(args.host, args.port); client.set_timeout(30)
    world = client.get_world()
    if world.get_settings().synchronous_mode or world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
        raise RuntimeError('Capture requires an idle dedicated CARLA world')
    args.output.mkdir(parents=True, exist_ok=False)
    session = CarlaSession(args.host, args.port, args.tm_port, args.spawn_index, camera_rig='surround')
    try:
        with session:
            session.configure_scene(traffic=args.traffic, follow_camera=True)
            # Ego remains braked. Allow exposure and actors to settle before capture.
            for _ in range(10):
                session.tick()
            for _ in range(args.frames):
                record = session.tick()
                export_frame(record, session.metadata['cameras'], args.output / f"{record['frame']:08d}")
                print('Captured', record['frame'], flush=True)
    finally:
        (args.output / 'capture.json').write_text(json.dumps(session.metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
