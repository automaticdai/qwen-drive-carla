"""Run isolated CARLA scenarios and retain failures, metrics and optional replays."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


def summarize(directory):
    summary_path = directory / 'summary.json'
    result = json.loads(summary_path.read_text()) if summary_path.exists() else {'status': 'setup_error'}
    result.pop('inference_seconds', None)
    result.pop('traceback', None)
    metadata_path = directory / 'metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    result['scene'] = metadata.get('scene', {})
    result['cleanup_errors'] = metadata.get('cleanup_errors', [])
    result['cleanup_verified'] = 'cleanup_errors' in metadata and not metadata['cleanup_errors']
    if metadata.get('error'):
        result['error'] = metadata['error']
    frames_path = directory / 'frames.jsonl'
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()] if frames_path.exists() else []
    distances = [r['nearest_traffic_m'] for r in frames if r.get('nearest_traffic_m') is not None]
    result['nearest_traffic_center_m'] = min(distances) if distances else None
    result['frames_with_traffic_within_15m'] = sum(d < 15 for d in distances)
    result['commands_observed'] = sorted({r['command'] for r in frames})
    signal_frames = [r for r in frames if 'at_traffic_light' in r]
    result['signal_logged_frames'] = len(signal_frames)
    result['stopped_at_red_frames'] = sum(r['at_traffic_light'] and r['traffic_light_state'] == 'Red'
                                         and math.hypot(*r['velocity']) < .1 for r in signal_frames)
    result['signal_check_available'] = bool(frames) and all(r.get('signal_check_available', False) for r in frames)
    # Completion without collision or lane events is deliberately stricter than route completion.
    result['clean_completion'] = (result['status'] == 'completed_route' and
                                  result.get('collision_events') == 0 and
                                  result.get('lane_invasion_events') == 0 and
                                  result.get('rejected_plans') == 0 and
                                  result['signal_check_available'] and
                                  result.get('red_light_events') == 0 and
                                  result.get('signal_ambiguous_events') == 0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--tm-port', type=int, default=8000, help='Local CARLA Traffic Manager port')
    parser.add_argument('--suite', type=Path, default=Path('scenarios/town01-junctions.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--planner-url', help='Qwen service through a local SSH tunnel')
    parser.add_argument('--planner-timeout', type=float, default=120)
    parser.add_argument('--camera-profile', choices=('small', 'high'), default='small')
    parser.add_argument('--reference', action='store_true', help='Run an autopilot reference for each case')
    parser.add_argument('--follow-camera', action='store_true')
    parser.add_argument('--replays', action='store_true', help='Create front-view MP4s if ffmpeg is installed')
    parser.add_argument('--resume', action='store_true', help='Resume the same saved suite, skipping recorded results')
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    if args.resume:
        if json.loads((args.output / 'suite.json').read_text()) != suite:
            raise RuntimeError('Cannot resume with a different suite')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'suite.json').write_text(json.dumps(suite, indent=2))
    import carla
    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    if client.get_server_version() != suite['carla_version']:
        raise RuntimeError('Scenario suite requires CARLA ' + suite['carla_version'])
    results_path = args.output / 'results.json'
    results = json.loads(results_path.read_text()) if args.resume and results_path.exists() else []
    for scenario in suite['scenarios']:
        for mode in (['record', 'closed-loop'] if args.reference else ['closed-loop']):
            name = scenario['name'] + '-' + mode
            if any(r['name'] == name for r in results):
                continue
            directory = args.output / name
            if directory.exists():
                raise RuntimeError(f'Incomplete episode directory exists: {directory}; inspect it before retrying')
            print('START ' + name, flush=True)
            # Actors/weather are restored by the session. Avoid repeated UE map loads.
            world = client.get_world()
            if world.get_settings().synchronous_mode:
                raise RuntimeError('Server is synchronous; another client or failed cleanup owns it')
            world.wait_for_tick(10)
            if world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
                raise RuntimeError('Scenario suite needs a dedicated world without existing vehicles/sensors')
            if world.get_map().name.split('/')[-1] != suite['map']:
                world = client.load_world(suite['map'])
            world.reset_all_traffic_lights()
            command = [sys.executable, '-m', 'qwen_drive_carla.run', '--host', args.host,
                       '--port', str(args.port), '--tm-port', str(args.tm_port), '--mode', mode, '--precision', 'nf4',
                       '--warmup-driver', 'autopilot', '--spawn-index', str(scenario['spawn']),
                       '--destination-index', str(scenario['destination']), '--seconds', str(args.seconds),
                       '--max-speed', '4', '--seed', str(args.seed), '--traffic', str(scenario['traffic']),
                       '--weather', scenario['weather'], '--camera-profile', args.camera_profile, '--output', str(directory)]
            if args.follow_camera:
                command.append('--follow-camera')
            if args.planner_url and mode != 'record':
                command.extend(['--planner-url', args.planner_url, '--planner-timeout', str(args.planner_timeout)])
            with (args.output / (name + '.log')).open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            result = dict(name=name, exit_code=completed.returncode, **summarize(directory))
            result['runtime_ok'] = completed.returncode == 0 and result['cleanup_verified']
            result['clean_completion'] = result['clean_completion'] and result['runtime_ok']
            results.append(result)
            (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(json.dumps(result), flush=True)
            if args.replays and shutil.which('ffmpeg') and list((directory / 'front').glob('*.png')):
                video = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-framerate', '10',
                                '-pattern_type', 'glob', '-i', str(directory / 'front' / '*.png'),
                                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                str(directory / 'front-replay.mp4')])
                if video.returncode:
                    print('Replay encoding failed for ' + name, flush=True)
            if result['cleanup_errors']:
                raise RuntimeError('Cleanup failed; inspect the server before continuing')
    print('Results: ' + str(args.output / 'results.json'), flush=True)


if __name__ == '__main__':
    main()
