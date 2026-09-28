"""Replay exact archived planner inputs under one precision, without a simulator."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from qwen_drive_carla.adapter import ego_vectors
from qwen_drive_carla.controller import validate_trajectory
from qwen_drive_carla.dense_traffic import ROAD, LANES
from qwen_drive_carla.input_archive import load_payload
from qwen_drive_carla.planner import QwenPlanner
from qwen_drive_carla.safety import CarlaRoad, Footprint, GroundTruthGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=Path, nargs='+', required=True)
    parser.add_argument('--precision', choices=['bf16', 'nf4'], required=True)
    parser.add_argument('--model', type=Path, default=Path('models/Qwen-Drive-1.0-4B'))
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(precision=args.precision, image_profile='small', seed=42,
                  status='loading', results=[],
                  limitations=['Offline replay on NF4-generated states, not a BF16 closed-loop rollout.',
                               'Road checks assume a centered 4.8 x 2.0 m footprint plus 0.25 m margin.',
                               'No other-vehicle collision replay; differences are not accuracy scores.'])

    def save():
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    save()
    try:
        import carla
        world_map = carla.Map('Town05', args.map.read_text())
        guard = GroundTruthGuard(Footprint(2.4, 1.), CarlaRoad(world_map, ROAD, LANES))
        started = time.perf_counter()
        planner = QwenPlanner(args.model, precision=args.precision, image_profile='small')
        planner.model.config.noise_seed = 42
        report.update(load_seconds=time.perf_counter()-started, loading=planner.loading_info,
                      status='running', guard=guard.configuration())
        save()
        for episode in args.episodes:
            rows = {row['frame']: row for row in
                    map(json.loads, (episode / 'steps.jsonl').read_text().splitlines())}
            for archive in sorted((episode / 'planner-inputs').glob('input-*.json'),
                                  key=lambda p: int(p.stem.split('-')[-1])):
                payload = load_payload(archive)
                frame = int(payload['token'])
                baseline = json.loads((episode / f'prediction-{frame:08d}.json').read_text())
                settings = json.loads(archive.read_text())['model']
                if (settings['image_profile'] != 'small' or settings['planner'] != 'sft'
                        or baseline['metrics']['seed_base'] != 42):
                    raise ValueError('Archive does not match small/SFT/seed-42 comparison')
                trajectory, metrics = planner.plan_payload(payload)
                np.save(args.output / f'{episode.name}-{frame}.npy', trajectory)
                pose = rows[frame]['pose']
                xy = ego_vectors(trajectory[:, :2], pose[2]) + np.asarray(pose[:2])
                road = guard.check_path(
                    np.vstack((pose, np.column_stack((xy, pose[2]-np.degrees(trajectory[:, 2]))))),
                    np.arange(51)*.1)
                validation_error = None
                try:
                    validate_trajectory(trajectory)
                except ValueError as error:
                    validation_error = str(error)
                diff = np.linalg.norm(trajectory[:, :2]-np.asarray(baseline['trajectory'])[:, :2], axis=1)
                result = dict(episode=str(episode), frame=frame, archive=str(archive),
                              metrics=metrics, validation_error=validation_error,
                              guard_decision=asdict(guard.check_plan(trajectory, pose)),
                              road_decision=asdict(road),
                              mean_xy_difference_from_saved_nf4_m=float(diff.mean()),
                              endpoint_difference_from_saved_nf4_m=float(diff[-1]))
                report['results'].append(result)
                save()
                print(json.dumps(result), flush=True)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='error', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
