"""Compare guarded one- versus multi-candidate driving with one local model load."""
import argparse
import json
from pathlib import Path

import numpy as np

from qwen_drive_carla.dashboard import Dashboard
from qwen_drive_carla.debug_inference import LocalDebugPlanner
from run_dense_debug import episode


def episode_metrics(path):
    summary = json.loads((path/'summary.json').read_text())
    rows = [json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()]
    controlled = [row for row in rows if row['driver'] != 'autopilot']
    xy = np.array([row['pose'][:2] for row in controlled])
    distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.
    predictions = [json.loads(p.read_text()) for p in sorted(path.glob('prediction-*.json'))]
    request_times = [p['metrics']['request_seconds'] for p in predictions]
    reasons = {}
    for prediction in predictions:
        for candidate in prediction.get('selection', {}).get('candidates', []):
            reason = candidate['decision']['reason']
            reasons[reason] = reasons.get(reason, 0)+1
    return dict(**summary, controlled_distance_m=distance,
                moving_controlled_ticks=sum(row['speed_mps'] > .2 for row in controlled),
                mean_inference_request_seconds=float(np.mean(request_times)) if request_times else None,
                candidate_decisions=reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='172.30.64.1')
    parser.add_argument('--model', type=Path, default=Path('models/Qwen-Drive-1.0-4B'))
    parser.add_argument('--precision', choices=['nf4', 'bf16'], default='nf4')
    parser.add_argument('--image-profile', choices=['small', 'high'], default='small')
    parser.add_argument('--counts', type=int, nargs='+', choices=range(1, 7), default=[1, 3])
    parser.add_argument('--repeats', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--fallback', choices=['none', 'lane-follow'], default='none')
    parser.add_argument('--tm-port', type=int, default=8010)
    parser.add_argument('--dashboard-port', type=int, default=8877)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 2 <= args.seconds <= 120:
        parser.error('Use 2–120 simulation seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    args.local_planner = True
    args.image_transport = 'jpeg'
    args.safety_mode = 'carla'
    results = []
    dashboard = Dashboard(args.dashboard_port)
    try:
        dashboard.publish(phase='loading local model', execution='local', bev_enabled=False)
        args.local_planner_instance = LocalDebugPlanner(args.model, args.precision, args.image_profile, args.counts[0])
        for repeat in range(args.repeats):
            # Alternate order to expose warmup/order effects rather than attributing them to sampling.
            counts = args.counts if repeat % 2 == 0 else list(reversed(args.counts))
            for count in counts:
                args.local_planner_instance.num_candidates = count
                args.local_planner_instance.loading_info['candidate_count'] = count
                path = args.output/f'candidates-{count}-repeat-{repeat+1}'
                print(f'Starting {path.name}', flush=True)
                episode(args, dashboard, path)
                result = dict(candidate_count=count, repeat=repeat+1, episode=str(path), **episode_metrics(path))
                results.append(result)
                (args.output/'comparison.json').write_text(json.dumps(results, indent=2)+'\n')
                print(json.dumps(result), flush=True)
                if result['status'] in ('error', 'user_stopped'):
                    return
    finally:
        dashboard.close()


if __name__ == '__main__':
    main()
