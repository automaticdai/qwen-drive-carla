"""Offline dense-episode diagnosis using a CARLA OpenDRIVE map; no simulator ticks.

Historical episodes lack exact ego bounding boxes and other-vehicle snapshots.
The fallback footprint is an explicit approximation; missing collision evidence
is never reconstructed from current simulator actors.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from qwen_drive_carla.adapter import ego_vectors
from qwen_drive_carla.dense_traffic import ROAD, LANES
from qwen_drive_carla.safety import CarlaRoad, Footprint, GroundTruthGuard, Obstacle


def distance_to_path(point, path):
    a, delta = path[:-1], np.diff(path, axis=0)
    fraction = np.clip(np.sum((point-a)*delta, axis=1)/np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0, 1)
    return float(np.linalg.norm(point-(a+fraction[:, None]*delta), axis=1).min())


def diagnose(episode, world_map, footprint=None):
    rows = [json.loads(line) for line in (episode/'steps.jsonl').read_text().splitlines()]
    metadata = json.loads((episode/'metadata.json').read_text())
    recorded = metadata.get('safety', {}).get('footprint')
    footprint_source = 'recorded' if recorded else 'assumed 4.8 x 2.0 m, centered at actor origin'
    footprint = footprint or (Footprint(**recorded) if recorded else Footprint(2.4, 1.))
    config = metadata.get('safety', {})
    guard = GroundTruthGuard(footprint, CarlaRoad(world_map, ROAD, LANES),
                             wheelbase=config.get('wheelbase_m', 2.8),
                             max_steer_degrees=config.get('max_steer_degrees', 35.),
                             margin=config.get('margin_m', .25),
                             deceleration=config.get('deceleration_mps2', 3.),
                             reaction_seconds=config.get('reaction_seconds', .3))
    checks, errors, timing_errors, world_paths = [], [], [], []
    for i, row in enumerate(rows):
        file = episode/f"prediction-{row['frame']:08d}.json"
        if not file.exists():
            continue
        prediction = json.loads(file.read_text())
        trajectory = np.array(prediction['trajectory'])
        source = prediction.get('trajectory_source', 'Qwen')
        obstacles = [Obstacle(**{**o, 'footprint': Footprint(**o['footprint'])}) for o in row.get('obstacles', [])]
        decision = guard.check_plan(trajectory, row['pose'], obstacles)
        xy = ego_vectors(trajectory[:, :2], row['pose'][2]) + np.array(row['pose'][:2])
        path = np.vstack((row['pose'][:2], xy))
        world_paths.append((row, path, source))
        segment_errors = [distance_to_path(np.array(future['pose'][:2]), path)
                          for future in rows[i+1:i+6] if future['timestamp']-row['timestamp'] <= .50001]
        errors.extend(segment_errors)
        for future in rows[i+1:i+6]:
            age = future['timestamp']-row['timestamp']
            if age <= .50001:
                target = [np.interp(age, np.arange(51)*.1, path[:, axis]) for axis in (0, 1)]
                timing_errors.append(float(np.linalg.norm(np.array(future['pose'][:2])-target)))
        road_decision = guard.check_path(
            np.vstack((row['pose'], np.column_stack((xy, row['pose'][2]-np.degrees(trajectory[:, 2]))))),
            np.arange(51)*.1)
        checks.append(dict(frame=row['frame'], sim_seconds=row['sim_seconds'], trajectory_source=source, decision=asdict(decision),
                           road_only_decision=asdict(road_decision),
                           maximum_path_tracking_error_m=max(segment_errors, default=None)))
    report = dict(episode=str(episode), original_summary=json.loads((episode/'summary.json').read_text()),
                  footprint_source=footprint_source, guard=guard.configuration(),
                  collision_replay_available=bool(recorded) and all('obstacles' in row for row in rows if row['driver'] != 'autopilot'),
                  checked_plans=len(checks), rejected_plans=sum(c['decision']['reason'] != 'clear' for c in checks),
                  trajectory_sources=sorted({c['trajectory_source'] for c in checks}),
                  first_rejection=next((c for c in checks if c['decision']['reason'] != 'clear'), None),
                  first_road_rejection=next((c for c in checks if c['road_only_decision']['reason'] != 'clear'), None),
                  mean_path_tracking_error_m=float(np.mean(errors)) if errors else None,
                  maximum_path_tracking_error_m=max(errors, default=None), checks=checks,
                  mean_time_aligned_position_error_m=float(np.mean(timing_errors)) if timing_errors else None,
                  maximum_time_aligned_position_error_m=max(timing_errors, default=None),
                  limitations=['Offline checks on original states do not establish the outcome after braking.',
                               'Tracking error is nearest path distance over the next 0.5 seconds, not timing error.',
                               'Historical physics parameters use guard defaults if absent from metadata.'])
    return report, rows, world_paths


def plot(output, world_map, rows, paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout='constrained')
    for lane in LANES:
        points = [world_map.get_waypoint_xodr(ROAD, lane, float(s)) for s in np.arange(195, 260, .5)]
        points = [p for p in points if p is not None]
        center = np.array([[p.transform.location.x, p.transform.location.y] for p in points])
        yaw = np.radians([p.transform.rotation.yaw for p in points])
        normal = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        width = np.array([p.lane_width/2 for p in points])[:, None]
        for sign in (-1, 1):
            edge = center + sign*width*normal
            axes[0].plot(edge[:, 0], -edge[:, 1], color='gray', lw=.7)
    plotted_sources = set()
    for i, (row, path, source) in enumerate(paths):
        if i % 3 == 0:
            axes[0].plot(path[:, 0], -path[:, 1], alpha=.5, lw=1,
                         label=source+' proposed paths' if source not in plotted_sources else None)
            plotted_sources.add(source)
    xy = np.array([r['pose'][:2] for r in rows])
    axes[0].plot(xy[:, 0], -xy[:, 1], 'k-', lw=2, label='Actual motion')
    axes[0].scatter(*[xy[-1, 0], -xy[-1, 1]], color='red', label='Episode end')
    axes[0].set(xlabel='CARLA world X (m)', ylabel='−CARLA world Y (m)', title='Road lanes, predicted paths and actual motion', aspect='equal')
    axes[0].legend(fontsize=8)
    times = [r['sim_seconds'] for r in rows]
    axes[1].plot(times, [r['pose'][2] for r in rows], label='Vehicle yaw')
    import carla
    waypoints = [world_map.get_waypoint(carla.Location(x=r['pose'][0], y=r['pose'][1])) for r in rows]
    axes[1].plot(times, [p.transform.rotation.yaw for p in waypoints], label='Nearest lane yaw')
    axes[1].set(xlabel='Simulation time (s)', ylabel='CARLA yaw (degrees)', title='Vehicle versus nearest lane heading')
    axes[1].legend(); axes[1].grid(alpha=.25)
    fig.savefig(output/'diagnosis.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--map', type=Path, required=True, help='Town05.xodr, read offline')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true', help='Requires matplotlib')
    args = parser.parse_args()
    import carla
    world_map = carla.Map('Town05', args.map.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    report, rows, paths = diagnose(args.episode, world_map)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.plot:
        plot(args.output, world_map, rows, paths)
    print(json.dumps({k: v for k, v in report.items() if k not in ('checks', 'guard', 'original_summary')}, indent=2))


if __name__ == '__main__':
    main()
