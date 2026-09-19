"""Persistent dense-traffic debug dashboard beside a visible CARLA server."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import numpy as np

from qwen_drive_carla.bev import SURROUND
from qwen_drive_carla.controller import Control, TrajectoryTracker
from qwen_drive_carla.dashboard import Dashboard
from qwen_drive_carla.debug_inference import RemoteDebugPlanner, image_string
from qwen_drive_carla.dense_traffic import ROAD, EGO_LANE, LANES, START_S, OvertakeMonitor, populate, scenario_waypoint, spawn_transform
from qwen_drive_carla.session import CarlaSession


def episode(args, dashboard, output):
    import carla
    client = carla.Client(args.host, 2000); client.set_timeout(60)
    world = client.get_world()
    if world.get_settings().synchronous_mode or world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
        raise RuntimeError('Dense debug needs an idle dedicated CARLA world')
    if world.get_map().name.split('/')[-1] != 'Town05':
        world = client.load_world('Town05')
    transform = spawn_transform(world.get_map())
    planner = RemoteDebugPlanner(args.planner_url, timeout=120, transport=args.image_transport)
    if not planner.loading_info.get('bev'):
        raise RuntimeError('Cloud service must advertise shared planning + BEV support')
    session = CarlaSession(args.host, tm_port=args.tm_port, camera_profile='high', camera_rig='debug', spawn_transform=transform)
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', collision_events=0, lane_invasion_events=0, red_light_events=0,
                   signal_ambiguous_events=0, plans=0, overtake='not yet')
    monitor = OvertakeMonitor()
    history, last_plan_tick, bev_time = deque(maxlen=16), None, None
    started = time.perf_counter()
    try:
        with session, (output / 'steps.jsonl').open('w') as log:
            session.configure_scene(follow_camera=True)
            lead = populate(session)
            world_map = session.world.get_map()
            locations = [scenario_waypoint(world_map, EGO_LANE, s).transform.location for s in np.arange(START_S + 2, START_S + 182, 2)]
            session.autopilot(True)
            session.tm.set_desired_speed(session.vehicle, 2 * 3.6)
            session.tm.set_path(session.vehicle, locations)
            physics = session.vehicle.get_physics_control()
            front, rear = physics.wheels[:2], physics.wheels[2:]
            wheelbase = np.linalg.norm(np.mean([[w.position.x,w.position.y,w.position.z] for w in front],axis=0)-np.mean([[w.position.x,w.position.y,w.position.z] for w in rear],axis=0))/100
            tracker = TrajectoryTracker(max_speed=4, wheelbase=float(wheelbase), max_steer_degrees=max(w.max_steer_angle for w in front))
            (output / 'metadata.json').write_text(json.dumps({**session.metadata, 'cloud':planner.loading_info,
                                                           'image_transport':args.image_transport}, indent=2))
            dashboard.publish(phase='ready', episode=output.name, traffic_count=24, initial_lane=EGO_LANE,
                              lead_id=lead.id, bev=None, trajectory=[], images=None, frame=None, metrics={},
                              outcome=None, error=None, driver='autopilot warmup', events=[], overtake='not yet',
                              sim_seconds=0, speed_mps=None, lane_id=None, lead_gap_m=None, passed_ticks=0,
                              control={}, bev_wall_received=None, bev_sim_age=None, qwen_inputs=None)
            for tick in range(round(args.seconds * 10)):
                # One preview tick populates cameras in a freshly reset paused scene.
                if (tick > 0 and not dashboard.allow_tick()) or dashboard.stopped:
                    summary['status'] = 'user_stopped'; break
                record = session.tick()
                record['command'] = 'straight'
                history.append(record)
                events = session.drain_events()
                for event in events:
                    summary[event['kind'] + '_events'] += 1
                ego_wp = world_map.get_waypoint(session.vehicle.get_location(), project_to_road=False)
                lead_wp = world_map.get_waypoint(lead.get_location())
                on_road = ego_wp is not None and ego_wp.road_id == ROAD and lead_wp.road_id == ROAD
                evidence = monitor.update(ego_wp.s if ego_wp else 0, lead_wp.s, ego_wp.lane_id if ego_wp else None,
                                          model_active=tick > 15, same_road=on_road, collision=bool(summary['collision_events']))
                speed = float(np.linalg.norm(record['velocity']))
                images = {name:image_string(record['images'][name].resize((448,256)), 'JPEG') for name,_,_ in SURROUND}
                state = dict(frame=record['frame'], sim_seconds=(tick+1)*.1, speed_mps=speed, lane_id=ego_wp.lane_id if ego_wp else None,
                             images=images, signal=record['traffic_light_state'], **evidence,
                             events={k:v for k,v in summary.items() if k.endswith('_events')},
                             bev_sim_age=None if bev_time is None else record['timestamp']-bev_time)
                dashboard.publish(**state)
                if summary['collision_events']:
                    summary['status']='collision'
                elif summary['red_light_events']:
                    summary['status']='red_light_violation'
                elif not on_road or ego_wp.lane_id not in LANES:
                    summary['status']='left_driving_carriageway'
                elif evidence['overtake']=='passed lead':
                    summary['status']='passed_lead'
                control = Control(reason='autopilot warmup' if summary['status']=='running' else 'episode_end')
                if summary['status']=='running' and len(history)==16:
                    if last_plan_tick is None:
                        session.autopilot(False)
                    if last_plan_tick is None or tick-last_plan_tick>=5:
                        (output / f"cameras-{record['frame']:08d}.json").write_text(json.dumps(dict(
                            source_frame=record['frame'], timestamp=record['timestamp'], images=images,
                            pose=record['pose'])))
                        dashboard.publish(phase='inference', driver='Qwen', inference_source_frame=record['frame'])
                        trajectory, response = planner.debug(list(history), session.metadata['cameras'])
                        tracker.set_plan(trajectory, record['pose'], record['timestamp'])
                        last_plan_tick, bev_time = tick, record['timestamp']
                        summary['plans'] += 1
                        (output / f"prediction-{record['frame']:08d}.json").write_text(json.dumps(response))
                        dashboard.publish(bev=response['bev'], trajectory=trajectory.tolist(), metrics=response['metrics'],
                                          bev_wall_received=time.time(), bev_sim_age=0, qwen_inputs=response['qwen_inputs'])
                    # Pause/stop during inference must take effect before applying new control.
                    with dashboard.condition:
                        stopped = dashboard.stopped
                    if stopped:
                        summary['status']='user_stopped'
                    else:
                        control=tracker.step(record['pose'],speed,record['timestamp'])
                        session.apply(control)
                summary.update(ticks=tick+1, **evidence)
                row={k:v for k,v in state.items() if k!='images'}
                row.update(control=asdict(control), pose=record['pose'], timestamp=record['timestamp'], event_details=events,
                           status=summary['status'], driver='Qwen' if tick>=15 else 'autopilot')
                log.write(json.dumps(row)+'\n'); log.flush()
                dashboard.publish(phase='driving' if tick>=15 else 'autopilot warmup', control=asdict(control))
                if summary['status']!='running':
                    break
            else:
                summary['status']='time_limit'
            session.autopilot(False); session.apply(Control(reason='episode_end'))
    except BaseException as exc:
        summary.update(status='error',error=f'{type(exc).__name__}: {exc}')
        dashboard.publish(error=summary['error'])
        (output/'error.txt').write_text(traceback.format_exc())
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        summary.update(wall_seconds=time.perf_counter()-started,cleanup_errors=session.metadata.get('cleanup_errors'))
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (output/'metadata.json').write_text(json.dumps({**session.metadata,'cloud':planner.loading_info,
                                                      'image_transport':args.image_transport},indent=2)+'\n')
        dashboard.publish(phase='finished — Reset scenario to try again',outcome=summary,**{k:summary[k] for k in ['overtake']})
        print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='172.30.64.1')
    parser.add_argument('--planner-url',default='http://127.0.0.1:8765')
    parser.add_argument('--tm-port',type=int,default=8010)
    parser.add_argument('--dashboard-port',type=int,default=8877)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--start-paused',action='store_true')
    parser.add_argument('--image-transport',choices=['lossless','legacy','jpeg'],default='jpeg',
                        help='Cached lossless WebP, original PNG, or explicit lossy JPEG quality 95')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or not 2<=args.seconds<=120:
        parser.error('Use 2–120 simulation seconds')
    args.output.mkdir(parents=True,exist_ok=False)
    dashboard=Dashboard(args.dashboard_port)
    dashboard.paused=args.start_paused
    print(f'Dashboard: http://localhost:{args.dashboard_port}',flush=True)
    try:
        index=1
        while True:
            try:
                episode(args,dashboard,args.output/f'episode-{index:03d}')
            except Exception as exc:
                dashboard.publish(phase='setup failed',error=f'{type(exc).__name__}: {exc}')
                print(traceback.format_exc(),flush=True)
            with dashboard.condition:
                while not dashboard.restart:
                    dashboard.condition.wait(timeout=.5)
                dashboard.restart=False; dashboard.stopped=False; dashboard.paused=True; dashboard.steps=0
            index+=1
    finally:
        dashboard.close()


if __name__=='__main__':
    main()
