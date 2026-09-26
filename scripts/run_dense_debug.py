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
from qwen_drive_carla.candidates import candidate_array, select_candidate
from qwen_drive_carla.lane_following import carla_following_candidates
from qwen_drive_carla.dashboard import Dashboard
from qwen_drive_carla.debug_inference import LocalDebugPlanner, RemoteDebugPlanner, image_string
from qwen_drive_carla.dense_traffic import ROAD, EGO_LANE, LANES, START_S, OvertakeMonitor, populate, scenario_waypoint, spawn_transform
from qwen_drive_carla.session import CarlaSession
from qwen_drive_carla.safety import (CarlaRoad, Decision, GroundTruthGuard,
                                     actor_footprint, capture_obstacles)


def episode(args, dashboard, output):
    import carla
    client = carla.Client(args.host, 2000); client.set_timeout(60)
    world = client.get_world()
    if world.get_settings().synchronous_mode or world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
        raise RuntimeError('Dense debug needs an idle dedicated CARLA world')
    if world.get_map().name.split('/')[-1] != 'Town05':
        world = client.load_world('Town05')
    transform = spawn_transform(world.get_map())
    planner = args.local_planner_instance if args.local_planner else RemoteDebugPlanner(args.planner_url, timeout=120, transport=args.image_transport)
    if not args.local_planner and not planner.loading_info.get('bev'):
        raise RuntimeError('Cloud service must advertise shared planning + BEV support')
    session = CarlaSession(args.host, tm_port=args.tm_port, camera_profile=args.image_profile if args.local_planner else 'high', camera_rig='debug', spawn_transform=transform)
    output.mkdir(parents=True, exist_ok=False)
    if args.local_planner:
        planner.input_directory = output / 'planner-inputs'
    evaluation_mode = 'Qwen evaluation' if args.safety_mode == 'off' else 'Assisted driving'
    summary = dict(status='running', collision_events=0, lane_invasion_events=0, red_light_events=0,
                   signal_ambiguous_events=0, plans=0, overtake='not yet', rejected_plans=0,
                   safety_intervention_ticks=0, safety_mode=args.safety_mode, evaluation_mode=evaluation_mode,
                   candidates_evaluated=0, candidates_accepted=0, alternative_selections=0,
                   fallback_plans=0, qwen_rejected_batches=0, fallback=getattr(args, 'fallback', 'none'))
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
            guard = GroundTruthGuard(actor_footprint(session.vehicle),
                                     CarlaRoad(world_map, ROAD, LANES, session.vehicle.get_location().z),
                                     wheelbase=tracker.wheelbase,
                                     max_steer_degrees=np.degrees(tracker.max_steer)) if args.safety_mode == 'carla' else None
            session.metadata['safety'] = guard.configuration() if guard else {'source': 'disabled'}
            plan_decision = Decision('no_plan')
            active_plan_source = 'Qwen'
            session.metadata['evaluation'] = dict(mode=evaluation_mode, selection='fixed candidate zero' if not guard else 'CARLA oracle ranking',
                controller='pure pursuit + PID', speed_cap_mps=4, warmup_seconds=1.5, replan_seconds=.5, simulation_paused_during_inference=True)
            session.metadata['fallback'] = getattr(args, 'fallback', 'none')
            (output / 'metadata.json').write_text(json.dumps({**session.metadata, 'cloud':planner.loading_info,
                                                           'image_transport':args.image_transport}, indent=2))
            dashboard.publish(phase='ready', evaluation_mode=evaluation_mode, execution='local' if args.local_planner else 'cloud', bev_enabled=not args.local_planner, episode=output.name, traffic_count=24, initial_lane=EGO_LANE,
                              lead_id=lead.id, bev=None, trajectory=[], images=None, frame=None, metrics={},
                              outcome=None, error=None, driver='autopilot warmup', events=[], overtake='not yet',
                              sim_seconds=0, speed_mps=None, lane_id=None, lead_gap_m=None, passed_ticks=0,
                              control={}, bev_wall_received=None, bev_sim_age=None, qwen_inputs=None,
                              safety_source=guard.source if guard else 'disabled', safety={}, selection={},
                              trajectory_source='Qwen', fallback_selection=None)
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
                             observed_control=record.get('observed_control'),
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
                proposed_control = control
                decision = Decision()
                obstacles = []
                if summary['status']=='running' and len(history)==16:
                    if guard:
                        obstacles = capture_obstacles(session.world, session.vehicle)
                    if last_plan_tick is None:
                        session.autopilot(False)
                    if last_plan_tick is None or tick-last_plan_tick>=5:
                        (output / f"cameras-{record['frame']:08d}.json").write_text(json.dumps(dict(
                            source_frame=record['frame'], timestamp=record['timestamp'], images=images,
                            pose=record['pose'])))
                        dashboard.publish(phase='inference', inference_source_frame=record['frame'])
                        trajectory, response = planner.debug(list(history), session.metadata['cameras'])
                        candidates = candidate_array(response.get('candidates', [trajectory]))
                        response['candidates'] = candidates.tolist()
                        selection = None
                        if guard:
                            selection = select_candidate(candidates, guard, record['pose'], speed, obstacles,
                                                         progress=guard.road_contains.progress, max_speed=tracker.max_speed)
                            plan_decision = selection.decision
                            if selection.index is not None:
                                trajectory = candidates[selection.index]
                            response['selection'] = selection.log()
                            summary['candidates_evaluated'] += len(candidates)
                            summary['candidates_accepted'] += response['selection']['valid_count']
                            summary['qwen_rejected_batches'] += int(selection.index is None)
                        else:
                            # Unassisted mode always executes candidate zero, without ground-truth ranking.
                            trajectory, plan_decision = candidates[0], Decision()
                            response['selection'] = dict(selected_index=0, count=len(candidates),
                                                          valid_count=None, candidates=[], source='unassisted first sample')
                        active_plan_source = 'Qwen'
                        stationary = (selection is not None and selection.index is not None and
                                      selection.evaluations[selection.index].get('initial_target_speed', 0) < .15)
                        if guard and getattr(args, 'fallback', 'none') == 'lane-follow' and (not plan_decision.safe or stationary):
                            fallback = carla_following_candidates(world_map, record['pose'], speed, record['traffic_light_state'],
                                                                  road_id=ROAD, lanes=LANES)
                            response['fallback_candidates'] = fallback.tolist()
                            response['fallback_trigger'] = 'qwen_stationary' if stationary else 'qwen_rejected'
                            if len(fallback):
                                fallback_selection = select_candidate(fallback, guard, record['pose'], speed, obstacles,
                                                                      progress=guard.road_contains.progress, max_speed=tracker.max_speed)
                                response['fallback_selection'] = fallback_selection.log()
                                if fallback_selection.index is not None:
                                    trajectory = fallback[fallback_selection.index]
                                    plan_decision = fallback_selection.decision
                                    active_plan_source = 'CARLA lane following'
                                    summary['fallback_plans'] += 1
                        if selection is not None and selection.index is not None and active_plan_source == 'Qwen':
                            summary['alternative_selections'] += int(selection.index != 0)
                        response['trajectory_source'] = active_plan_source
                        response['trajectory'] = trajectory.tolist()
                        if plan_decision.safe:
                            try:
                                tracker.set_plan(trajectory, record['pose'], record['timestamp'])
                            except ValueError:
                                plan_decision = Decision('invalid_trajectory')
                                if not guard:
                                    summary['status'] = 'invalid_trajectory'
                        if not plan_decision.safe:
                            tracker.clear_plan()
                            summary['rejected_plans'] += 1
                        response['safety'] = dict(source=guard.source if guard else 'disabled',
                                                  plan_decision=asdict(plan_decision))
                        last_plan_tick, bev_time = tick, record['timestamp']
                        summary['plans'] += 1
                        (output / f"prediction-{record['frame']:08d}.json").write_text(json.dumps(response))
                        display_trajectory = trajectory.tolist() if trajectory.shape == (50, 3) and np.isfinite(trajectory).all() else []
                        dashboard.publish(bev=response['bev'], trajectory=display_trajectory, metrics=response['metrics'],
                                          bev_wall_received=time.time(), bev_sim_age=0, qwen_inputs=response['qwen_inputs'],
                                          selection=response['selection'], trajectory_source=active_plan_source,
                                          fallback_selection=response.get('fallback_selection'))
                    # Pause/stop during inference must take effect before applying new control.
                    with dashboard.condition:
                        stopped = dashboard.stopped
                    if stopped:
                        summary['status']='user_stopped'
                    else:
                        proposed_control=tracker.step(record['pose'],speed,record['timestamp'])
                        control = proposed_control
                        if guard:
                            control, decision = guard.filter_control(proposed_control, record['pose'], speed,
                                                                      obstacles, plan_decision=plan_decision)
                            if not decision.safe:
                                tracker.stop('safety_override')  # Reset PID windup during forced braking.
                                summary['safety_intervention_ticks'] += 1
                        session.apply(control)
                summary.update(ticks=tick+1, **evidence)
                row={k:v for k,v in state.items() if k!='images'}
                row.update(control=asdict(control), velocity=record['velocity'], acceleration=record.get('acceleration'), pose=record['pose'], timestamp=record['timestamp'], event_details=events,
                           status=summary['status'], driver=(('CARLA lane following + guard' if active_plan_source != 'Qwen'
                                                              else 'Qwen + CARLA guard') if guard else 'Qwen') if tick>=15 else 'autopilot',
                           proposed_control=asdict(proposed_control),
                           safety=dict(source=guard.source if guard else 'disabled', decision=asdict(decision),
                                       plan_decision=asdict(plan_decision)),
                           obstacles=[asdict(obstacle) for obstacle in obstacles])
                log.write(json.dumps(row)+'\n'); log.flush()
                dashboard.publish(phase='driving' if tick>=15 else 'autopilot warmup', control=asdict(control),
                                  safety=row['safety'], driver=row['driver'])
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
    parser.add_argument('--local-planner',action='store_true',help='Run planning on the local GPU; disables BEV perception')
    parser.add_argument('--model',type=Path,default=Path('models/Qwen-Drive-1.0-4B'))
    parser.add_argument('--precision',choices=['nf4','bf16'],default='nf4',help='Local planner precision')
    parser.add_argument('--image-profile',choices=['small','high'],default='small',help='Local planner camera resolution')
    parser.add_argument('--candidate-count', type=int, choices=range(1, 7), default=1,
                        help='Local Qwen samples per scene; remote count is configured on the service')
    parser.add_argument('--tm-port',type=int,default=8010)
    parser.add_argument('--dashboard-port',type=int,default=8877)
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--start-paused',action='store_true')
    parser.add_argument('--safety-mode', choices=['carla', 'off'], default='off',
                        help='Default off evaluates Qwen; carla enables ground-truth assistance')
    parser.add_argument('--fallback', choices=['none', 'lane-follow'], default='none',
                        help='Optional guarded CARLA lane following when Qwen rejects or stalls; requires carla safety mode')
    parser.add_argument('--image-transport',choices=['lossless','legacy','jpeg'],default='jpeg',
                        help='Cached lossless WebP, original PNG, or explicit lossy JPEG quality 95')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.fallback != 'none' and args.safety_mode != 'carla':
        parser.error('Lane-following fallback requires --safety-mode carla')
    if not np.isfinite(args.seconds) or not 2<=args.seconds<=120:
        parser.error('Use 2–120 simulation seconds')
    args.output.mkdir(parents=True,exist_ok=False)
    dashboard=Dashboard(args.dashboard_port)
    dashboard.paused=args.start_paused
    print(f'Dashboard: http://localhost:{args.dashboard_port}',flush=True)
    try:
        if args.local_planner:
            dashboard.publish(phase='loading local model', execution='local', bev_enabled=False)
            args.local_planner_instance=LocalDebugPlanner(args.model,args.precision,args.image_profile,args.candidate_count)
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
