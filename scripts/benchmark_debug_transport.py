"""Compare PNG and optimized JPEG on matched CARLA captures without applying plans."""
import argparse
import json
from pathlib import Path
import statistics

from qwen_drive_carla.debug_inference import RemoteDebugPlanner
from qwen_drive_carla.dense_traffic import spawn_transform, populate
from qwen_drive_carla.session import CarlaSession


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='172.30.64.1')
    parser.add_argument('--url',default='http://127.0.0.1:8765')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import carla
    client=carla.Client(args.host,2000);client.set_timeout(30);world=client.get_world()
    if world.get_settings().synchronous_mode or world.get_actors().filter('vehicle.*') or world.get_actors().filter('sensor.*'):
        raise RuntimeError('Need idle CARLA for matched transport captures')
    args.output.mkdir(parents=True,exist_ok=False)
    records=[]
    session=CarlaSession(args.host,tm_port=8010,camera_profile='high',camera_rig='debug',spawn_transform=spawn_transform(world.get_map()))
    try:
        with session:
            session.configure_scene(follow_camera=True)
            populate(session)
            for _ in range(26):
                records.append(session.tick())  # Ego stays braked; no model control.
    finally:
        (args.output/'capture.json').write_text(json.dumps(session.metadata,indent=2))
    clients={mode:RemoteDebugPlanner(args.url,transport=mode) for mode in ['legacy','jpeg']}
    samples=[]
    for index in range(3):
        history=records[index*5:index*5+16]
        for mode in (['legacy','jpeg'] if index%2==0 else ['jpeg','legacy']):
            _,response=clients[mode].debug(history,session.metadata['cameras'])
            metrics=response['metrics']
            metrics['non_inference_seconds']=metrics['request_seconds']-metrics['seconds']-metrics['bev_seconds']
            sample=dict(mode=mode,source_frame=history[-1]['frame'],metrics=metrics)
            samples.append(sample)
            (args.output/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
            print(json.dumps(sample),flush=True)
    comparison={}
    for mode in clients:
        rows=[s['metrics'] for s in samples if s['mode']==mode]
        comparison[mode]={k:statistics.mean(r[k] for r in rows) for k in
                          ['request_seconds','request_bytes','encode_seconds','http_seconds','non_inference_seconds','seconds','bev_seconds']}
    (args.output/'comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    print(json.dumps(comparison,indent=2),flush=True)


if __name__=='__main__':
    main()
