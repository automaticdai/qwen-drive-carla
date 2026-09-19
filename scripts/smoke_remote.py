"""Send a recorded CARLA observation to the inference service without local torch."""
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
from qwen_drive_carla.adapter import load_records
from qwen_drive_carla.remote import RemotePlanner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8765')
parser.add_argument('--recording', type=Path, required=True)
parser.add_argument('--index', type=int, default=30)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
records = load_records(args.recording)
if args.index < 15 or args.index >= len(records):
    parser.error('Index must have 16 preceding/current observations')
history = records[args.index-15:args.index+1]
for record in history:
    loaded = {}
    for name, relative in record['images'].items():
        with Image.open(args.recording / relative) as image:
            loaded[name] = image.convert('RGB')
    record['images'] = loaded
args.output.mkdir(parents=True, exist_ok=False)
planner = RemotePlanner(args.url)
trajectory, metrics = planner.plan(history)
np.save(args.output / 'trajectory.npy', trajectory)
report = dict(model_profile=planner.loading_info, metrics=metrics,
              shape=list(trajectory.shape), finite=bool(np.isfinite(trajectory).all()))
(args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
