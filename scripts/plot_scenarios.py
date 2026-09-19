"""Compare the planned, autopilot and Qwen paths for a scenario suite."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
args = parser.parse_args()
suite = json.loads((args.directory / 'suite.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
for ax, scenario in zip(axes.flat, suite['scenarios']):
    planned = False
    for mode, label, color in [('record', 'Autopilot', 'tab:green'), ('closed-loop', 'Qwen', 'tab:orange')]:
        directory = args.directory / (scenario['name'] + '-' + mode)
        if not (directory / 'frames.jsonl').exists():
            continue
        if not planned:
            route = np.array(json.loads((directory / 'route.json').read_text())['points'])
            ax.plot(*route.T, '--', color='gray', label='Planned route')
            ax.scatter(*route[0], marker='o', color='black', label='Start')
            ax.scatter(*route[-1], marker='x', color='black', label='Destination')
            planned = True
        rows = [json.loads(line) for line in (directory / 'frames.jsonl').read_text().splitlines()]
        if rows:
            xy = np.array([r['pose'][:2] for r in rows])
            ax.plot(*xy.T, color=color, label=label)
            ax.scatter(*xy[-1], color=color, marker='s', s=25)
    ax.set(title=scenario['name'], xlabel='CARLA world X (m)', ylabel='CARLA world Y (m)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle('Town01 junction paths · square marks episode termination')
path = args.directory / 'route-comparison.png'
fig.savefig(path, dpi=140)
print(path)
