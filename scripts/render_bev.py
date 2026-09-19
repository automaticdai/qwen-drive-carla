"""Render model-predicted BEV outputs; no simulator ground truth is substituted."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from qwen_drive_perception.configuration_perception import MAP_PALETTE, MAP_CLASS_NAMES, OCC_CLASS_NAMES, OCC_PALETTE, DET_CLASS_NAMES
from qwen_drive_perception.geometry import lidar_to_ego_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=Path, required=True)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score', type=float, default=.3)
    args = parser.parse_args()
    if not 0 <= args.score <= 1:
        parser.error('--score must be in [0,1]')
    from PIL import Image
    import torch
    frame = json.loads((args.frame / 'frame.json').read_text())
    with np.load(args.prediction, allow_pickle=False) as archive:
        data = dict(archive)
    with np.load(args.frame / 'calib.npz', allow_pickle=False) as calibration:
        boxes = lidar_to_ego_boxes(torch.tensor(data['boxes']), torch.tensor(calibration['lidar2ego'])).numpy()
    fig = plt.figure(figsize=(16, 12), layout='constrained')
    axes = fig.subplots(3, 3)
    for ax, camera in zip(axes[:2].flat, frame['cam_order']):
        path = args.frame / 'images' / (camera + ('.png' if frame.get('source') == 'carla' else '.jpg'))
        with Image.open(path) as image:
            ax.imshow(image)
        ax.set_title(camera); ax.axis('off')
    ax = axes[2, 0]
    for box, score, label in zip(boxes, data['scores'], data['labels']):
        if score < args.score:
            continue
        x, y, _, length, width, _, yaw = box[:7]
        corners = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
        c, s = np.cos(yaw), np.sin(yaw)
        corners = corners @ np.array([[c, s], [-s, c]]) + [x, y]
        ax.add_patch(Polygon(np.column_stack((-corners[:, 1], corners[:, 0])), fill=False, edgecolor='tab:orange'))
        ax.text(-y, x, f'{DET_CLASS_NAMES[int(label)]} {score:.2f}', fontsize=5)
    ax.scatter([0], [0], marker='^', color='black')
    ax.set(ylim=(-40, 40))
    # Horizontal display coordinate is right-positive (-ego Y); left is negative.
    ax.set_xlim(-40, 40)
    ax.set_title(f'Predicted boxes · score ≥ {args.score}'); ax.set_aspect('equal'); ax.grid(alpha=.2)
    seg = data['map'].astype(int)
    axes[2, 1].imshow(np.asarray(MAP_PALETTE)[seg].transpose(1, 0, 2)[::-1, ::-1], extent=(-15, 15, -30, 30))
    axes[2, 1].set_title('Predicted map · forward ↑')
    occ = data['occ'].astype(int)
    # Topmost nonempty voxel; this is a labeled 2D projection, not a 3D render.
    occupied = occ != OCC_CLASS_NAMES.index('empty')
    z = np.where(occupied, np.arange(occ.shape[2]), -1).max(axis=2)
    top = np.take_along_axis(occ, np.maximum(z, 0)[..., None], axis=2)[..., 0]
    top[z < 0] = OCC_CLASS_NAMES.index('empty')
    bound = 40 if frame['dataset_type'] == 'nuscenes' else 50
    axes[2, 2].imshow(np.asarray(OCC_PALETTE)[top][::-1, ::-1], extent=(-bound, bound, -bound, bound))
    axes[2, 2].set_title('Predicted occupancy · topmost voxel · forward ↑')
    for ax in axes[2]:
        ax.set_xlabel('Ego right (m)'); ax.set_ylabel('Ego forward (m)')
    fig.suptitle('Qwen BEV predictions · ' + ('experimental CARLA rig / out of distribution' if frame.get('source') == 'carla' else 'official demo') + '\nSeparate perception head; these outputs do not control the planner')
    fig.text(.01, .005, 'Map classes: ' + ', '.join(MAP_CLASS_NAMES), fontsize=8)
    fig.savefig(args.output, dpi=120)
    plt.close(fig)


if __name__ == '__main__':
    main()
