"""Run the official Qwen BEV heads on calibrated CARLA or upstream demo frames."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/Qwen-Drive-1.0-4B'))
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from transformers import AutoTokenizer
    from qwen_drive import QwenDriveForPlanning
    from qwen_drive_perception import QwenDrivePerception
    from qwen_drive_perception.dataset import PerceptionFrame, PerceptionProcessor
    from qwen_drive_carla.bev import load_frame

    holder = QwenDriveForPlanning.from_pretrained(str(args.model), dtype=torch.bfloat16,
                                                attn_implementation='sdpa', local_files_only=True)
    vlm = holder.vlm
    del holder.planning_expert
    del holder
    model = QwenDrivePerception.from_pretrained(str(args.model / 'perception'),
                                              dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
    processor = PerceptionProcessor(AutoTokenizer.from_pretrained(str(args.model), local_files_only=True))
    model.attach(vlm.to('cuda').eval(), processor)
    paths = sorted(p for p in args.frames.iterdir() if (p / 'frame.json').is_file())[:args.limit]
    if not paths:
        raise ValueError('No packed perception frames found')
    for path in paths:
        source = json.loads((path / 'frame.json').read_text()).get('source', 'upstream_demo')
        frame = load_frame(path) if source == 'carla' else PerceptionFrame(path)
        inputs, metadata = processor(frame, device='cuda')
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.infer(inputs, metadata)
        torch.cuda.synchronize()
        result = {key: value.detach().float().cpu().numpy() if torch.is_tensor(value) else value
                  for key, value in result.items()}
        if any(not np.isfinite(v).all() for v in result.values()):
            raise RuntimeError('Non-finite BEV output')
        np.savez_compressed(args.output / (frame.token + '.npz'), **result)
        report = dict(token=frame.token, source=source, out_of_distribution=source == 'carla',
                      seconds=time.perf_counter() - started,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      shapes={key: list(value.shape) for key, value in result.items()},
                      camera_count=len(frame.cam_order), ground_truth_available=bool(frame.gt),
                      feeds_planner=False)
        (args.output / (frame.token + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
