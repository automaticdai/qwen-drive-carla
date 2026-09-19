"""Persistent Qwen inference and in-memory scene construction."""
import time
import numpy as np

from .adapter import scene_payload
from .model_loading import load_model


class QwenPlanner:
    def __init__(self, model_path, attention="sdpa", precision="bf16", planner="sft",
                 planning_mode="direct", image_profile="small"):
        if planner == "rl" and planning_mode != "reasoning":
            raise ValueError("The RL expert requires reasoning planning")
        if planning_mode not in ("direct", "reasoning") or image_profile not in ("small", "high"):
            raise ValueError("Unknown planning or image profile")
        import torch
        self.torch = torch
        self.precision = precision
        self.planning_mode = planning_mode
        self.image_sizes = [(320, 192)] * 3 + [(640, 384)] if image_profile == "small" else [(640, 384)] * 3 + [(1280, 768)]
        self.model = load_model(model_path, precision=precision, attention=attention, planner=planner)
        self.loading_info = {**self.model.drive_loading_info, "planning_mode": planning_mode,
                             "image_profile": image_profile, "image_sizes": self.image_sizes}

    def plan(self, records):
        return self.plan_payload(scene_payload(records, len(records) - 1))

    def plan_payload(self, payload):
        from qwen_drive import CameraFrame, DrivingScene, InferenceMode
        payload = dict(payload)
        payload["views"] = {tag: [CameraFrame(image, target_size=self.image_sizes[i])
                                   for i, image in enumerate(images)] for tag, images in payload["views"].items()}
        scene = DrivingScene(**payload)
        self.torch.cuda.reset_peak_memory_stats()
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            mode = InferenceMode.REASONING_PLANNING if self.planning_mode == "reasoning" else InferenceMode.DIRECT_PLANNING
            result = self.model.run(mode, scene=scene, num_samples=1)
        self.torch.cuda.synchronize()
        # Return raw predictions so the runner can retain rejected samples for
        # diagnosis. Validation happens before any proposal reaches control.
        points = np.asarray(result.trajectories[0])
        return points, dict(seconds=time.perf_counter() - started, precision=self.precision,
                            planning_mode=self.planning_mode, reasoning=result.reasoning,
                            peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
