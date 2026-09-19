"""Measure single-scene direct planning, with explicit memory and image settings."""
import argparse
import json
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/Qwen-Drive-1.0-4B"))
    parser.add_argument("--upstream", type=Path, default=Path("vendor/qwen-drive"))
    parser.add_argument("--recording", type=Path)
    parser.add_argument("--index", type=int, default=15, help="Recording query index, at least 15")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--image-profile", choices=("native", "small"), default="small")
    parser.add_argument("--runs", type=int, default=2, help="First run is cold; remaining runs are warm")
    parser.add_argument("--attention", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--precision", choices=("bf16", "nf4"), default="bf16")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("runs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "started", "settings": {k: str(v) if isinstance(v, Path) else v
                                                for k, v in vars(args).items()},
              "mode": "direct_planning", "planner": "planner-sft", "num_samples": 1,
              "dtype": "bfloat16" if args.precision == "bf16" else "NF4 language layers; BF16 vision/planner",
              "runs": []}
    torch = None
    stage = "imports"
    try:
        import numpy as np
        import torch
        from qwen_drive import InferenceMode
        from .model_loading import load_model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run drive-check --torch outside a restricted sandbox")
        report.update(torch_version=torch.__version__, cuda_build=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(), gpu_free_before_bytes=torch.cuda.mem_get_info()[0])
        revision = args.model / "REVISION"
        report["model_revision"] = revision.read_text().strip() if revision.exists() else None
        torch.cuda.reset_peak_memory_stats()
        stage = "load_model"
        started = time.perf_counter()
        model = load_model(args.model, precision=args.precision, attention=args.attention)
        report["model_profile"] = model.drive_loading_info
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        report["load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        stage = "load_scene"
        reference = None
        if args.recording:
            from .adapter import driving_scene, ego_poses, load_records
            scene = driving_scene(args.recording, args.index)
            records = load_records(args.recording)
            future = records[args.index + 1:args.index + 51]
            if future:
                reference = ego_poses([r["pose"] for r in future], records[args.index]["pose"])
        else:
            from qwen_drive.benchmarks import read_scene_file
            from qwen_drive.images import ImageArchive
            demo = args.upstream / "data/demo"
            sample = next(read_scene_file(demo / "planning_scenes.jsonl",
                                         image_archive=ImageArchive.open(demo / "frames.parquet")))
            scene, reference = sample.scene, sample.future_trajectory
        if args.image_profile == "small":
            # Explicit demo sizes bypass the model's pixel budgets: override each frame.
            # This changes model inputs and is a feasibility setting, not benchmark parity.
            for frames in scene.views.values():
                for index, frame in enumerate(frames):
                    frame.target_size = (640, 384) if index == len(frames) - 1 else (320, 192)
        report["scene_token"] = scene.token
        report["image_sizes"] = [[f.target_size for f in frames] for frames in scene.views.values()]
        for index in range(args.runs):
            stage = f"inference_{index}"
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.run(InferenceMode.DIRECT_PLANNING, scene=scene, num_samples=1)
            torch.cuda.synchronize()
            trajectory = np.asarray(result.trajectories)
            if trajectory.shape != (1, 50, 3) or not np.isfinite(trajectory).all():
                raise RuntimeError(f"Invalid prediction: shape={trajectory.shape}")
            report["runs"].append(dict(seconds=time.perf_counter() - started, cold=index == 0,
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                       gpu_free_after_bytes=torch.cuda.mem_get_info()[0]))
            np.save(args.output / f"trajectories_{index}.npy", trajectory)
        stage = "plot"
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        # Top-down view: forward is up, ego-left is left on the page.
        ax.plot(-trajectory[0, :, 1], trajectory[0, :, 0], label="Qwen prediction")
        if reference is not None:
            ax.plot(-reference[:, 1], reference[:, 0], label="Recorded reference")
        ax.scatter([0], [0], color="black", label="Ego now")
        ax.set(xlabel="Ego right (m)", ylabel="Ego forward (m)", title="Offline trajectory comparison")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
        ax.grid(True)
        fig.savefig(args.output / "trajectory.png", dpi=150)
        plt.close(fig)
        report["status"] = "ok"
    except Exception as exc:
        report.update(status="failed", stage=stage, error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
        if torch is not None and torch.cuda.is_available():
            report["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
