"""Read-only environment diagnostics; no GPU package is required."""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "output": str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/environment.json"))
    parser.add_argument("--torch", action="store_true", help="Also initialize CUDA and run a matrix multiply")
    args = parser.parse_args()
    report = {
        "platform": platform.platform(), "python": platform.python_version(),
        "disk_free_gib": round(shutil.disk_usage(Path.cwd()).free / 2**30, 1),
        "memory": command(["free", "-h"]),
        "nvidia_smi": command(["nvidia-smi"]),
        "cuda_compiler": command(["nvcc", "--version"]),
        "vulkan": command(["vulkaninfo", "--summary"]),
        "packages": {},
    }
    for name in ("torch", "torchvision", "transformers", "carla", "qwen_drive"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    if args.torch:
        try:
            import torch
            report["torch"] = {"cuda_available": torch.cuda.is_available(), "cuda_build": torch.version.cuda}
            if torch.cuda.is_available():
                x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
                value = (x @ x)[0, 0].item()
                report["torch"].update(device=torch.cuda.get_device_name(),
                                       capability=torch.cuda.get_device_capability(),
                                       matrix_check=value == 64, free_bytes=torch.cuda.mem_get_info()[0])
        except Exception as exc:
            report["torch_error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
