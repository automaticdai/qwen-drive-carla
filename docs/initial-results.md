# Initial feasibility results — 2026-09-18

The standalone Qwen-Drive planning test succeeded on this machine. CARLA
recording and closed-loop driving have not been exercised.

| Check | Result |
| --- | --- |
| Host | WSL2, Ubuntu 24.04, Python 3.12.3 |
| GPU | RTX 5080, 16,303 MiB reported by NVIDIA |
| CUDA smoke test | BF16 matrix multiplication passed |
| Runtime | PyTorch 2.14.0+cu130, Transformers 5.14.1, SDPA |
| Model | Qwen-Drive-1.0-4B + planner-sft, BF16 |
| Inference | Direct planning, one trajectory, one upstream demo scene |
| Camera inputs | 3 views × 4 timestamps; history 320×192, current 640×384 |
| Model load | 6.91 seconds |
| Cold inference | 19.41 seconds |
| Second inference | 0.83 seconds |
| Peak allocated memory | 12,238,479,872 bytes (11.40 GiB) |
| Peak reserved memory | 12,773,752,832 bytes (11.90 GiB) |
| Output | Finite trajectory of shape (1, 50, 3), comparison plot generated |
| Unit tests | 9 passed |
| CARLA client | 0.9.16 installed; import passed |
| WSL Vulkan | Only llvmpipe CPU renderer detected |

These are measurements from two calls on the same scene, not a throughput or
driving-quality evaluation. Reduced image dimensions differ from the published
benchmark. No quantization, offloading, FlashAttention or native recurrent
acceleration extensions were used. The model alone fits this configuration;
concurrent CARLA rendering has not been tested and has little VRAM headroom.

Machine-readable report: `outputs/qwen-small-sdpa/report.json`.
Plot: `outputs/qwen-small-sdpa/trajectory.png`.
Environment report: `outputs/environment-qwen.json`.
Graphics diagnostic: `outputs/vulkan-summary.txt`.
These generated files are available locally and intentionally ignored by Git.

Source revision: `28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0`.
Model revision: `28484089a7cc8c335cf5089fb0745cf7c49b6eaa`.

## Next execution step

Install and launch the official Windows CARLA 0.9.16 package, then connect from
WSL using `drive-record`. The Linux server download was stopped and its partial
archive removed after the Vulkan probe found no hardware renderer. No Windows
CARLA installation, firewall changes, or WSL graphics changes were performed.
The README includes Windows startup and WSL connection instructions.
