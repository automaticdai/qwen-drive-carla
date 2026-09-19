# Higher-quality cloud model and local rendering

Three service profiles are available on the L4. `quality` uses the official
Qwen-Drive-1.0 4B backbone and SFT planning expert with unquantized BF16 weights
and larger images. `reasoning` uses the official reward-optimized RL expert in
its required reasoning-conditioned mode. `baseline` retains NF4 / SFT / small
images for comparison. These are compatible driving-model profiles, not swaps
to an unrelated larger language model. Published RL gains do not establish
better driving on our CARLA routes; see the [live comparison](quality-results.md).

| Setting | Earlier baseline | Quality profile |
| --- | --- | --- |
| Cloud weights | NF4 language layers | BF16 throughout |
| Planning expert | SFT | SFT (RL available separately) |
| Planning mode | Direct | Direct (reasoning available separately) |
| Current model image, each view | 640×384 | 1280×768 |
| Historical model image, each view | 320×192 | 640×384 |
| Local RGB camera capture | 640×384 | 1280×768 |
| CARLA graphics | Low | Epic |
| Visible window | 1280×720 | 2560×1440 |

Camera field of view, mounts, 10 Hz simulation and physics substeps are
unchanged. Higher pixels are captured by real sensors, not obtained by upscaling
old footage. Historical images are resized with PIL BICUBIC before upload,
matching the upstream PIL/torchvision target-size resize. PNG remains lossless
after this explicit downsample. The server advertises its image sizes, and the
client avoids unnecessary full-resolution history uploads. The request limit
is 64 MiB; supported image sizes remain bounded.

Epic enables the standard CARLA shadows, post-processing and longer drawing
distance. This does not verify hardware ray tracing, DLSS or a UE5 renderer.
The installed simulator remains CARLA 0.9.16 / UE4.26. Higher visual detail does
not itself improve physical dynamics or prove better planning.

## Activate

On the Google Cloud VM, in the deployed release directory:

```bash
cd ~/qwen-drive-20260919T050311Z-2343929
# Only if changing a running service profile:
tmux send-keys -t qwen-drive-agent C-c
# Wait for that session to exit before restarting.
bash scripts/start_remote_service.sh quality
```

`quality` is the launcher's default. Use `reasoning` to test BF16 / RL / reasoning /
high images. `baseline` selects NF4 / SFT / direct / small images for comparison. The two services are not loaded simultaneously.
Fresh cloud setup downloads the RL expert; existing deployments can run
`.venv/bin/python scripts/download_model.py --include-rl`. The pinned model
revision is unchanged. The service's health response records the active profile.

Start the idle Windows simulator from PowerShell:

```powershell
& '\\wsl.localhost\Ubuntu-24.04\home\yfrl\projects\qwen-drive-carla\scripts\start_carla_windows.ps1' `
  -CarlaRoot 'C:\Users\autom\AppData\Local\CARLA\0.9.16' `
  -Visible -Quality Epic -Width 2560 -Height 1440
```

The packaged Windows launcher opens its default map. Select Town01 through the
Python API before using the route below; the server must be idle when loading:

```bash
.venv/bin/python -c 'import carla; c=carla.Client("172.30.64.1",2000); c.set_timeout(60); c.load_world("Town01")'
```

With the [SSH tunnel](remote-inference.md) running, from local WSL:

```bash
.venv/bin/drive-run --host 172.30.64.1 \
  --planner-url http://127.0.0.1:8765 --planner-timeout 120 \
  --mode closed-loop --warmup-driver autopilot \
  --spawn-index 2 --destination-index 133 --seconds 25 --max-speed 3 \
  --camera-profile high --follow-camera --output outputs/quality-demo-NEW
```

The WSL Windows-host address can change after a reboot. Generic runner camera
profiles remain explicit so earlier small-camera experiments can be reproduced.
The scenario suite also accepts `--camera-profile high` and `--planner-url`.
Generated reasoning is retained in per-plan metrics as a model output, not an
independent explanation or proof that its trajectory is safe.

References: [official Qwen model and expert instructions](https://github.com/QwenLM/Qwen-Drive-1.0/blob/main/README.md),
[CARLA graphics quality](https://carla.readthedocs.io/en/0.9.16/adv_rendering_options/).
